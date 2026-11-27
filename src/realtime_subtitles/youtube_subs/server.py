"""
ARIA YouTube subtitle server.

Runs on http://127.0.0.1:8765. The browser userscript sends the video id and
the current playback time; the server downloads the video's audio once with
yt-dlp, then recognises (kotoba-whisper) and translates (NLLB) it in 30 s
windows, starting at the viewer's position and working ahead of playback.
Finished subtitles are cached in <project>/cache/youtube/<id>.json.

Run with:  python -m realtime_subtitles.youtube_subs.server
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

import numpy as np

from ..logger import info, warning, error, debug
from . import groq_client

# Wi-Fi mode (ARIA_LAN=1 or --lan): listen on the local network so a phone on the same
# Wi-Fi can connect. Requests from other devices must carry the access token (?k=...).
# On by default: phones/tablets reach the server over Wi-Fi (with the token) and over USB
# (adb reverse, loopback). ARIA_LAN=0 limits it to this PC and USB.
LAN_MODE = os.environ.get("ARIA_LAN", "1") != "0"
HOST = "0.0.0.0" if LAN_MODE else "127.0.0.1"
PORT = int(os.environ.get("ARIA_PORT", "8765"))
# Audio decoding for downloaded videos:
#   "window" (default) keep the downloaded file and decode only the window being recognised
#            (sample-identical to a whole decode; first subtitle ~10-30 s sooner on long videos)
#   "full"   decode the whole file into memory once, then slice windows from it
DECODE_MODE = "full" if os.environ.get("ARIA_DECODE_MODE") == "full" else "window"
FILE_IDLE_DELETE_SEC = 30 * 60   # window mode: delete a kept audio file unused this long
TOKEN_FILE = Path.home() / ".config" / "realtime-subtitles" / "mobile_token.txt"
CUE_WINDOW_BEHIND = 30.0    # with ?win=1 only cues near the viewer are sent
CUE_WINDOW_AHEAD = 300.0
# Subtitles only for phones/tablets: PC browsers get "disabled" and cost no recognition or
# translation quota. ARIA_ALLOW_DESKTOP=1 serves PC browsers again.
ALLOW_DESKTOP = os.environ.get("ARIA_ALLOW_DESKTOP") == "1"
MOBILE_UA_RE = re.compile(r"Android|iPhone|iPad|iPod|Mobile", re.I)
SAMPLE_RATE = 16000
BLOCK_SEC = 10 * 60
STREAM_READ_TIMEOUT_SEC = 40
STREAM_SEEK_GAP_SEC = 30
CUE_TIMING_VERSION = 1
WINDOW_SEC = 30.0          # audio recognised per step
LOOKBEHIND_SEC = 2.0       # start a little before the viewer's position
MAX_SENTENCE_CHARS = 40    # merge Whisper pieces into sentences up to this length
MERGE_GAP_SEC = 0.6        # pieces closer than this may belong to one sentence
# Translation quality vs. how far processing is ahead of the viewer:
FAST_AHEAD_SEC = 45.0      # less ready than this -> fast 600M translation
UPGRADE_AHEAD_SEC = 90.0   # more ready than this -> spend time re-translating with 1.3B
UPGRADE_BATCH = 4

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = Path(os.environ["ARIA_CACHE_DIR"]) if os.environ.get("ARIA_CACHE_DIR") \
    else PROJECT_ROOT / "cache" / "youtube"
TMP_DIR = CACHE_DIR / "tmp"
# ARIA_COLLECT=1: instead of making subtitles, save the first COLLECT_SEC of each stream the phone's
# player opens as a small audio file (for building a reference set; see /collect).
COLLECT = os.environ.get("ARIA_COLLECT") == "1"
COLLECT_SEC = int(os.environ.get("ARIA_COLLECT_SEC", "1500"))
COLLECT_DIR = Path(os.environ["ARIA_COLLECT_DIR"]) if os.environ.get("ARIA_COLLECT_DIR")     else PROJECT_ROOT / "cache" / "collect"
ASR_MODEL_DIR = MODELS_DIR / "kotoba-whisper-v2.0-faster"       # optional local fallback
NLLB_MODEL_DIR = MODELS_DIR / "nllb-200-distilled-600M-ct2-int8"  # optional local fallback

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
KEY_RE = re.compile(r"^(?:[A-Za-z0-9_-]{11}|u_[0-9a-f]{20})$")
USERSCRIPT_PATH = PROJECT_ROOT / "youtube" / "aria-youtube.user.js"


def resolve_source(page_url: str) -> Optional[tuple]:
    """Map a page URL to (job key, URL for yt-dlp). YouTube links use the video id as key."""
    try:
        u = urlparse(page_url)
    except ValueError:
        return None
    if u.scheme not in ("http", "https") or not u.netloc:
        return None
    host = u.netloc.lower().split(":")[0]
    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com") or host == "youtu.be":
        vid = None
        if host == "youtu.be":
            vid = u.path.strip("/").split("/")[0]
        elif u.path == "/watch":
            vid = (parse_qs(u.query).get("v") or [""])[0]
        else:
            m = re.match(r"^/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})", u.path)
            vid = m.group(1) if m else None
        if vid and VIDEO_ID_RE.match(vid):
            return vid, f"https://www.youtube.com/watch?v={vid}"
        return None
    clean = u._replace(fragment="").geturl()
    return "u_" + hashlib.sha1(clean.encode("utf-8")).hexdigest()[:20], clean
# Streams some players expose only as "<base>/playlist.m3u8" also offer a 480p variant
# at a fixed path; tried as an extra candidate for that CDN.
FIXED_VARIANT_HOST = "surrit.com"
# ffmpeg's default User-Agent ("Lavf/...") is refused by many video CDNs.
BROWSER_UA = "Mozilla/5.0 (Android 14; Mobile; rv:132.0) Gecko/132.0 Firefox/132.0"


def public_media_url(media: str) -> str:
    """Accept a player-observed media URL (HLS playlist or media file) on a public host.

    The phone script sends the stream its page is playing. Only http(s) URLs whose host
    resolves exclusively to public internet addresses are fetched, so the server cannot
    be pointed at this PC or other devices on the local network.
    """
    if not media or len(media) > 4096:
        return ""
    try:
        u = urlparse(media)
        port = u.port
    except ValueError:
        return ""
    if u.scheme not in ("http", "https") or not u.hostname:
        return ""
    try:
        infos = socket.getaddrinfo(u.hostname, port or (443 if u.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return ""
    for info_ in infos:
        try:
            if not ipaddress.ip_address(info_[4][0].split("%")[0]).is_global:
                return ""
        except ValueError:
            return ""
    return media if infos else ""


def page_referer(page: str) -> str:
    """The Referer a player on this page sends: the page origin."""
    try:
        u = urlparse(page)
    except ValueError:
        return ""
    return f"{u.scheme}://{u.netloc}/" if u.scheme in ("http", "https") and u.netloc else ""


# Japanese sentence endings: punctuation or common final forms.
SENTENCE_END_RE = re.compile(r"(?:[。．！？!?]|です|ます|ました|でした|ません|ください|たい|ない|よね|ね|よ|か|な|だ|た)$")
# Moans and breathing ("あっあっ", "んっ…", "はぁはぁ"): no subtitle, so nothing is sent for
# translation. Only lines made entirely of these sounds; "あっそこ気持ちいい" is translated.
MOAN_CHARS_RE = re.compile(
    r"^[あいうえおぁぃぅぇぉんっゃゅょはひふへほアイウエオァィゥェォンッャュョハヒフヘホ"
    r"ー〜～\s、。．,.!！?？…・♡♥]+$")
# Short words written only with those sounds that do carry meaning.
# Short words written only with those sounds that do carry meaning ("yes", "good", "no", "hey",
# "a lie"). Back-channel sounds such as うん/ええ/へえ are treated like moans: no subtitle, no
# translation request.
MOAN_KEEP = {"はい", "いい", "いえ", "おい", "うそ"}
# Korean side of the same thing (reference subtitles): "응", "음", "아아" ... carry no dialogue.
KO_FILLER_RE = re.compile(r"^[아어오우으응음흐하앗읏흡흑헉흠아앙읍]{1,4}$")


def is_filler_ko(text: str) -> bool:
    core = re.sub(r"[\s\-/.,!?~…♡♥*()]", "", text)
    return not core or bool(KO_FILLER_RE.match(core))


def is_moan(text: str) -> bool:
    core = re.sub(r"[\s、。．,.!！?？…・♡♥ー〜～っッ]", "", text)
    if not core:
        return True
    return core not in MOAN_KEEP and bool(MOAN_CHARS_RE.match(text))


PUNCT_END = "。！？!?…"
LONG_END_RE = re.compile(r"(?:です|ます|ました|でした|ません|ください|ですか|ますか)$")
PUNCT_SOFT = "、,，"


def words_to_lines(words: list, offset: float) -> list:
    """Word timestamps -> short subtitle lines [(start, end, text)] (about 6-20 characters each).

    Used instead of Groq's segment times, which in about a third of windows are placeholder
    values on a 2-second grid; word times are real. Lines end at sentence ends and pauses; an
    over-long line is cut after its last comma rather than inside a word, and a 1-2 character
    leftover ("ま。" of a drawn-out "ただ…いま") is joined to its neighbour."""
    lines, cur = [], []                      # cur: [(start, end, text), ...] of the open line

    def flush(ws):
        if ws:
            lines.append([ws[0][0], ws[-1][1], "".join(w[2] for w in ws)])

    for ws, we, w in words:
        w = w.strip()
        if not w or (len(w) >= 3 and is_moan(w)):   # moans get no subtitle (1-2 char words are particles: keep)
            continue
        a, b = offset + ws, offset + we
        if cur:
            gap = a - cur[-1][1]
            n = sum(len(x[2]) for x in cur)
            ended = SENTENCE_END_RE.search("".join(x[2] for x in cur))
            if gap > 0.45 or (n >= 10 and gap > 0.12) or (ended and n >= 5 and gap > 0.2):
                flush(cur)
                cur = []
            elif n + len(w) > 20:
                cut = max((i for i, x in enumerate(cur) if x[2][-1] in PUNCT_SOFT + PUNCT_END), default=None)
                if cut is not None and cut < len(cur) - 1:
                    flush(cur[:cut + 1])
                    cur = cur[cut + 1:]
                else:
                    flush(cur)
                    cur = []
        cur.append((a, b, w))
        text = "".join(x[2] for x in cur)
        # Immediate line end only on punctuation or an unmistakable ending (です/ます/...); a
        # one-character ending (だ, た, な ...) may be inside a word ("ただいま"), so it only ends
        # the line when a pause follows (checked above).
        if text[-1] in PUNCT_END or (text[-1] in PUNCT_SOFT and len(text) >= 8)                 or (len(text) >= 7 and LONG_END_RE.search(text)):
            flush(cur)
            cur = []
    flush(cur)
    merged = []
    for ln in lines:
        core = re.sub(r"[、。．,.!！?？…\s「」『』()（）]", "", ln[2])
        if not core:                                   # punctuation only
            continue
        if merged and len(core) <= 2 and ln[0] - merged[-1][1] < 2.5 and len(merged[-1][2]) < 24                 and merged[-1][2][-1] not in PUNCT_END:
            merged[-1][1], merged[-1][2] = ln[1], merged[-1][2] + ln[2]
        else:
            merged.append(ln)
    return [tuple(x) for x in merged]


# Split points inside one subtitle line: after sentence-ending punctuation.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。．！？!?])")


# --------------------------------------------------------------------------- speech detection

def _energy_regions(audio: np.ndarray, min_silence_ms: int, pad_ms: int) -> list:
    """Speech regions from loudness alone (numpy only), used when faster-whisper's
    neural detector isn't installed (Android). Roughly: frames louder than a threshold
    relative to the loudest parts, joined across short gaps."""
    frame = SAMPLE_RATE * 30 // 1000
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt((audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))
    thr = max(0.003, 0.08 * float(np.percentile(rms, 95)))
    loud = rms > thr
    regions, start = [], None
    gap_frames = max(1, min_silence_ms // 30)
    quiet = 0
    for i, flag in enumerate(loud):
        if flag:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap_frames:
                regions.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        regions.append((start, n - quiet))
    pad = SAMPLE_RATE * pad_ms // 1000
    return [{"start": max(0, a * frame - pad), "end": min(len(audio), b * frame + pad)}
            for a, b in regions if (b - a) * 30 >= 120]


def speech_regions(audio: np.ndarray, min_silence_ms: int = 500, pad_ms: int = 80) -> list:
    """[{"start": sample, "end": sample}, ...] of speech in 16 kHz float audio."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return _energy_regions(audio, min_silence_ms, pad_ms)
    return get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms))


# --------------------------------------------------------------------------- models

class Engines:
    """Lazily loaded recogniser + translator, shared by all jobs (one worker uses them)."""

    def __init__(self) -> None:
        self._asr = None
        self._fast = None   # NLLB 600M, beam 1: ~0.25 s per sentence
        self._hq = None     # NLLB 1.3B, beam 4: ~0.9 s per sentence (None if missing)
        self.target_language = "kor_Hang"
        # Groq cloud recognition/translation when API keys exist (ARIA_CLOUD=0 turns it off);
        # the local models above stay loaded as the fallback.
        from .groq_client import GroqPool
        self.groq = GroqPool()
        self.cloud = self.groq.available and os.environ.get("ARIA_CLOUD", "1") != "0"
        self.last_asr = self.last_mt = "local"

    @property
    def has_hq(self) -> bool:
        return self._hq is not None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:        # e.g. the Android (Termux) install: Groq only
            WhisperModel = None
        # Korean unless ARIA_TARGET_LANG says otherwise. (The old PC app's settings file
        # defaults to Chinese, which a fresh install such as the Android one would inherit.)
        self.target_language = os.environ.get("ARIA_TARGET_LANG", "kor_Hang")

        # The PC models are an optional fallback for when Groq can't answer. Without them
        # the server runs on Groq alone (much less memory and disk).
        if WhisperModel is not None and (ASR_MODEL_DIR / "model.bin").exists():
            self._load_local_asr(WhisperModel)
        else:
            info("YouTubeSubs: no local recognition model; Groq only")
        if (NLLB_MODEL_DIR / "model.bin").exists():
            self._load_local_mt()
        else:
            info("YouTubeSubs: no local translation model; Groq only "
                 "(Japanese text is shown when Groq can't translate)")
        if self._asr is None and not self.cloud:
            raise RuntimeError("PC 예비 모델도 Groq API 키도 없어 자막을 만들 수 없습니다")

    def _load_local_asr(self, WhisperModel) -> None:
        t0 = time.perf_counter()
        # Measured on this PC (i5-1335U, 12 threads): 10 threads was fastest for a 30 s
        # window (avg 15.6-16.3 s vs 17.3-17.9 s at 8). Override with ARIA_ASR_THREADS.
        try:
            asr_threads = int(os.environ.get("ARIA_ASR_THREADS", "10"))
        except ValueError:
            asr_threads = 10
        asr_threads = max(1, min(asr_threads, os.cpu_count() or 4))
        self._asr = WhisperModel(str(ASR_MODEL_DIR), device="cpu", compute_type="int8",
                                 cpu_threads=asr_threads)
        info(f"YouTubeSubs: kotoba-whisper using {asr_threads} CPU threads")
        info(f"YouTubeSubs: kotoba-whisper loaded in {time.perf_counter() - t0:.1f}s")

    def _load_local_mt(self) -> None:
        from ..translation.translator import create_translator, JA_KO_GLOSSARY_1_3B, JA_KO_GLOSSARY_600M
        korean = self.target_language == "kor_Hang"
        threads = min(8, os.cpu_count() or 4)
        self._fast = create_translator(
            engine="nllb", target_language=self.target_language, model_size="600m",
            beam_size=1, intra_threads=threads,
            glossary=JA_KO_GLOSSARY_600M if korean else None)
        try:
            self._hq = create_translator(
                engine="nllb", target_language=self.target_language, model_size="1.3b",
                beam_size=4, intra_threads=threads,
                glossary=JA_KO_GLOSSARY_1_3B if korean else None)
        except Exception as e:
            warning(f"YouTubeSubs: NLLB 1.3B unavailable ({e}); fast translation only")
            self._hq = None
        info(f"YouTubeSubs: translators ready (fast 600M{', hq 1.3B' if self._hq else ''}) "
             f"-> {self.target_language}")

    def transcribe(self, audio: np.ndarray):
        if self.cloud:
            from .groq_client import GroqUnavailable
            # No speech (music, silence): send nothing. Saves quota and avoids Whisper
            # inventing text over silence.
            if not speech_regions(audio, min_silence_ms=500):
                self.last_asr = "groq"
                return []
            try:
                segments = self.groq.transcribe(audio, words=True)
                self.last_asr = "groq"
                return segments
            except GroqUnavailable as e:
                if self._asr is None:
                    raise CloudUnavailable(str(e)) from e
                warning(f"YouTubeSubs: Groq recognition unavailable ({e}); using local model")
        if self._asr is None:
            raise CloudUnavailable("Groq를 쓸 수 없고 PC 예비 모델도 없습니다")
        segments, _ = self._asr.transcribe(
            audio, language="ja", beam_size=1, vad_filter=True,
            condition_on_previous_text=False)
        self.last_asr = "local"
        return list(segments)

    def translate_lines(self, lines: list, context: list) -> tuple:
        """Translate a window's subtitle lines. Returns (korean lines, quality tag per line).

        Groq translates them together with earlier lines as context ("llm"); any line it
        can't deliver (no key, limit, refusal) is translated locally with NLLB ("fast").
        """
        out = [None] * len(lines)
        if self.cloud and lines and self.target_language == "kor_Hang":
            from .groq_client import GroqUnavailable
            try:
                out = self.groq.translate(lines, context)
            except GroqUnavailable as e:
                warning(f"YouTubeSubs: Groq translation unavailable ({e}); using NLLB")
        tags = ["llm" if t else "fast" for t in out]
        for i, ja in enumerate(lines):
            if not out[i]:
                if self._fast is None:
                    # No PC translation model: show the Japanese line rather than nothing.
                    out[i], tags[i] = ja, "ja"
                    continue
                try:
                    out[i] = self.translate(ja, hq=False)
                except Exception as e:
                    warning(f"YouTubeSubs: translate failed: {e}")
                    out[i] = ""
        llm = tags.count("llm")
        model = "groq:" + (self.groq.last_model.split("/")[-1] or "?")
        self.last_mt = model if llm == len(lines) else ("nllb" if not llm else model + "+nllb")
        return out, tags

    def translate(self, text: str, hq: bool) -> str:
        tr = self._hq if (hq and self._hq is not None) else self._fast
        # NLLB tends to drop everything after the first sentence ("何?って考えると…" lost
        # its second half), so translate each sentence separately and join them.
        parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
        if len(parts) <= 1:
            return tr.translate(text, source_language="ja")
        return " ".join(t for t in (tr.translate(p, source_language="ja") for p in parts) if t)


# --------------------------------------------------------------------------- jobs

@dataclass
class Job:
    video_id: str                   # job key: YouTube id or "u_<hash>" for other sites
    status: str = "queued"          # queued | downloading | processing | done | error
    source_url: str = ""            # page URL given to yt-dlp
    message: str = ""
    duration: float = 0.0
    stream_input: str = ""
    viewer_off: bool = False        # the viewer turned subtitles off: no new work until polled
    media_mode: bool = False        # source_url is a stream the phone's player is playing
    referer: str = ""               # Referer header for that stream (the page origin)
    audio: Optional[np.ndarray] = None
    audio_path: Optional[Path] = None   # window decode mode: downloaded file kept on disk
    seek_shift: Optional[int] = None    # samples a seeked decode is late vs. a decode from 0
    stream_audio: Optional[StreamAudioBuffer] = field(default=None, repr=False)
    seek_audio: Optional[StreamAudioBuffer] = field(default=None, repr=False)
    window_audio: Optional[StreamAudioBuffer] = field(default=None, repr=False)
    cues: list = field(default_factory=list)       # [{"s","e","ja","ko"}], sorted by s
    covered: list = field(default_factory=list)    # merged [[a, b], ...] processed ranges
    want_pos: float = 0.0
    playing: bool = True            # viewer's video is playing in a visible tab
    playing_reported: bool = False  # client sent "playing" (userscript 2.6+)
    playback_anchor: Optional[float] = None
    stream_retry_at: float = 0.0
    restart_stream: bool = False
    work_range: Optional[tuple[float, float]] = None
    work_started_at: float = 0.0
    last_poll: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- coverage helpers
    def add_covered(self, a: float, b: float) -> None:
        ranges = sorted(self.covered + [[a, b]])
        merged: list = []
        for r in ranges:
            if merged and r[0] <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], r[1])
            else:
                merged.append(list(r))
        self.covered = merged

    def ahead(self) -> float:
        """Seconds of finished subtitles ahead of the viewer."""
        for a, b in self.covered:
            if a <= self.want_pos + 0.01 < b:
                return b - self.want_pos
        return 0.0

    def update_playback(self, position: float, now: float) -> None:
        elapsed = now - self.last_poll
        movement = position - self.want_pos
        if (self.playback_anchor is None or elapsed > 60
                or movement < -2 or movement > 5 + 2 * elapsed):
            self.playback_anchor = max(0.0, position - LOOKBEHIND_SEC)
            self.stream_retry_at = 0.0
        self.want_pos = position
        self.last_poll = now

    def block_bounds(self) -> tuple[float, float, float]:
        position = min(self.want_pos, max(0.0, self.duration - 0.001))
        start = int(position // BLOCK_SEC) * BLOCK_SEC
        return start, min(start + BLOCK_SEC, self.duration), min(start + 2 * BLOCK_SEC, self.duration)

    def next_start(self) -> Optional[float]:
        """Where to recognise next: one ten-minute block of spare ahead of the viewer.

        1. from the viewer to the end of the current ten-minute block,
        2. then the whole next block (the spare).
        When the viewer moves into the spare block, the block after it becomes the spare.
        The part of the block the viewer skipped (before a seek point) is not processed
        unless they go back to it, so no quota is spent on it.
        """
        if self.duration <= 0:
            return None
        pos = max(0.0, min(self.want_pos - LOOKBEHIND_SEC, self.duration))
        block_start, block_end, next_end = self.block_bounds()
        anchor = pos if self.playback_anchor is None else max(block_start, self.playback_anchor)
        for candidate, limit in ((anchor, block_end), (block_end, next_end)):
            for a, b in self.covered:
                if a <= candidate < b:
                    candidate = b
            if candidate < limit - 0.05:
                return candidate
        return None

    def progress(self) -> dict:
        start, end, next_end = self.block_bounds()
        blocks = []
        for a, b in ((start, end), (end, next_end)):
            if b > a:
                complete = sum(max(0, min(y, b) - max(x, a)) for x, y in self.covered)
                blocks.append({"start": a, "end": b, "completed": round(complete, 2)})
        return {"blocks": blocks, "working": self.work_range,
                "retry_in": math.ceil(max(0, self.stream_retry_at - time.monotonic()))}

    def to_json(self) -> dict:
        with self.lock:
            return {
                "video_id": self.video_id,
                "status": self.status,
                "message": self.message,
                "duration": round(self.duration, 2),
                "covered": [[round(a, 2), round(b, 2)] for a, b in self.covered],
                "cues": list(self.cues),
                "progress": self.progress(),
            }


class StreamReadError(RuntimeError):
    """A stream interruption that can be retried from the current subtitle position."""

class CloudUnavailable(RuntimeError):
    """Groq can't answer now (limits, internet) and there is no PC model: retry later."""


class StreamAccessError(RuntimeError):
    """The origin refused access to stream audio."""


class StreamAudioPending(Exception):
    """The sequential download is healthy but has not reached this window yet."""


class StreamAudioBuffer:
    """Retain a sequential PCM download on disk for subtitle windows and seeks."""

    def __init__(self, command: list[str], source: str, start: int, end: int) -> None:
        self.source = source
        self.start = start
        self.end = end
        self.audio = tempfile.TemporaryFile(mode="w+b", dir=TMP_DIR, buffering=0)
        self.size = 0
        self.changed = threading.Condition()
        self.denied_at: Optional[int] = None
        self.interrupted = False
        try:
            self._start(command)
        except BaseException:
            self.audio.close()
            raise

    def _start(self, command: list[str]) -> None:
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        bufsize=0, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.finished = False
        self.last_progress = time.monotonic()
        self.reader = threading.Thread(target=self._receive, daemon=True)
        self.diagnostics = threading.Thread(target=self._read_diagnostics, daemon=True)
        self.diagnostics.start()
        self.reader.start()

    def _read_diagnostics(self) -> None:
        with self.process.stderr as pipe:
            for line in pipe:
                if re.search(rb"\b(?:401 Unauthorized|403 Forbidden)\b", line, re.I):
                    with self.changed:
                        if self.denied_at is None:
                            self.denied_at = self.size & ~1
                        self.changed.notify_all()
                    if self.process.poll() is None:
                        self.process.kill()
                    break

    def _receive(self) -> None:
        try:
            with self.process.stdout as pipe:
                while block := pipe.read(64 * 1024):
                    with self.changed:
                        if self.denied_at is not None:
                            break
                        self.audio.seek(self.size)
                        self.size += self.audio.write(block)
                        self.last_progress = time.monotonic()
                        self.changed.notify_all()
        finally:
            self.process.wait()
            self.diagnostics.join()
            with self.changed:
                self.finished = True
                self.changed.notify_all()

    def read(self, start: int, end: int, allow_short_tail: bool) -> bytes:
        first, last = (start - self.start) * 2, (end - self.start) * 2
        with self.changed:
            self.changed.wait_for(lambda: self.size >= last or self.finished or self.denied_at is not None,
                                  timeout=1.0)
            if self.denied_at is not None and last > self.denied_at:
                raise StreamAccessError("영상 서버가 오디오 접근을 거부했습니다 (401/403).")
            available = self.size & ~1
            if available < last:
                if not self.finished:
                    if time.monotonic() - self.last_progress >= STREAM_READ_TIMEOUT_SEC:
                        raise TimeoutError("스트림에서 40초 동안 새 오디오를 받지 못했습니다")
                    raise StreamAudioPending()
                if not allow_short_tail or last - available > SAMPLE_RATE // 2 or available <= first:
                    raise RuntimeError("현재 구간 오디오를 끝까지 읽지 못했습니다")
            self.audio.seek(first)
            return self.audio.read(min(last, available) - first)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.reader.join()

    def restart(self, command: list[str]) -> None:
        self.stop()
        with self.changed:
            self.size &= ~1
            self.denied_at = None
            self.interrupted = False
        self._start(command)

    def close(self) -> None:
        self.stop()
        with self.changed:
            self.audio.close()

    def received_through(self) -> float:
        with self.changed:
            usable = self.size if self.denied_at is None else self.denied_at
            return (self.start + usable // 2) / SAMPLE_RATE


class JobManager:
    def __init__(self, engines: Engines) -> None:
        self.engines = engines
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.ready = threading.Event()
        self.load_error: Optional[str] = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True, name="YouTubeSubs_Worker").start()

    # ---- API
    def stop(self, video_id: str) -> bool:
        """Viewer turned subtitles off: the worker skips this job until it is polled again.

        The window being recognised right now still finishes; nothing further is started.
        """
        with self.lock:
            job = self.jobs.get(video_id)
            if job is None:
                return False
            with job.lock:
                # A flag, not an old last_poll: idle cleanup must not delete the kept
                # audio, so turning subtitles back on continues without a new download.
                job.viewer_off = True
                job.playing = False
        info(f"YouTubeSubs: {video_id} stopped by viewer (subtitles off)")
        return True

    def poll(self, video_id: str, t: float, source_url: str = "", media: str = "",
             playing: bool = True, playing_reported: bool = False, retry: bool = False,
             referer: str = "") -> dict:
        with self.lock:
            job = self.jobs.get(video_id)
            if job is None:
                job = self._load_cached(video_id) or Job(video_id)
                self.jobs[video_id] = job
            with job.lock:
                # The stream the phone is playing beats asking yt-dlp about the page: it
                # works on sites yt-dlp doesn't know. A new stream URL (players refresh
                # signed URLs) is taken while the job is waiting or has failed.
                if media and job.status in ("queued", "error") and media != job.source_url:
                    job.source_url = media
                    job.media_mode = True
                    job.referer = referer
                    job.stream_input = ""
                    job.status = "queued"
                    job.message = ""
                elif source_url and not job.source_url:
                    job.source_url = source_url
                if retry and job.status == "error":
                    job.status = "queued"
                    job.message = ""
                    job.stream_retry_at = 0.0
                    job.restart_stream = True
                job.viewer_off = False   # any poll means subtitles are on again
                job.update_playback(max(0.0, t), time.time())
                job.playing = playing
                job.playing_reported = playing_reported
                if job.status == "ready" and job.next_start() is not None:
                    job.status = "queued"
        self.wake.set()
        data = job.to_json()
        buffer = job.stream_audio
        working = data["progress"]["working"]
        if buffer is not None and buffer.source == job.stream_input:
            downloaded = buffer.received_through()
            data["progress"]["downloaded_through"] = round(downloaded, 1)
            if data["message"] == "전체 스트림 오디오 받는 중…":
                minute, second = divmod(int(downloaded), 60)
                pct = min(100, int(100 * downloaded / data["duration"])) if data["duration"] else 0
                data["message"] += f" {minute}:{second:02d} 확보 ({pct}%)"
        window_audio = job.window_audio
        if (data["status"] == "processing" and working and window_audio is not None
                and window_audio.source == job.stream_input
                and window_audio.start / SAMPLE_RATE <= working[0] + 0.01):
            data["progress"]["audio_received"] = round(
                max(0.0, min(working[1], window_audio.received_through()) - working[0]), 1)
        if not self.ready.is_set() and job.status not in ("done", "error"):
            data["message"] = self.load_error or "모델 불러오는 중..."
            if self.load_error:
                data["status"] = "error"
        return data

    # ---- cache
    def _cache_path(self, video_id: str) -> Path:
        return CACHE_DIR / f"{video_id}.json"

    def _load_cached(self, video_id: str) -> Optional[Job]:
        p = self._cache_path(video_id)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("cue_timing_version") != CUE_TIMING_VERSION:
                return None
            job = Job(video_id, status=d.get("status", "done"), duration=d.get("duration", 0.0),
                      cues=d.get("cues", []), covered=d.get("covered", []))
            if job.status != "done" or any(c.get("q") == "fast" for c in job.cues):
                job.status = "queued"  # resume: audio is downloaded again, covered parts skipped
            info(f"YouTubeSubs: loaded cache for {video_id} ({len(job.cues)} cues, {job.status})")
            return job
        except Exception as e:
            warning(f"YouTubeSubs: bad cache {p}: {e}")
            return None

    def _save(self, job: Job) -> None:
        d = job.to_json()
        d.pop("message", None)
        d.pop("progress", None)
        d["cue_timing_version"] = CUE_TIMING_VERSION
        tmp = self._cache_path(job.video_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._cache_path(job.video_id))

    # ---- worker
    def _pick_job(self) -> Optional[Job]:
        """The job that needs subtitles soonest (only one is processed at a time).

        A video that is playing in a visible tab beats paused or background tabs, which
        keep polling but don't need subtitles right now. Among those, the one with the
        least subtitles ready ahead of the viewer goes first; ties go to the latest poll.
        """
        with self.lock:
            now = time.time()
            retry_clock = time.monotonic()
            active = [j for j in self.jobs.values()
                      if j.status in ("queued", "downloading", "processing")
                      and now - j.last_poll < 60 and j.stream_retry_at <= retry_clock
                      and not j.viewer_off]
            if not active:
                return None

            def priority(j: Job) -> tuple:
                ahead = j.ahead() if j.duration > 0 else 0.0
                return (0 if j.playing else 1, ahead, -j.last_poll)

            return min(active, key=priority)

    def _worker(self) -> None:
        try:
            self.engines.load()
        except Exception as e:
            self.load_error = f"모델을 불러오지 못했습니다: {e}"
            error(f"YouTubeSubs: {self.load_error}")
            return
        self.ready.set()
        next_cleanup = 0.0
        while True:
            now = time.monotonic()
            if now >= next_cleanup:
                self._release_idle()
                next_cleanup = now + 10
            job = self._pick_job()
            if job is None:
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            try:
                if job.media_mode:
                    if not job.stream_input:
                        self._open_stream(job)
                    self._step(job)
                    continue
                # Audio is needed only while there is unrecognised audio left.
                has_audio = job.audio is not None or (
                    job.audio_path is not None and job.audio_path.exists())
                if job.duration <= 0 or (not has_audio and job.next_start() is not None):
                    self._download(job)
                    continue
                self._step(job)
            except StreamAudioPending:
                continue
            except CloudUnavailable as e:
                warning(f"YouTubeSubs: {job.video_id}: {e}. Retrying in 30s.")
                with job.lock:
                    job.stream_retry_at = time.monotonic() + 30
                    job.work_range = None
                    job.message = "Groq를 쓸 수 없음 (한도 또는 인터넷). 30초 후 다시 시도…"
            except StreamAccessError as e:
                warning(f"YouTubeSubs: {job.video_id}: stream audio access denied")
                with job.lock:
                    job.status = "error"
                    job.message = str(e)
            except StreamReadError as e:
                warning(f"YouTubeSubs: {job.video_id}: {e}. Retrying stream audio in 10s.")
                with job.lock:
                    job.stream_retry_at = time.monotonic() + 10
                    job.message = "스트림 연결이 지연되어 다시 시도하는 중…"
            except Exception as e:
                error(f"YouTubeSubs: {job.video_id}: {e}")
                with job.lock:
                    job.status = "error"
                    job.message = str(e)

    def _release_idle(self) -> None:
        """Free decoded audio of videos nobody has watched for a while."""
        stale_streams = []
        stale_files = []
        now = time.time()
        with self.lock:
            for j in self.jobs.values():
                # A kept file costs disk, not memory: keep it longer so a viewer who comes
                # back (screen was off) needs no new download.
                if j.audio_path is not None and now - j.last_poll > FILE_IDLE_DELETE_SEC:
                    stale_files.append(j.audio_path)
                    j.audio_path = None
                    if j.status == "processing":
                        j.status = "queued"
                if now - j.last_poll <= 300:
                    continue
                if j.stream_audio is not None:
                    stale_streams.append(j.stream_audio)
                    j.stream_audio = None
                if j.seek_audio is not None:
                    stale_streams.append(j.seek_audio)
                    j.seek_audio = None
                j.window_audio = None
                if j.audio is not None:
                    j.audio = None
                    if j.status == "processing":
                        j.status = "queued"
        for buffer in stale_streams:
            buffer.close()
        for path in stale_files:
            path.unlink(missing_ok=True)

    @staticmethod
    def _download_error(output: str) -> str:
        """Turn yt-dlp output into a short Korean message for the overlay."""
        low = output.lower()
        if any(reason in low for reason in ("connectionreseterror", "10054", "connection reset")):
            return "영상 페이지에 연결하지 못했습니다. 브라우저의 스트림 주소를 확인합니다."
        if "unsupported url" in low:
            return "이 사이트는 지원되지 않습니다 (yt-dlp가 영상을 찾지 못함)"
        if "drm" in low:
            return "DRM으로 보호된 영상이라 자막을 만들 수 없습니다"
        if "is_live" in low or "live event" in low or "is live" in low:
            return "생방송은 지원하지 않습니다"
        if "sign in" in low or "login" in low or "cookies" in low or "private" in low:
            return "로그인이 필요한 영상이라 받을 수 없습니다"
        last = [l for l in output.strip().splitlines() if l.strip()]
        return "음성 다운로드 실패: " + (last[-1][:160] if last else "?")

    def _download(self, job: Job) -> None:
        with job.lock:
            job.status = "downloading"
            job.message = "음성 받는 중..."
        ytdlp = shutil.which("yt-dlp")
        ffmpeg = shutil.which("ffmpeg")
        if not ytdlp or not ffmpeg:
            raise RuntimeError("yt-dlp 또는 ffmpeg를 찾을 수 없습니다 (scoop install yt-dlp ffmpeg)")
        for old in TMP_DIR.glob(f"{job.video_id}.*"):
            old.unlink(missing_ok=True)
        source = job.source_url or f"https://www.youtube.com/watch?v={job.video_id}"
        cmd = [ytdlp, "--no-playlist", "--no-progress", "--http-chunk-size", "10M",
               # audio only when the site offers it; otherwise the smallest full file
               "-f", "worstaudio[acodec=opus]/worstaudio/bestaudio/worst/best",
               "--match-filter", "!is_live",
               "-o", str(TMP_DIR / f"{job.video_id}.%(ext)s"),
               "--print", "after_move:filepath"]
        if shutil.which("node"):
            cmd += ["--js-runtimes", "node"]
        cmd += ["--", source]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        path = Path(lines[-1].strip()) if lines else None
        if r.returncode != 0 or not path or not path.exists():
            raise RuntimeError(self._download_error(r.stdout + "\n" + r.stderr))
        info(f"YouTubeSubs: {job.video_id} audio downloaded in {time.perf_counter() - t0:.1f}s")

        if DECODE_MODE == "window":
            duration = self._file_duration(path)
            shift = self._calibrate_seek(path) if duration > 15 else None
            if shift is None:
                warning(f"YouTubeSubs: {job.video_id}: seeking is not sample-exact for this file; "
                        f"decoding it whole instead")
            else:
                job.seek_shift = shift
        if DECODE_MODE == "window" and job.seek_shift is not None:
            with job.lock:
                job.audio_path = path
                job.audio = None
                job.duration = duration
                job.status = "processing"
                job.message = ""
            info(f"YouTubeSubs: {job.video_id} duration {duration / 60:.1f} min (window decode)")
            return

        dec = subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
                              "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
                             capture_output=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
        path.unlink(missing_ok=True)  # keep nothing on disk but the subtitles
        if dec.returncode != 0 or not dec.stdout:
            raise RuntimeError("음성 디코딩 실패: " + dec.stderr.decode("utf-8", "replace")[-200:])
        audio = np.frombuffer(dec.stdout, dtype=np.int16)
        with job.lock:
            job.audio = audio
            job.duration = len(audio) / SAMPLE_RATE
            job.status = "processing"
            job.message = ""
        info(f"YouTubeSubs: {job.video_id} duration {job.duration / 60:.1f} min")

    @staticmethod
    def _file_duration(path: Path) -> float:
        probe = shutil.which("ffprobe")
        if not probe:
            raise RuntimeError("ffprobe를 찾을 수 없습니다")
        r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            duration = float(r.stdout.strip())
        except ValueError:
            duration = 0.0
        if not (0 < duration < float("inf")):
            raise RuntimeError("음성 파일 길이를 확인하지 못했습니다")
        return duration

    def _decode_pcm(self, path: Path, seek_sec: Optional[float], length_sec: float) -> np.ndarray:
        seek = ["-ss", f"{seek_sec:.3f}"] if seek_sec else []
        dec = subprocess.run(
            [self._ffmpeg_executable(), "-nostdin", "-loglevel", "error",
             *seek, "-i", str(path), "-t", f"{length_sec:.3f}",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if dec.returncode != 0:
            raise RuntimeError("음성 디코딩 실패: " + dec.stderr.decode("utf-8", "replace")[-200:])
        return np.frombuffer(dec.stdout[: len(dec.stdout) & ~1], dtype=np.int16)

    # A seeked ffmpeg decode can start a few samples off the timeline of a decode from the
    # file start (16 samples for YouTube opus). Whisper output changed with that 1 ms shift,
    # so each file is calibrated once and window audio is made sample-identical to the
    # whole-file decode; files where that cannot be shown are decoded whole instead.
    SEEK_PREROLL_SEC = 0.5

    def _calibrate_seek(self, path: Path) -> Optional[int]:
        try:
            ref = self._decode_pcm(path, None, 12.0)
            seek_ms = 4500
            win = self._decode_pcm(path, seek_ms / 1000, 6.0)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            return None
        a, n = 5 * SAMPLE_RATE, 3 * SAMPLE_RATE
        target = ref[a:a + n]
        if len(target) < n or not target.any():
            return None
        base = a - seek_ms * SAMPLE_RATE // 1000
        for shift in sorted(range(-400, 401), key=abs):
            if 0 <= base + shift and np.array_equal(win[base + shift:base + shift + n], target):
                return shift
        return None

    def _decode_file_window(self, job: Job, start: float, end: float) -> np.ndarray:
        """Decode only [start, end) of the kept audio file, sample-identical to a whole decode."""
        path = job.audio_path
        if path is None or not path.exists():
            # The kept file was removed (idle cleanup): the worker downloads it again.
            with job.lock:
                job.audio_path = None
                job.status = "queued"
                job.work_range = None
            raise StreamAudioPending()
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        seek_ms = max(0, math.floor((start - self.SEEK_PREROLL_SEC) * 1000))
        if seek_ms == 0:
            pcm = self._decode_pcm(path, None, end + 0.1)
            return pcm[a:b]
        pcm = self._decode_pcm(path, seek_ms / 1000, end - seek_ms / 1000 + 0.1)
        first = a - seek_ms * SAMPLE_RATE // 1000 + (job.seek_shift or 0)
        return pcm[first:first + (b - a)]

    @staticmethod
    def _is_hls(url: str) -> bool:
        return urlparse(url).path.lower().endswith(".m3u8")

    @classmethod
    def _media_input(cls, url: str, referer: str) -> list[str]:
        """ffmpeg input options for a stream the phone's player is playing."""
        opts = ["-rw_timeout", "8000000", "-user_agent", BROWSER_UA]
        if referer:
            opts += ["-referer", referer]
        if cls._is_hls(url):
            # Some CDNs disguise TS segments with other extensions.
            opts += ["-allowed_segment_extensions", "ALL", "-extension_picky", "0"]
        return opts + ["-i", url]

    @staticmethod
    def _master_variants(master_url: str, playlist: str) -> list[tuple[str, bool]]:
        base = urlparse(master_url)
        directory = base.path.rsplit("/", 1)[0] + "/"
        variants = []
        attributes = ""
        for line in playlist.splitlines():
            line = line.strip()
            if line.startswith("#EXT-X-STREAM-INF:"):
                attributes = line.split(":", 1)[1]
            elif attributes and line and not line.startswith("#"):
                url = urljoin(master_url, line)
                target = urlparse(url)
                bandwidth = re.search(r"(?:^|,)BANDWIDTH=(\d+)(?:,|$)", attributes)
                codecs = re.search(r'(?:^|,)CODECS="([^"]+)"', attributes)
                audio_group = re.search(r'(?:^|,)AUDIO="[^"]+"', attributes)
                attributes = ""
                # A variant whose audio lives in a separate rendition (AUDIO group) has
                # no sound of its own; the master playlist (always a candidate) covers it.
                if (bandwidth and not audio_group and target.scheme in ("http", "https")
                        and target.path.lower().endswith(".m3u8")
                        and (target.netloc == base.netloc or public_media_url(url))):
                    has_aac = bool(codecs and any(
                        c.strip().startswith("mp4a.") for c in codecs[1].split(",")))
                    variants.append((int(bandwidth[1]), url, has_aac))
        return [(url, has_aac) for _, url, has_aac in sorted(variants)]

    def _playlist_text(self, url: str, referer: str) -> str:
        if not self._is_hls(url):
            return ""
        try:
            response = subprocess.run(
                [self._ffmpeg_executable(), "-nostdin", "-v", "error", "-rw_timeout", "8000000",
                 "-user_agent", BROWSER_UA, *(["-referer", referer] if referer else []),
                 "-f", "data", "-i", url,
                 "-map", "0:0", "-c", "copy", "-f", "data", "-"],
                capture_output=True, timeout=12,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
            return ""
        return response.stdout.decode("utf-8-sig", "replace") if response.returncode == 0 else ""

    @staticmethod
    def _playlist_duration(playlist: str) -> float:
        lines = [line.strip() for line in playlist.splitlines() if line.strip()]
        if not lines or lines[0] != "#EXTM3U" or "#EXT-X-ENDLIST" not in lines:
            return 0.0
        durations = [line.partition(":")[2].partition(",")[0]
                     for line in lines if line.startswith("#EXTINF:")]
        try:
            values = [float(value) for value in durations]
            if any(not math.isfinite(value) or value < 0 for value in values):
                return 0.0
            return math.fsum(values)
        except (ValueError, OverflowError):
            return 0.0

    def _stream_candidates(self, master_url: str, referer: str) -> dict[str, bool]:
        """Inputs to try, cheapest first: low-bandwidth variants, then the URL itself."""
        if not self._is_hls(master_url):
            return {master_url: False}   # a plain media file (mp4, webm, ...)
        candidates = dict(self._master_variants(master_url, self._playlist_text(master_url, referer)))
        if urlparse(master_url).hostname in (FIXED_VARIANT_HOST, "www." + FIXED_VARIANT_HOST):
            candidates.setdefault(master_url.rsplit("/", 1)[0] + "/480p/video.m3u8", False)
        candidates.setdefault(master_url, False)
        return candidates

    def _open_stream(self, job: Job) -> None:
        started = time.perf_counter()
        probe = shutil.which("ffprobe")
        with job.lock:
            job.status = "downloading"
            job.message = "영상 길이 확인 중…"
        referer = job.referer
        for url, has_aac in self._stream_candidates(job.source_url, referer).items():
            # A complete media playlist already provides duration. Probing its media
            # would download initial segments again before the decoder can start.
            duration = self._playlist_duration(self._playlist_text(url, referer)) if has_aac else 0.0
            method = "playlist"
            if not duration:
                if not probe:
                    raise RuntimeError("ffprobe를 찾을 수 없습니다")
                method = "ffprobe"
                try:
                    result = subprocess.run(
                        [probe, "-v", "error", *self._media_input(url, referer), "-show_entries",
                         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1"],
                        capture_output=True, timeout=25,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    duration = float(result.stdout.strip()) if result.returncode == 0 else 0.0
                except (ValueError, subprocess.TimeoutExpired):
                    continue
            if 0 < duration < float("inf"):
                with job.lock:
                    job.duration = duration
                    job.stream_input = url
                    job.message = ""
                    job.status = "processing"
                parts = urlparse(url).path.split("/")
                variant = parts[-2] if len(parts) > 1 else ""
                kind = ("HLS variant " + (variant if re.fullmatch(r"[0-9]+p", variant) else "playlist")
                        if self._is_hls(url) else "media file")
                info(f"YouTubeSubs: {job.video_id} selected {kind} "
                     f"in {time.perf_counter() - started:.1f}s ({method}), {duration / 60:.1f} min")
                return
        raise RuntimeError("재생 중인 영상 스트림을 열지 못했습니다 (길이 확인 실패). "
                           "사이트가 외부 접속을 막고 있을 수 있습니다.")

    def _close_streams(self) -> None:
        with self.lock:
            buffers = [buffer for job in self.jobs.values()
                       for buffer in (job.stream_audio, job.seek_audio) if buffer is not None]
        for buffer in buffers:
            buffer.close()

    def _stream_buffer(self, job: Job, first: int, last: int, duration: int,
                       restart: bool) -> StreamAudioBuffer:
        primary = job.stream_audio
        seek = job.seek_audio
        downloaded = primary.received_through() if primary else 0.0
        if primary and downloaded >= last / SAMPLE_RATE:
            return primary
        if seek and seek.start <= first < last <= seek.end:
            return seek
        if first / SAMPLE_RATE >= downloaded + STREAM_SEEK_GAP_SEC:
            # Keep the main sequential download, but jump ahead for an uncached seek.
            # A known access denial remains an error until the user chooses Retry.
            if not restart and any(buffer and buffer.denied_at is not None for buffer in (primary, seek)):
                raise StreamAccessError("영상 서버가 오디오 접근을 거부했습니다 (401/403).")
            if seek:
                seek.close()
                job.seek_audio = None
            block_samples = BLOCK_SEC * SAMPLE_RATE
            limit = min(duration, max(last, (first // block_samples + 1) * block_samples))
            seek = StreamAudioBuffer(self._stream_command(job.stream_input, first, limit, job.referer),
                                     job.stream_input, first, limit)
            job.seek_audio = seek
            info(f"YouTubeSubs: {job.video_id} started seek audio at {first / SAMPLE_RATE:.2f}s")
            return seek
        if primary is None:
            primary = StreamAudioBuffer(self._stream_command(job.stream_input, 0, duration, job.referer),
                                        job.stream_input, 0, duration)
            job.stream_audio = primary
            info(f"YouTubeSubs: {job.video_id} started whole-stream audio ({job.duration:.1f}s)")
        return primary

    def _read_stream_window(self, job: Job, start: float, end: float) -> np.ndarray:
        first, last = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        duration = int(job.duration * SAMPLE_RATE)
        with job.lock:
            restart = job.restart_stream
            job.restart_stream = False
            job.status = "processing"
            job.message = "스트림 오디오 받는 중…"
        buffer = self._stream_buffer(job, first, last, duration, restart)
        with job.lock:
            job.window_audio = buffer
            job.message = ("재생 위치 오디오 받는 중…" if buffer.start
                           else "전체 스트림 오디오 받는 중…")
        should_resume = buffer.interrupted or (restart and buffer.denied_at is not None)
        if should_resume and buffer.received_through() < end:
            offset = int(round(buffer.received_through() * SAMPLE_RATE))
            buffer.restart(self._stream_command(job.stream_input, offset, buffer.end, job.referer))
            info(f"YouTubeSubs: {job.video_id} resumed stream audio at {offset / SAMPLE_RATE:.2f}s")
        try:
            pcm = buffer.read(first, last, allow_short_tail=last >= duration - SAMPLE_RATE // 20)
        except StreamAccessError:
            raise
        except (TimeoutError, RuntimeError) as e:
            buffer.stop()
            buffer.interrupted = True
            raise StreamReadError(str(e)) from e
        with job.lock:
            job.message = "일본어 인식 및 한국어 번역 중…"
        return np.frombuffer(pcm, dtype=np.int16)

    @staticmethod
    def _ffmpeg_executable() -> str:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg를 찾을 수 없습니다")
        # Killing a Scoop shim can leave the underlying FFmpeg process running.
        shim = Path(ffmpeg).with_suffix(".shim")
        if shim.is_file():
            match = re.search(r'^path\s*=\s*"([^"]+)"', shim.read_text(encoding="utf-8-sig"), re.M)
            if match and Path(match[1]).is_file():
                ffmpeg = match[1]
        return ffmpeg

    def _stream_command(self, url: str, start: int, end: int, referer: str = "") -> list[str]:
        seek = ["-ss", str(start / SAMPLE_RATE)] if start else []
        return [self._ffmpeg_executable(), "-nostdin", "-loglevel", "warning", *seek,
                *self._media_input(url, referer),
                "-t", str((end - start) / SAMPLE_RATE), "-map", "0:a:0",
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]

    def _step(self, job: Job) -> None:
        """Do one unit of work: recognise a window, or upgrade some fast translations."""
        hq = self.engines.has_hq
        ahead = job.ahead()
        if hq and ahead >= UPGRADE_AHEAD_SEC and self._upgrade(job, near_viewer=True):
            return
        start = job.next_start()
        if start is None:
            if hq and self._upgrade(job, near_viewer=False):
                return
            with job.lock:
                complete = bool(len(job.covered) == 1 and job.covered[0][0] <= 0.05
                                and job.covered[0][1] >= job.duration - 0.05)
                job.status = ("done" if complete else "ready") if job.next_start() is None else "processing"
                job.message = ""
                job.work_range = None
                job.window_audio = None
                if job.status == "done":
                    job.audio = None
                    if job.audio_path is not None:
                        job.audio_path.unlink(missing_ok=True)
                        job.audio_path = None
            self._save(job)
            info(f"YouTubeSubs: {job.video_id} {job.status} ({len(job.cues)} cues)")
            return
        self._process_window(job, start, use_hq=hq and ahead >= FAST_AHEAD_SEC)

    def _upgrade(self, job: Job, near_viewer: bool) -> bool:
        """Re-translate a few fast (600M) cues with the 1.3B model. Returns False if none left."""
        with job.lock:
            todo = [c for c in job.cues if c.get("q") == "fast"
                    and (not near_viewer or c["s"] >= job.want_pos - 1.0)]
            todo = sorted(todo, key=lambda c: (c["s"] < job.want_pos - 1.0, c["s"]))[:UPGRADE_BATCH]
        if not todo:
            return False
        t0 = time.perf_counter()
        done = {}
        for c in todo:
            try:
                done[(c["s"], c["ja"])] = self.engines.translate(c["ja"], hq=True)
            except Exception as e:
                warning(f"YouTubeSubs: upgrade failed: {e}")
                done[(c["s"], c["ja"])] = None
        with job.lock:
            for c in job.cues:
                key = (c["s"], c["ja"])
                if key in done:
                    if done[key]:
                        c["ko"] = done[key]
                    c["q"] = "hq"
        self._save(job)
        debug(f"YouTubeSubs: upgraded {len(todo)} cues in {time.perf_counter() - t0:.1f}s")
        return True

    def _process_window(self, job: Job, start: float, use_hq: bool) -> None:
        anchor = job.playback_anchor
        first_window = anchor is None or start <= anchor + 0.05
        window = 12.0 if job.stream_input and first_window else WINDOW_SEC
        end = min(start + window, job.duration)
        block_end = (int(start // BLOCK_SEC) + 1) * BLOCK_SEC
        end = min(end, block_end)
        # Don't re-process audio that a previous run already covered.
        for a, _b in job.covered:
            if start < a < end:
                end = a
        is_last = end >= job.duration - 0.05
        work_range = (round(start, 2), round(end, 2))
        with job.lock:
            job.status = "processing"
            if job.work_range != work_range:
                job.work_range = work_range
                job.work_started_at = time.perf_counter()
                job.window_audio = None
            t_fetch = job.work_started_at
            job.message = "일본어 인식 및 한국어 번역 중…"
        a_i, b_i = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        if job.stream_input:
            audio = self._read_stream_window(job, start, end)
        elif job.audio is None and job.audio_path is not None:
            audio = self._decode_file_window(job, start, end)
        else:
            audio = job.audio[a_i:b_i]
        fetch_t = time.perf_counter() - t_fetch
        chunk = audio.astype(np.float32) / 32768.0

        t0 = time.perf_counter()
        # Zero-length pieces are Whisper repetition hallucinations.
        segs = [s for s in self.engines.transcribe(chunk)
                if s.text.strip() and s.end - s.start >= 0.2]
        asr_t = time.perf_counter() - t0

        # Choose where this window really ends: after the last finished sentence,
        # so the next window starts at a sentence boundary.
        cut = end
        if not is_last and segs and end - start > 5:
            win = end - start
            final_idx = [i for i, s in enumerate(segs)
                         if SENTENCE_END_RE.search(s.text.strip()) and s.end > win * 0.4]
            if final_idx:
                keep = final_idx[-1] + 1
            else:
                keep = len(segs) - 1 if segs[-1].end > win - 1.5 else len(segs)
                keep = max(keep, 1)
            if keep < len(segs):
                cut = start + segs[keep].start
                segs = segs[:keep]
            else:
                cut = start + max(segs[-1].end, 0.0) if segs[-1].end > win - 1.5 else end
            cut = min(end, max(cut, start + 2.0))

        # Lines from word times when Groq gave them (its segment times are often placeholders).
        words = [w for s in segs for w in (getattr(s, "words", None) or [])]
        sentences = words_to_lines(words, start) if words else self._merge(segs, start)
        if sentences:
            sentences = self._align_cue_starts(sentences, chunk, start)
        with job.lock:
            before = [c for c in job.cues if c["s"] < start]
            prev_ja = before[-1]["ja"] if before else None
            context = [(c["ja"], c.get("ko", "")) for c in before[-3:]]
        t1 = time.perf_counter()
        kept = []
        moans = 0
        for s_start, s_end, ja in sentences:
            if ja == prev_ja:  # Whisper repeated the previous sentence
                continue
            prev_ja = ja
            if is_moan(ja):    # no subtitle and no translation request for moans
                moans += 1
                continue
            kept.append((s_start, s_end, ja))
        if use_hq and not self.engines.cloud:
            kos, tags = [], []
            for _, _, ja in kept:
                try:
                    kos.append(self.engines.translate(ja, hq=True))
                except Exception as e:
                    warning(f"YouTubeSubs: translate failed: {e}")
                    kos.append("")
                tags.append("hq")
        else:
            kos, tags = self.engines.translate_lines([ja for _, _, ja in kept], context)
        new_cues = [{"s": round(s_start, 2), "e": round(s_end, 2), "ja": ja, "ko": ko, "q": q}
                    for (s_start, s_end, ja), ko, q in zip(kept, kos, tags)]
        tr_t = time.perf_counter() - t1

        with job.lock:
            job.cues = sorted(
                [c for c in job.cues if not (start - 0.01 <= c["s"] < cut)] + new_cues,
                key=lambda c: c["s"])
            job.add_covered(start, cut)
            job.message = ""
            job.work_range = None
            job.window_audio = None
        self._save(job)
        span = cut - start
        info(f"YouTubeSubs: {job.video_id} {start:7.1f}-{cut:7.1f}s: {len(new_cues)} cues"
             f"{f' (+{moans} moans skipped)' if moans else ''} "
             f"(asr {self.engines.last_asr}, mt {self.engines.last_mt}), "
             f"audio wait {fetch_t:.1f}s + asr {asr_t:.1f}s + translate {tr_t:.1f}s for {span:.0f}s audio "
             f"({span / max(fetch_t + asr_t + tr_t, 0.01):.1f}x realtime)")

    @staticmethod
    def _align_cue_starts(sentences, audio: np.ndarray, offset: float) -> list:
        speech = speech_regions(audio, min_silence_ms=350, pad_ms=80)
        intervals = [(offset + span["start"] / SAMPLE_RATE,
                      offset + span["end"] / SAMPLE_RATE) for span in speech]
        aligned = []
        for start, end, text in sentences:
            for speech_start, speech_end in intervals:
                if speech_start >= end:
                    break
                if min(end, speech_end) - max(start, speech_start) >= 0.4:
                    start = max(start, speech_start)
                    break
            aligned.append((start, end, text))
        return aligned

    @staticmethod
    def _merge(segs, offset: float) -> list:
        """Join Whisper pieces into sentences: [(start, end, text)]."""
        out = []
        cur = None
        for s in segs:
            text = s.text.strip()
            a, b = offset + s.start, offset + s.end
            if cur is not None:
                gap = a - cur[1]
                if (gap > MERGE_GAP_SEC or SENTENCE_END_RE.search(cur[2])
                        or len(cur[2]) + len(text) > MAX_SENTENCE_CHARS):
                    out.append(tuple(cur))
                    cur = None
            if cur is None:
                cur = [a, b, text]
            else:
                cur[1] = b
                cur[2] += text
        if cur is not None:
            out.append(tuple(cur))
        return out


# --------------------------------------------------------------------------- HTTP

def load_access_token() -> str:
    """A persistent random token, so an installed phone script keeps working after restarts."""
    import secrets
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{12,64}", token):
            return token
    except OSError:
        pass
    token = secrets.token_urlsafe(12)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    return token


def lan_addresses() -> list:
    """Private IPv4 addresses phones can reach this PC at, the default-route one first.

    Link-local (169.254.x), loopback and Windows' loopback pseudo-interface are skipped.
    """
    import socket

    def usable(addr: str) -> bool:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        return ip.version == 4 and ip.is_private and not ip.is_loopback and not ip.is_link_local

    found = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # UDP connect sends nothing; picks the default route
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            if "loopback" in name.lower():
                continue
            found += [a.address for a in addrs if a.family == socket.AF_INET]
    except Exception:
        pass
    out = []
    for addr in found:
        if usable(addr) and addr not in out:
            out.append(addr)
    return out


_collecting: set = set()
_collect_lock = threading.Lock()


def _collect_name(page: str) -> str:
    seg = [s for s in urlparse(page).path.split("/") if s]
    return re.sub(r"[^A-Za-z0-9_-]", "", seg[-1]).upper() if seg else ""


def _audio_rendition(master_url: str, playlist: str) -> str:
    """URL of the audio-only playlist in a master playlist (EXT-X-MEDIA TYPE=AUDIO), or ""."""
    best, first = "", ""
    for line in playlist.splitlines():
        line = line.strip()
        if not line.startswith("#EXT-X-MEDIA:") or "TYPE=AUDIO" not in line:
            continue
        uri = re.search(r'URI="([^"]+)"', line)
        if not uri:
            continue
        url = urljoin(master_url, uri.group(1))
        first = first or url
        if "DEFAULT=YES" in line and not best:
            best = url
    return best or first


def _collect_inputs(manager, media: str, referer: str) -> list:
    """Streams to read for a collection, best first. Audio-only playlist when the site has one:
    the master playlist itself makes ffmpeg fetch the biggest video, which is what was slow."""
    if not manager._is_hls(media):
        return [media]
    text = manager._playlist_text(media, referer)
    audio = _audio_rendition(media, text)
    if audio:
        return [audio]
    variants = [url for url, _ in manager._master_variants(media, text)]
    return variants[:1] or [media]          # cheapest video variant (sound is in it) as last resort


def _collect_worker(manager, name: str, media: str, referer: str) -> None:
    out = COLLECT_DIR / f"{name}.m4a"
    tmp = COLLECT_DIR / f"{name}.part.m4a"
    try:
        COLLECT_DIR.mkdir(parents=True, exist_ok=True)
        for url in _collect_inputs(manager, media, referer):
            cmd = [manager._ffmpeg_executable(), "-nostdin", "-loglevel", "error", "-y",
                   *manager._media_input(url, referer), "-vn", "-t", str(COLLECT_SEC),
                   "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "32k", "-f", "ipod", str(tmp)]
            t0 = time.time()
            info(f"collect: {name} reading {url[-60:]}")
            r = subprocess.run(cmd, capture_output=True, timeout=COLLECT_SEC * 3 + 120)
            if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > COLLECT_SEC * 1500:
                tmp.replace(out)
                info(f"collect: {name} saved {out.stat().st_size // 1024} KB in {time.time() - t0:.0f}s")
                return
            warning(f"collect: {name} failed via {url[:60]}: {r.stderr.decode('utf-8', 'replace')[-150:]}")
    except Exception as e:
        warning(f"collect: {name} error {e}")
    finally:
        tmp.unlink(missing_ok=True)
        with _collect_lock:
            _collecting.discard(name)


def collect_poll(manager, page: str, media: str) -> dict:
    name = _collect_name(page)
    base = {"status": "processing", "cues": [], "covered": [], "duration": 0, "progress": {}}
    if not name:
        return {**base, "message": "수집: 품번을 알 수 없는 주소"}
    if (COLLECT_DIR / f"{name}.m4a").exists():
        return {**base, "message": f"수집 완료: {name}"}
    if not media:
        return {**base, "message": f"수집 대기: {name} (재생을 시작하세요)"}
    with _collect_lock:
        if name not in _collecting:
            _collecting.add(name)
            threading.Thread(target=_collect_worker, args=(manager, name, media, page_referer(page)),
                             daemon=True).start()
    return {**base, "message": f"수집 중: {name}"}


class Handler(BaseHTTPRequestHandler):
    manager: JobManager = None  # set in main()
    token: str = ""

    def log_message(self, fmt, *args):  # keep the console quiet
        # Request queries can contain a media URL.
        return

    def _authorized(self, query: dict) -> bool:
        """Loopback (this PC, or an Android phone through adb reverse) needs no token."""
        import hmac
        if self.client_address[0] in ("127.0.0.1", "::1"):
            return True
        given = (query.get("k") or [""])[0] or self.headers.get("X-ARIA-Token", "")
        return bool(self.token) and hmac.compare_digest(given.encode(), self.token.encode())

    def _userscript(self) -> None:
        try:
            body = USERSCRIPT_PATH.read_text(encoding="utf-8")
        except OSError:
            self._json(404, {"error": "userscript not found"})
            return
        host = self.headers.get("Host", f"127.0.0.1:{PORT}")
        if not re.fullmatch(r"[A-Za-z0-9.\-]+(?::\d{1,5})?|\[[0-9a-fA-F:]+\](?::\d{1,5})?", host):
            host = f"127.0.0.1:{PORT}"
        loopback = self.client_address[0] in ("127.0.0.1", "::1")
        body = (body.replace("__ARIA_SERVER__", f"http://{host}")
                    .replace("__ARIA_TOKEN__", "" if loopback else self.token)
                    .replace("__ARIA_CONNECT_HOST__", host.rsplit(":", 1)[0].strip("[]")))
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ("https://www.youtube.com", "https://m.youtube.com"):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if not self._authorized(parse_qs(url.query)):
            self._json(403, {"error": "forbidden", "message": "접속 토큰이 맞지 않습니다. 스크립트를 다시 설치하세요."})
            return
        if url.path == "/health":
            engines = self.manager.engines
            self._json(200, {"ok": True, "ready": self.manager.ready.is_set(),
                             "error": self.manager.load_error,
                             "cloud": engines.cloud, "groq": engines.groq.status(),
                             "glossary": groq_client.GLOSSARY_MODE,
                             "last_asr": engines.last_asr, "last_mt": engines.last_mt})
            return
        if url.path == "/jobs":
            # Diagnostics: what the worker sees (job keys only, no page URLs).
            now = time.time()
            with self.manager.lock:
                jobs = list(self.manager.jobs.values())
            self._json(200, {"jobs": [{
                "key": j.video_id, "status": j.status, "playing": j.playing,
                "script_2_6": j.playing_reported,
                "position": round(j.want_pos, 1), "ready_ahead": round(j.ahead(), 1),
                "seconds_since_poll": round(now - j.last_poll, 1), "cues": len(j.cues),
            } for j in sorted(jobs, key=lambda j: -j.last_poll)]})
            return
        if url.path == "/subs" and not ALLOW_DESKTOP and \
                not MOBILE_UA_RE.search(self.headers.get("User-Agent", "")) and \
                parse_qs(url.query).get("touch") != ["1"]:
            # A PC browser: no job is created, so nothing is downloaded or sent to Groq.
            self._json(200, {"status": "disabled", "message": "PC 브라우저 자막은 꺼져 있습니다",
                             "cues": [], "covered": [], "duration": 0, "progress": {}})
            return
        if url.path == "/subs":
            q = parse_qs(url.query)
            page = (q.get("url") or [""])[0]
            vid = (q.get("v") or [""])[0]
            if page:
                resolved = resolve_source(page)
                if not resolved:
                    self._json(400, {"error": "unsupported url"})
                    return
                key, source = resolved
            elif VIDEO_ID_RE.match(vid):  # older userscript (YouTube only)
                key, source = vid, f"https://www.youtube.com/watch?v={vid}"
            else:
                self._json(400, {"error": "missing url"})
                return
            if (q.get("off") or ["0"])[0] == "1":
                # The viewer turned subtitles off: stop working on this video now.
                self._json(200, {"status": "off", "stopped": self.manager.stop(key)})
                return
            try:
                t = float((q.get("t") or ["0"])[0])
            except ValueError:
                t = 0.0
            if not math.isfinite(t):
                t = 0.0
            # YouTube audio always comes from yt-dlp; other sites may send the stream their
            # player is playing, used when it points to a public host.
            raw_media = (q.get("media") or [""])[0]
            media = "" if VIDEO_ID_RE.match(key) else public_media_url(raw_media)
            # Older userscripts don't send "playing"; treat them as playing.
            playing = (q.get("playing") or ["1"])[0] != "0"
            retry = (q.get("retry") or ["0"])[0] == "1"
            if COLLECT:
                self._json(200, collect_poll(self.manager, page, media))
                return
            data = self.manager.poll(key, t, source, media, playing, "playing" in q, retry,
                                     referer=page_referer(source))
            if (q.get("win") or ["0"])[0] == "1":
                # Mobile clients poll every second: send only cues near the viewer.
                lo, hi = t - CUE_WINDOW_BEHIND, t + CUE_WINDOW_AHEAD
                data["cues"] = [c for c in data["cues"] if c["e"] >= lo and c["s"] <= hi]
                data["window"] = [round(lo, 2), round(hi, 2)]
            self._json(200, data)
            return
        if url.path == "/collect":
            self._json(200, {"enabled": COLLECT, "dir": str(COLLECT_DIR), "files": sorted(
                f"{p.name} {p.stat().st_size // 1024}KB" for p in COLLECT_DIR.glob("*.m4a")) if COLLECT_DIR.exists() else [],
                "working": sorted(_collecting)})
            return
        if url.path in ("/aria.user.js", "/aria-youtube.user.js"):
            # Opening this in the browser lets Tampermonkey / Userscripts install the script,
            # preconfigured with this server's address (and the token for Wi-Fi devices).
            self._userscript()
            return
        self._json(404, {"error": "not found"})


def _adb_reverse_loop() -> None:
    """Let USB-connected Android devices reach this server at their own 127.0.0.1:PORT.

    `adb reverse` forwards the device's localhost port to this PC over the USB debugging
    link, so no firewall change or network exposure is needed. Re-checked every 10 s so a
    tablet plugged in later is picked up.
    """
    adb = shutil.which("adb")
    if not adb:
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    announced: set = set()
    while True:
        try:
            out = subprocess.run([adb, "devices"], capture_output=True, text=True,
                                 timeout=10, creationflags=flags).stdout
            serials = [l.split()[0] for l in out.splitlines()[1:] if l.strip().endswith("device")]
            for s in serials:
                rev = subprocess.run([adb, "-s", s, "reverse", "--list"], capture_output=True,
                                     text=True, timeout=10, creationflags=flags).stdout
                if f"tcp:{PORT}" not in rev:
                    subprocess.run([adb, "-s", s, "reverse", f"tcp:{PORT}", f"tcp:{PORT}"],
                                   capture_output=True, timeout=10, creationflags=flags)
                if s not in announced:
                    announced.add(s)
                    print(f" 안드로이드 기기 연결됨 ({s}): 기기에서도 자막 서버를 쓸 수 있습니다.")
                    info(f"YouTubeSubs: adb reverse tcp:{PORT} set for {s}")
            announced &= set(serials)
        except Exception as e:
            debug(f"YouTubeSubs: adb reverse check failed: {e}")
        time.sleep(10)


# --------------------------------------------------------------------------- cleanup
# Nothing but the program, models and settings is meant to outlive a session:
# logs, subtitle records and temporary audio are deleted when the server stops
# (window closed, Ctrl+C, Windows shutdown/sign-out) and again at the next start
# in case the PC was switched off hard. Set ARIA_KEEP_FILES=1 to keep them.

LOG_DIR = PROJECT_ROOT / "logs"


# Finished subtitles (cache/youtube/<id>.json) are kept across restarts so a video watched
# again needs no recognition or translation; ones unused for SUBS_KEEP_DAYS are removed.
# ARIA_CLEAR_SUBS=1 restores the old behaviour (subtitles deleted with the session).
SUBS_KEEP_DAYS = 3
CLEAR_SUBS = os.environ.get("ARIA_CLEAR_SUBS") == "1"


def _session_files() -> list:
    files = []
    cache_patterns = ("*.json", "*.tmp") if CLEAR_SUBS else ("*.tmp",)
    for pattern_dir, patterns in ((LOG_DIR, ("*.log",)),
                                  (CACHE_DIR, cache_patterns),
                                  (TMP_DIR, ("*",))):
        if pattern_dir.is_dir():
            for pat in patterns:
                files += [p for p in pattern_dir.glob(pat) if p.is_file()]
    if not CLEAR_SUBS and CACHE_DIR.is_dir():
        too_old = time.time() - SUBS_KEEP_DAYS * 86400
        files += [p for p in CACHE_DIR.glob("*.json") if p.is_file() and p.stat().st_mtime < too_old]
    return files


def _current_log_files() -> set:
    from ..logger import get_logger
    return {Path(h.file_path).resolve() for h in get_logger().handlers if hasattr(h, "file_path")}


def cleanup_session_files(keep_current_log: bool) -> tuple:
    """Delete logs / subtitle records / temp audio. Returns (files, bytes) removed."""
    if os.environ.get("ARIA_KEEP_FILES") == "1":
        return 0, 0
    keep = _current_log_files() if keep_current_log else set()
    if not keep_current_log:
        # Stop writing to log files first so nothing is recreated after deletion.
        from ..logger import get_logger
        lg = get_logger()
        for h in list(lg.handlers):
            if hasattr(h, "file_path"):
                lg.removeHandler(h)
    count = size = 0
    for p in _session_files():
        try:
            if p.resolve() in keep:
                continue
            s = p.stat().st_size
            p.unlink()
            count += 1
            size += s
        except OSError:
            pass  # in use by another process; removed next time
    return count, size


_exit_cleaned = threading.Event()


def _cleanup_at_exit(*_args) -> None:
    if not _exit_cleaned.is_set():
        _exit_cleaned.set()
        cleanup_session_files(keep_current_log=False)


def _install_exit_cleanup() -> None:
    import atexit
    atexit.register(_cleanup_at_exit)
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes
    # CTRL_C, CTRL_BREAK, CTRL_CLOSE (window closed), CTRL_LOGOFF, CTRL_SHUTDOWN
    handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def handler(_event):
        _cleanup_at_exit()
        return False  # let Windows continue closing the process

    _install_exit_cleanup.handler = handler_type(handler)  # keep a reference alive
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_install_exit_cleanup.handler, True)


def main() -> None:
    resume_session = os.environ.pop("ARIA_RESUME_SESSION", "") == "1"
    removed, freed = (0, 0) if resume_session else cleanup_session_files(keep_current_log=True)
    if removed:
        info(f"YouTubeSubs: removed {removed} leftover files ({freed / 1e6:.1f} MB) from the last session")
    _install_exit_cleanup()
    engines = Engines()
    manager = JobManager(engines)
    Handler.manager = manager
    Handler.token = load_access_token()
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"포트 {PORT}를 열 수 없습니다. 서버가 이미 실행 중인지 확인하세요. ({e})")
        sys.exit(1)
    print("=" * 60)
    print(" ARIA 영상 자막 서버 (유튜브 + yt-dlp 지원 사이트)")
    print(f" http://127.0.0.1:{PORT}  (이 창을 닫으면 서버가 꺼집니다)")
    print(" 일본어 영상을 재생하면 자동으로 자막을 만듭니다.")
    if engines.cloud:
        print(f" 음성 인식·번역: Groq 클라우드 (API 키 {len(engines.groq.keys)}개, 한도 시 다음 키)."
              " 모두 안 되면 PC 모델로 자동 전환")
    else:
        print(" 음성 인식·번역: PC 모델 (Groq 키 파일 없음 또는 ARIA_CLOUD=0)")
    if os.environ.get("ARIA_KEEP_FILES") != "1":
        if CLEAR_SUBS:
            print(" 창을 닫거나 PC를 끄면 로그·자막 기록·임시 음성 파일은 자동으로 지워집니다.")
        else:
            print(f" 만든 자막은 남겨 두어 다시 볼 때 바로 나옵니다 ({SUBS_KEEP_DAYS}일 안 본 자막은 삭제)."
                  " 로그·임시 음성 파일은 창을 닫으면 지워집니다.")
    print(f" PC / USB 연결 안드로이드 스크립트 설치: http://127.0.0.1:{PORT}/aria.user.js")
    if LAN_MODE:
        addrs = lan_addresses()
        print("-" * 60)
        print(" [Wi-Fi 모드] 같은 Wi-Fi의 휴대폰 브라우저에서 아래 주소를 여세요:")
        for a in addrs or ["<PC의 IP 주소>"]:
            print(f"   http://{a}:{PORT}/aria.user.js?k={Handler.token}")
        print(" (토큰이 들어 있는 주소이니 다른 사람과 공유하지 마세요)")
        print(f" 토큰 초기화: {TOKEN_FILE} 파일을 지우고 서버를 다시 켜세요")
    if os.environ.get("ARIA_NO_ADB") != "1":
        threading.Thread(target=_adb_reverse_loop, daemon=True, name="YouTubeSubs_ADB").start()
    info(f"YouTubeSubs: audio decode mode = {DECODE_MODE}")
    print("=" * 60)
    info(f"YouTubeSubs: listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager._close_streams()


if __name__ == "__main__":
    main()
