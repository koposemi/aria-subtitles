"""Subtitle a video file on this device: make <name>.srt, and optionally burn the
subtitles into a new mp4 (<name>.자막.mp4).

    aria-file /path/to/video.mp4            # .srt + burned-in mp4
    aria-file video.mp4 --no-burn           # .srt only
    aria-file video.mp4 --burn-only         # reuse an existing .srt, only burn

Uses the same recognition/translation engines and sentence logic as the server
(server.py: Whisper via Groq -> Japanese sentences -> Korean via Groq LLM).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import server as S
from ..logger import info, warning

SR = S.SAMPLE_RATE
WINDOW = S.WINDOW_SEC


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg 가 없습니다 (Termux: pkg install ffmpeg)")
    return exe


def extract_audio(video: Path, raw: Path) -> float:
    """Whole audio track -> 16 kHz mono s16 raw file. Returns duration in seconds."""
    r = subprocess.run([_ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
                        "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", str(raw)],
                       capture_output=True)
    if r.returncode != 0 or not raw.exists() or raw.stat().st_size < SR * 2:
        sys.exit("음성을 꺼내지 못했습니다: " + r.stderr.decode("utf-8", "replace")[-300:])
    return raw.stat().st_size / 2 / SR


PUNCT_END = "。！？!?…"
PUNCT_SOFT = "、,，"


def words_to_lines(words: list, offset: float) -> list:
    """Word timestamps -> short subtitle lines [(start, end, text)] (about 6-16 characters each)."""
    out, cur = [], None
    for ws, we, w in words:
        w = w.strip()
        if not w:
            continue
        a, b = offset + ws, offset + we
        if cur is not None:
            gap = a - cur[1]
            n = len(cur[2])
            if gap > 0.45 or (n >= 10 and gap > 0.12) or n + len(w) > 20:
                out.append(tuple(cur))
                cur = None
        if cur is None:
            cur = [a, b, w]
        else:
            cur[1] = b
            cur[2] += w
        if cur[2][-1] in PUNCT_END or (cur[2][-1] in PUNCT_SOFT and len(cur[2]) >= 8)                 or (len(cur[2]) >= 7 and S.SENTENCE_END_RE.search(cur[2])):
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def asr_window(engines: S.Engines, chunk: np.ndarray):
    """One window -> (segments, words); waits and retries while Groq is rate limited."""
    if not S.speech_regions(chunk, min_silence_ms=500):
        return [], []
    from .groq_client import GroqUnavailable
    while True:
        try:
            segs = engines.groq.transcribe(chunk, words=True)
            return [s for s in segs if s.text.strip() and s.end - s.start >= 0.2], engines.groq.last_words
        except GroqUnavailable as e:
            warning(f"Groq 대기 중 ({e}) - 30초 뒤 다시 시도")
            time.sleep(30)


def make_cues(engines: S.Engines, audio: np.ndarray, duration: float) -> list:
    cues: list = []
    start = 0.0
    t_begin = time.perf_counter()
    prev_ja = None
    while start < duration - 0.3:
        end = min(start + WINDOW, duration)
        is_last = end >= duration - 0.05
        chunk = audio[int(start * SR):int(end * SR)].astype(np.float32) / 32768.0
        segs, words = asr_window(engines, chunk)
        cut = end
        if not is_last and segs and end - start > 5:       # same sentence-boundary cut as the server
            win = end - start
            final_idx = [i for i, s in enumerate(segs)
                         if S.SENTENCE_END_RE.search(s.text.strip()) and s.end > win * 0.4]
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
        if words:
            # keep only words inside the accepted segments (drops hallucinated / cut-off ones)
            words = [w for w in words
                     if any(s.start - 0.15 <= (w[0] + w[1]) / 2 <= s.end + 0.15 for s in segs)
                     and start + w[0] < cut]
            sentences = words_to_lines(words, start)
        else:
            sentences = S.JobManager._merge(segs, start)
        if sentences:
            sentences = S.JobManager._align_cue_starts(sentences, chunk, start)
        kept = []
        for s0, s1, ja in sentences:
            if ja == prev_ja:
                continue
            prev_ja = ja
            if S.is_moan(ja):
                continue
            kept.append((s0, s1, ja))
        if kept:
            context = [(c["ja"], c["ko"]) for c in cues[-3:]]
            while True:
                try:
                    kos, _tags = engines.translate_lines([ja for _, _, ja in kept], context)
                    break
                except S.CloudUnavailable as e:
                    warning(f"Groq 대기 중 ({e}) - 30초 뒤 다시 시도")
                    time.sleep(30)
            for (s0, s1, ja), ko in zip(kept, kos):
                cues.append({"s": round(s0, 2), "e": round(s1, 2), "ja": ja, "ko": ko or ja})
        start = max(cut, start + 1.0)
        done = min(start / duration, 1.0)
        el = time.perf_counter() - t_begin
        eta = el / done * (1 - done) if done > 0.02 else 0
        print(f"\r  자막 만드는 중 {done * 100:5.1f}%  (자막 {len(cues)}줄, 남은 시간 약 {eta / 60:.1f}분)   ",
              end="", flush=True)
    print()
    return cues


def _ts(t: float) -> str:
    t = max(t, 0.0)
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _wrap(text: str, width: int = 22) -> str:
    text = text.strip()
    if len(text) <= width:
        return text
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def write_srt(cues: list, path: Path, duration: float, wrap: int = 22) -> None:
    out = []
    n = 0
    for i, c in enumerate(cues):
        text = _wrap(c["ko"], wrap)
        if not text:
            continue
        s = max(c["s"] - 0.1, 0.0)                     # appear a hair early: reads as "in sync"
        e = max(c["e"] + 0.25, s + 0.9)                # short lines stay readable
        e = min(e, s + 5.0)
        if i + 1 < len(cues):
            e = min(e, max(cues[i + 1]["s"] - 0.12, s + 0.3))
        e = min(e, duration)
        n += 1
        out.append(f"{n}\n{_ts(s)} --> {_ts(e)}\n{text}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def _video_size(video: Path) -> tuple:
    probe = shutil.which("ffprobe")
    if not probe:
        return 0, 0
    r = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation", "-of", "json", str(video)],
                       capture_output=True)
    try:
        st = json.loads(r.stdout)["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        rot = int(abs(float((st.get("side_data_list") or [{}])[0].get("rotation", 0)))) % 180
        return (h, w) if rot == 90 else (w, h)
    except Exception:
        return 0, 0


FONT_DIR = Path(__file__).resolve().parents[3] / "fonts"       # bundled Pretendard (OFL)


def burn(video: Path, srt: Path, out: Path) -> None:
    w, h = _video_size(video)
    # SRT subtitles live on a 384x288 canvas that is stretched to the video, so the font size is
    # given in those units: ~6% of the short side in real pixels.
    size = max(round(0.065 * min(w, h) * 288 / h), 8) if w and h else 16
    style = (f"FontName=Pretendard SemiBold,FontSize={size},Bold=0,Outline={max(size * 0.09, 1):.1f},"
             f"Shadow={max(size * 0.04, 0.5):.1f},MarginV={int(size * 2.2)}")
    work = Path(tempfile.mkdtemp(prefix="aria_burn_"))
    try:
        shutil.copy(srt, work / "s.srt")               # simple name: no quoting problems in the filter
        for f in FONT_DIR.glob("*.otf"):
            shutil.copy(f, work / f.name)
        flt = f"subtitles=s.srt:fontsdir=.:force_style='{style}'"
        cmd = [_ffmpeg(), "-nostdin", "-loglevel", "error", "-stats", "-y", "-i", str(video),
               "-vf", flt, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out)]
        print("  자막을 영상에 입히는 중 (영상 길이에 따라 오래 걸립니다)")
        r = subprocess.run(cmd, cwd=work)
        if r.returncode != 0:
            sys.exit("자막 입히기 실패 (위 ffmpeg 오류 참고). .srt 파일은 만들어져 있습니다.")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def media_scan(path: Path) -> None:
    """Tell Android's gallery about a new file (files written from Termux are not listed otherwise)."""
    try:
        if shutil.which("termux-media-scan"):
            subprocess.run(["termux-media-scan", str(path)], capture_output=True, timeout=20)
        elif shutil.which("am"):
            subprocess.run(["am", "broadcast", "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
                            "-d", "file://" + str(path)], capture_output=True, timeout=20)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="영상 파일에 한국어 자막 만들기")
    ap.add_argument("video")
    ap.add_argument("--no-burn", action="store_true", help=".srt 만 만들고 영상에는 입히지 않음")
    ap.add_argument("--burn-only", action="store_true", help="이미 있는 .srt 로 영상에 입히기만 함")
    a = ap.parse_args()

    video = Path(a.video).expanduser().resolve()
    if not video.is_file():
        sys.exit(f"파일이 없습니다: {video}")
    srt = video.with_suffix(".srt")
    out = video.with_name(video.stem + ".자막.mp4")

    if not a.burn_only:
        engines = S.Engines()
        engines.load()
        tmp = Path(tempfile.mkdtemp(prefix="aria_file_"))
        try:
            print(f"[1/3] 소리 꺼내는 중: {video.name}")
            raw = tmp / "a.raw"
            duration = extract_audio(video, raw)
            print(f"      길이 {duration / 60:.1f}분")
            audio = np.memmap(raw, dtype=np.int16, mode="r")
            print("[2/3] 일본어 인식 + 한국어 번역")
            cues = make_cues(engines, audio, duration)
            del audio
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        w, h = _video_size(video)
        wrap = max(16, min(int(18 * w / min(w, h)), 34)) if w and h else 22
        write_srt(cues, srt, duration, wrap)
        media_scan(srt)
        print(f"      자막 파일 저장: {srt}")
    elif not srt.is_file():
        sys.exit(f"자막 파일이 없습니다: {srt}")

    if not a.no_burn:
        print("[3/3] 영상에 자막 입히기")
        burn(video, srt, out)
        media_scan(out)
        print(f"      저장 완료: {out}  (갤러리에 안 보이면 갤러리 앱을 닫았다 다시 여세요)")
    print("끝.")


if __name__ == "__main__":
    main()
