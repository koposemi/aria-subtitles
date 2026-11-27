"""
Groq cloud API for the subtitle server: Japanese speech recognition (Whisper) and
Japanese -> Korean subtitle translation (LLM).

Keys are read from ~/.config/realtime-subtitles/groq_key.txt, one per line (blank lines and
lines starting with # are ignored). When a key hits a rate limit (HTTP 429) it rests until
the time Groq names and the next key is used. Every caller falls back to the local models
when no key can answer, so subtitles never stop because of the cloud.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from ..logger import info, warning, debug

KEY_FILE = Path.home() / ".config" / "realtime-subtitles" / "groq_key.txt"
API = "https://api.groq.com/openai/v1"
ASR_MODEL = "whisper-large-v3"
# Translation models in order of use. Each model is tried on every key (key 1, 2, 3, starting
# from the key that last worked) before the next model is used; each has its own quota.
MT_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
SHORT_WAIT_SEC = 8.0      # wait this long at most for a per-minute limit to reset
# Adult-video glossary in the translation instructions (av_glossary.py):
#   "off", "20"/"30" (first 20/30 entries), "50" (all entries), or "match" (only entries whose
#   Japanese form occurs in the lines being translated). Set with ARIA_GLOSSARY.
#   Default "match": same accuracy as the full glossary in tests (98-99%) with ~40% fewer
#   tokens than 30 fixed entries.
GLOSSARY_MODE = os.environ.get("ARIA_GLOSSARY", "match")
SAMPLE_RATE = 16000


class GroqUnavailable(RuntimeError):
    """No key could answer now (missing, rate-limited or network down): use local models."""


def _reset_seconds(value: str) -> float:
    """Parse Groq reset values such as '2.29s', '1m26.4s', '7h3m2s', '350ms'."""
    if not value:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        n = float(num)
        total += {"ms": n / 1000, "s": n, "m": n * 60, "h": n * 3600}[unit]
    return total


class GroqPool:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.rest_until: dict[tuple[int, str], float] = {}   # (key index, model) -> time
        self.current: dict[str, int] = {}                     # model -> key index last used
        self.last_model = ""
        self.last_usage: dict = {}                             # token usage of the last translation
        self.lock = threading.Lock()
        self.reload()

    # ---- keys
    def reload(self) -> None:
        try:
            lines = KEY_FILE.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            lines = []
        keys = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and line not in keys:
                keys.append(line)
        with self.lock:
            self.keys = keys
            self.current = {}
        if keys:
            info(f"Groq: {len(keys)} API key(s) loaded")

    @property
    def available(self) -> bool:
        return bool(self.keys)

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            resting = {f"key{i + 1}/{m}": round(t - now) for (i, m), t in self.rest_until.items() if t > now}
            return {"keys": len(self.keys),
                    "current_key": {m: i + 1 for m, i in self.current.items()},
                    "last_translation_model": self.last_model, "resting": resting}

    def _order(self, model: str) -> list[int]:
        with self.lock:
            n = len(self.keys)
            first = self.current.get(model, 0)
            return [(first + i) % n for i in range(n)]

    # ---- transport
    def _request(self, key: str, path: str, body: bytes, content_type: str, timeout: float):
        req = urllib.request.Request(API + path, data=body, method="POST", headers={
            "Authorization": "Bearer " + key, "Content-Type": content_type,
            "User-Agent": "aria-subtitles/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def call(self, model: str, path: str, body: bytes, content_type: str, timeout: float = 30.0,
             wait: bool = True) -> dict:
        """POST to Groq with key rotation. Raises GroqUnavailable when no key can answer.

        wait: when every key is resting on a per-minute limit that resets within a few
        seconds, wait for it (False: give up at once so the caller can use another model).
        """
        if not self.keys:
            raise GroqUnavailable("Groq API 키가 없습니다")
        for attempt in range(2):
            soonest = None
            for idx in self._order(model):
                now = time.time()
                rest = self.rest_until.get((idx, model), 0.0)
                if rest > now:
                    soonest = rest if soonest is None else min(soonest, rest)
                    continue
                try:
                    data = self._request(self.keys[idx], path, body, content_type, timeout)
                    with self.lock:
                        if self.current.get(model, 0) != idx:
                            info(f"Groq: {model} switched to key {idx + 1}")
                        self.current[model] = idx
                    return data
                except urllib.error.HTTPError as e:
                    detail = e.read().decode("utf-8", "replace")[:300]
                    if e.code == 429:
                        wait = (float(e.headers.get("retry-after") or 0)
                                or _reset_seconds(e.headers.get("x-ratelimit-reset-tokens", ""))
                                or _reset_seconds(e.headers.get("x-ratelimit-reset-requests", ""))
                                or 60.0)
                        with self.lock:
                            self.rest_until[(idx, model)] = time.time() + wait
                        warning(f"Groq: key {idx + 1} rate-limited on {model} for {wait:.0f}s")
                        soonest = time.time() + wait if soonest is None else min(soonest, time.time() + wait)
                        continue
                    if e.code in (401, 403):
                        warning(f"Groq: key {idx + 1} refused ({e.code}); skipping it for 1 hour")
                        with self.lock:
                            self.rest_until[(idx, model)] = time.time() + 3600
                        continue
                    if e.code == 400:
                        raise GroqUnavailable(f"Groq 요청 오류 400: {detail[:150]}") from e
                    warning(f"Groq: HTTP {e.code} on {model}: {detail[:120]}")
                    raise GroqUnavailable(f"Groq HTTP {e.code}") from e
                except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                    raise GroqUnavailable(f"Groq 연결 실패: {e}") from e
            # Every key is resting: a per-minute limit resets in seconds, so wait once.
            if wait and attempt == 0 and soonest is not None and soonest - time.time() <= SHORT_WAIT_SEC:
                time.sleep(max(0.0, soonest - time.time()) + 0.2)
                continue
            break
        raise GroqUnavailable("모든 Groq 키가 한도에 도달했습니다")

    # ---- speech recognition
    def transcribe(self, audio: np.ndarray, words: bool = False) -> list:
        """Japanese speech -> segments with .start/.end/.text (seconds from window start)."""
        pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        boundary = uuid.uuid4().hex
        fields = [("model", ASR_MODEL), ("language", "ja"), ("response_format", "verbose_json"),
                  ("temperature", "0"), ("timestamp_granularities[]", "segment")]
        if words:       # word-level times for precisely timed, short subtitle lines (file_subs.py)
            fields.append(("timestamp_granularities[]", "word"))
        parts = []
        for name, value in fields:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="window.wav"\r\n'
                     f'Content-Type: audio/wav\r\n\r\n'.encode() + wav.getvalue() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = self.call(ASR_MODEL, "/audio/transcriptions", b"".join(parts),
                         f"multipart/form-data; boundary={boundary}", timeout=60)
        segments = []
        for s in data.get("segments") or []:
            text = (s.get("text") or "").strip()
            # Whisper invents text over music/silence; Groq reports how likely that is.
            if not text or (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -0.8):
                continue
            if HALLUCINATION_RE.search(text):
                continue
            segments.append(Segment(float(s["start"]), float(s["end"]), text))
        self.last_words = [(float(w["start"]), float(w["end"]), w.get("word") or "")
                           for w in (data.get("words") or [])]
        return segments

    # ---- translation
    def translate(self, lines: list[str], context: list[tuple[str, str]]) -> list[Optional[str]]:
        """Translate subtitle lines together (with earlier lines as context).

        Returns one Korean line per input; None where a model gave nothing usable.
        Raises GroqUnavailable when no model/key can answer at all.
        """
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
        ctx = "\n".join(f"- {ja} => {ko}" for ja, ko in context if ja and ko)
        user = (("이전 자막 (참고용, 번역하지 말 것):\n" + ctx + "\n\n") if ctx else "") + \
               "번역할 자막:\n" + numbered
        last_error = None
        system = MT_SYSTEM + glossary_text(lines, GLOSSARY_MODE)
        for n, model in enumerate(MT_MODELS):
            body = {"model": model, "temperature": 0.2, "max_tokens": 1500,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}]}
            body.update({"reasoning_effort": "none"} if model.startswith("qwen/")
                        else {"reasoning_effort": "low"})
            try:
                # Only the last model waits for a per-minute limit; earlier ones hand over
                # to the next model at once (all keys of this model are busy).
                data = self.call(model, "/chat/completions", json.dumps(body).encode(), "application/json",
                                 wait=(n == len(MT_MODELS) - 1))
            except GroqUnavailable as e:
                last_error = e
                continue
            try:
                content = data["choices"][0]["message"]["content"] or ""
                obj = json.loads(content[content.find("{"):content.rfind("}") + 1])
                out = []
                for i in range(len(lines)):
                    v = obj.get(str(i + 1))
                    out.append(v.strip() if isinstance(v, str) and v.strip() else None)
            except (KeyError, IndexError, ValueError, AttributeError) as e:
                debug(f"Groq: unusable translation from {model}: {e}")
                last_error = GroqUnavailable(f"{model} 응답 형식 오류")
                continue
            if any(out):
                self.last_model = model
                self.last_usage = data.get("usage") or {}
                return out
            last_error = GroqUnavailable(f"{model} 빈 번역")
        raise last_error or GroqUnavailable("번역 실패")


def glossary_text(lines: list[str], mode: str) -> str:
    """Glossary section for the translation instructions ("" when off or nothing matches)."""
    if mode not in ("20", "30", "50", "match"):
        return ""
    from .av_glossary import GLOSSARY
    entries = GLOSSARY[:int(mode)] if mode in ("20", "30") else GLOSSARY
    if mode == "match":
        joined = "".join(lines)
        entries = [e for e in GLOSSARY if any(form in joined for form in e[0])]
    if not entries:
        return ""
    return ("\n\n용어집 (성인 영상 자막: 이 일본어 표현은 아래 한국어로 번역):\n"
            + "\n".join(f"- {' / '.join(forms)} → {ko}" for forms, ko in entries))


class Segment:
    """Same fields the server reads from faster-whisper segments."""
    __slots__ = ("start", "end", "text")

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text


# Phrases Whisper is known to invent over silence/music in Japanese videos.
HALLUCINATION_RE = re.compile(
    r"(ご視聴ありがとうございました|チャンネル登録(を)?お願いします|字幕(は|作成)|"
    r"ご覧いただきありがとうございます|最後までご視聴|おやすみなさい。?$)")

MT_SYSTEM = (
    "당신은 일본어 영상의 한국어 자막 번역가입니다. 입력은 음성 인식 결과라서 구두점이 없거나 "
    "가끔 같은 말이 반복되는 오류가 있습니다. 규칙:\n"
    "1. 번호가 붙은 각 줄을 같은 번호의 자연스러운 한국어 자막 한 줄로 번역합니다. 줄을 합치거나 나누지 않습니다.\n"
    "2. 원문에 없는 말을 덧붙이지 않고, 문장 끝(~と思います, ~です 등)까지 빠짐없이 번역합니다.\n"
    "3. 말투는 화자의 말투를 따르고, 자막답게 간결하게 씁니다. 반복 인식 오류는 한 번만 번역합니다.\n"
    "4. 이전 자막은 문맥 참고용입니다.\n"
    '5. 출력은 JSON 객체 하나만: {"1": "번역", "2": "번역", ...}'
)
