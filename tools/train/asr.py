"""Step 2: Japanese recognition (Groq whisper) of the collected audio, with word times.

    python tools/train/asr.py            # all files in cache/train/audio -> cache/train/asr/<code>.json

Resumable: finished files are skipped; a file is written only when complete.
Waits and retries when every Groq key is at its limit.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from realtime_subtitles.youtube_subs import server as S          # noqa: E402
from realtime_subtitles.youtube_subs import file_subs as F       # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import GroqUnavailable  # noqa: E402

T = ROOT / "cache" / "train"
WIN = 30.0
SR = S.SAMPLE_RATE


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(T / "asr.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def transcribe(groq, chunk):
    tries = 0
    while True:
        try:
            segs = groq.transcribe(chunk, words=True)
            return segs, groq.last_words
        except GroqUnavailable as e:
            log(f"  all keys limited ({e}); waiting 60s")
            time.sleep(60)
        except Exception as e:            # network hiccup etc.
            tries += 1
            if tries > 5:
                raise
            log(f"  error {e!r}; retry in 15s")
            time.sleep(15)


def main() -> None:
    (T / "asr").mkdir(parents=True, exist_ok=True)
    groq = S.Engines().groq
    files = sorted((T / "audio").glob("*.m4a"))
    for n, path in enumerate(files, 1):
        out = T / "asr" / f"{path.stem}.json"
        if out.exists():
            continue
        raw = T / "asr" / "tmp.raw"
        dur = F.extract_audio(path, raw)
        audio = np.fromfile(raw, dtype=np.int16)
        segs_out, words_out = [], []
        t0 = time.time()
        start = 0.0
        while start < dur - 0.3:
            end = min(start + WIN, dur)
            chunk = audio[int(start * SR):int(end * SR)].astype(np.float32) / 32768.0
            if S.speech_regions(chunk, min_silence_ms=500):
                segs, words = transcribe(groq, chunk)
                segs_out += [{"s": round(start + s.start, 2), "e": round(start + s.end, 2), "ja": s.text}
                             for s in segs]
                words_out += [{"s": round(start + w[0], 2), "e": round(start + w[1], 2), "w": w[2]}
                              for w in words]
            start = end
        out.write_text(json.dumps({"code": path.stem, "duration": dur, "segments": segs_out,
                                   "words": words_out}, ensure_ascii=False), encoding="utf-8")
        raw.unlink(missing_ok=True)
        log(f"[{n}/{len(files)}] {path.stem}: {len(segs_out)} segments in {time.time() - t0:.0f}s")
    log("ASR DONE")


if __name__ == "__main__":
    main()
