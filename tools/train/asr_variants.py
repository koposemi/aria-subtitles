"""Three suggested Groq recognition tweaks, measured on the first 500 s of 4 hard videos
(4 x 500 s x 3 variants = 100 min of audio). Baseline = the saved results for the same span.

    python tools/train/asr_variants.py

  A: dynaudnorm  - ffmpeg dynamic loudness normalisation before sending (no cutting, times kept)
  B: prompt      - Whisper prompt with words that come up a lot (it is read as preceding text,
                   not as an instruction, so no "ignore breathing" orders)
  C: turbo       - whisper-large-v3-turbo instead of whisper-large-v3
Metrics vs the human subtitles (same as precision_check.py): human-line hit within 0.5 s / 1 s,
and how our lines land (matched / inside a human line / 1-3 s off / no human line).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A                                                              # noqa: E402
import precision_check as P                                                    # noqa: E402
import whisperjav_check as W                                                   # noqa: E402
from realtime_subtitles.youtube_subs import server as S                      # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import GroqPool, GroqUnavailable, fix_word_times  # noqa: E402

T = ROOT / "cache" / "train"
CODES = ["KAWD-343", "SHKD-736", "JUX-241", "SDSI-002"]
SPAN = 500.0
SR = 16000
PROMPT = "はい、大丈夫です。気持ちいい？もっと。イク、イッちゃう。ダメ、やめて。奥さん、お願いします。"


def dynaudnorm(audio):
    r = subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(SR),
                        "-ac", "1", "-i", "-", "-af", "dynaudnorm", "-f", "f32le", "-"],
                       input=audio.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)[: len(audio)]


def recognise(pool, audio, prompt="", model=""):
    words = []
    for w in range(int(len(audio) / SR // 30) + 1):
        chunk = audio[int(w * 30 * SR):int((w + 1) * 30 * SR)]
        if len(chunk) < SR // 2 or not S.speech_regions(chunk, min_silence_ms=500):
            continue
        while True:
            try:
                segs = pool.transcribe(chunk, words=True, prompt=prompt, model=model)
                break
            except GroqUnavailable as e:
                print("  waiting", str(e)[:60], flush=True)
                time.sleep(60)
        words += [(w * 30 + s, w * 30 + e, t) for g in segs for s, e, t in g.words]
    return words


def evaluate(lines, refs, spans):
    rs = np.array(sorted(refs))
    ls = np.array(sorted(l[0] for l in lines))
    h05 = h1 = 0
    for r in rs:
        d = float(np.min(np.abs(ls - r))) if len(ls) else 99
        h05 += d <= 0.5
        h1 += d <= 1.0
    cat = {"matched": 0, "inside": 0, "off1-3": 0, "none": 0}
    for s, e, t in lines:
        dn = float(np.min(np.abs(rs - s))) if len(rs) else 99
        inside = any(a - 0.3 <= s <= b + 0.3 for a, b, _ in spans)
        cat["matched" if dn <= 1 else ("inside" if inside else ("off1-3" if dn <= 3 else "none"))] += 1
    return np.array([h05, h1, len(rs), cat["matched"], cat["inside"], cat["off1-3"], cat["none"], len(lines)])


def main():
    pool = GroqPool()
    names = ["baseline (now)", "A dynaudnorm", "B prompt", "C turbo"]
    tot = {n: np.zeros(8) for n in names}
    for code in CODES:
        sub = next((T / "subs").glob(f"{code}.*"))
        d = json.loads((T / "asr" / f"{code}.json").read_text(encoding="utf-8"))
        ref = A.load_ref(sub)
        segs, dur = A.load_asr(T / "asr" / f"{code}.json")
        sc, off, _ = A.best_timing(ref, segs, dur)
        spans = [(s * sc + off, e * sc + off, ko) for s, e, ko in ref if 0 <= s * sc + off <= SPAN]
        refs = [s for s, e, ko in spans if not S.is_filler_ko(ko)]
        audio = W.load_audio(code)[: int(SPAN * SR)]
        cache = T / "vad" / f"{code}.variants.json"
        got = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        base = [(w["s"], w["e"], w["w"]) for w in P.kept_words(d) if w["s"] < SPAN]
        runs = {"baseline (now)": base}
        for name, fn in (("A dynaudnorm", lambda: recognise(pool, dynaudnorm(audio))),
                         ("B prompt", lambda: recognise(pool, audio, prompt=PROMPT)),
                         ("C turbo", lambda: recognise(pool, audio, model="whisper-large-v3-turbo"))):
            if name not in got:
                got[name] = fn()
                cache.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
            runs[name] = [tuple(x) for x in got[name]]
        for name in names:
            lines = [l for l in S.words_to_lines(fix_word_times(sorted(runs[name])), 0.0)
                     if not S.is_moan(l[2]) and l[0] <= SPAN]
            v = evaluate(lines, refs, spans)
            tot[name] += v
            print(f"{code:9s} {name:15s} hit<=0.5s {v[0] / v[2]:.0%} hit<=1s {v[1] / v[2]:.0%} "
                  f"none {v[6] / max(v[7], 1):.0%} lines {int(v[7])}", flush=True)
    print(f"\n{'':15s} {'hit<=0.5s':>9s} {'hit<=1s':>8s} {'matched':>8s} {'inside':>7s} {'off1-3':>7s} {'none':>6s} {'lines':>6s}")
    for name, v in tot.items():
        n = v[7]
        print(f"{name:15s} {v[0] / v[2]:9.1%} {v[1] / v[2]:8.1%} {v[3] / n:8.1%} {v[4] / n:7.1%} "
              f"{v[5] / n:7.1%} {v[6] / n:6.1%} {int(n):6d}")


if __name__ == "__main__":
    main()
