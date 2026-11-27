"""WhisperJAV's main idea, tested: send Groq only the speech (Silero VAD regions, packed into
<=28 s chunks with short silences between), map the word times back, and compare with sending
plain 30 s windows (the saved results).

    python tools/train/vadclamp_test.py CODE [CODE ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A                                                              # noqa: E402
import whisperjav_check as W                                                   # noqa: E402
from realtime_subtitles.youtube_subs import server as S                      # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import GroqPool, GroqUnavailable, fix_word_times  # noqa: E402

T = ROOT / "cache" / "train"
SR = 16000
PAD, JOIN, SEP, MAXLEN = 0.2, 0.3, 0.3, 28.0


def chunks(regions, dur):
    rs = []
    for a, b in regions:
        a, b = max(0.0, a - PAD), min(dur, b + PAD)
        if rs and a - rs[-1][1] < JOIN:
            rs[-1][1] = b
        else:
            rs.append([a, b])
    out, cur, length = [], [], 0.0
    for a, b in rs:
        while b - a > MAXLEN:                    # very long region: cut it
            out.append([(a, a + MAXLEN)])
            a += MAXLEN
        if cur and length + SEP + (b - a) > MAXLEN:
            out.append(cur)
            cur, length = [], 0.0
        cur.append((a, b))
        length += (SEP if length else 0.0) + (b - a)
    if cur:
        out.append(cur)
    return out


def run(code, pool):
    audio = W.load_audio(code)
    dur = len(audio) / SR
    regions = W.silero(code)
    words = []
    for pieces in chunks(regions, dur):
        parts, spans, t = [], [], 0.0
        for a, b in pieces:
            if parts:
                parts.append(np.zeros(int(SEP * SR), np.float32))
                t += SEP
            seg = audio[int(a * SR):int(b * SR)]
            parts.append(seg)
            spans.append((t, t + len(seg) / SR, a))
            t += len(seg) / SR
        chunk = np.concatenate(parts)
        while True:
            try:
                pool.transcribe(chunk, words=True)
                break
            except GroqUnavailable as e:
                print("  waiting", e, flush=True)
                time.sleep(60)

        def orig(x):
            best = min(spans, key=lambda s: 0 if s[0] <= x <= s[1] else min(abs(x - s[0]), abs(x - s[1])))
            x = min(max(x, best[0]), best[1])
            return best[2] + (x - best[0])
        words += [(orig(s), orig(e), w) for s, e, w in pool.last_words]
    return words


def main():
    pool = GroqPool()
    tot = {"30s windows (now)": np.zeros(5), "speech only (VAD clamp)": np.zeros(5)}
    for code in sys.argv[1:]:
        sub = next((T / "subs").glob(f"{code}.*"))
        d = json.loads((T / "asr" / f"{code}.json").read_text(encoding="utf-8"))
        ref = A.load_ref(sub)
        segs, dur = A.load_asr(T / "asr" / f"{code}.json")
        sc, off, _ = A.best_timing(ref, segs, dur)
        refs = [s * sc + off for s, e, ko in ref
                if 0 <= s * sc + off <= min(dur, A.LIMIT) and not S.is_filler_ko(ko)]
        base_w = fix_word_times([(w["s"], w["e"], w["w"]) for w in d["words"]])
        cache = T / "vad" / f"{code}.clamp.json"
        if cache.exists():
            clamp_w = [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
        else:
            clamp_w = run(code, pool)
            cache.write_text(json.dumps(clamp_w, ensure_ascii=False), encoding="utf-8")
        for name, ws in (("30s windows (now)", base_w), ("speech only (VAD clamp)", clamp_w)):
            lines = [l for l in S.words_to_lines(sorted(ws), 0.0) if not S.is_moan(l[2])]
            h05, h1, p1, nl, nr = W.score(lines, refs)
            tot[name] += np.array([h05 * nr, h1 * nr, p1 * nl, nl, nr])
            print(f"{code:10s} {name:24s} hit<=0.5s {h05:.0%} hit<=1s {h1:.0%} precision {p1:.0%} lines {nl}", flush=True)
    print()
    for name, v in tot.items():
        print(f"ALL {name:24s} hit<=0.5s {v[0] / v[4]:.1%} hit<=1s {v[1] / v[4]:.1%} precision {v[2] / v[3]:.1%} lines {int(v[3])}")


if __name__ == "__main__":
    main()
