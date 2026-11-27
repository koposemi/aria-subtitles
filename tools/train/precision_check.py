"""Where do our subtitle lines land relative to the human subtitles? (situation breakdown)

    python tools/train/precision_check.py

Per our line (built like the live server: only words of segments the Groq client kept):
  matched : a human line starts within 1 s
  inside  : falls inside a human line (humans joined several utterances into one line) - not an error
  off1-3  : nearest human line start is 1-3 s away (timing error)
  none    : no human line nearby (sounds, unsubtitled talk, hallucinations)
Also prints the human-line hit rates (<=0.5 s, <=1 s) so a filter can't "win" by deleting dialogue.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A                                                              # noqa: E402
from realtime_subtitles.youtube_subs import server as S                      # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import fix_word_times, _norm  # noqa: E402

T = ROOT / "cache" / "train"
SKIP = {"STAR-774", "MIMK-031"}


def kept_words(d) -> list:
    """Words belonging to the segments the client kept (matched by text, in order)."""
    words, out, wi = d["words"], [], 0
    for seg in (_norm(s["ja"]) for s in d["segments"] if s["ja"].strip()):
        j = wi
        while j < len(words):
            acc, k = "", j
            while k < len(words) and len(acc) < len(seg):
                acc += _norm(words[k]["w"])
                k += 1
            n = max(1, len(seg) - 2)
            if acc[:n] == seg[:n]:
                out += words[j:k]
                wi = k
                break
            j += 1
    return out


def main() -> None:
    cat, none_txt = collections.Counter(), collections.Counter()
    hit05 = hit1 = nref = 0
    for sub in sorted((T / "subs").iterdir()):
        code = sub.stem
        if code in SKIP:
            continue
        d = json.loads((T / "asr" / f"{code}.json").read_text(encoding="utf-8"))
        ref = A.load_ref(sub)
        segs, dur = A.load_asr(T / "asr" / f"{code}.json")
        sc, off, _ = A.best_timing(ref, segs, dur)
        lim = min(dur, A.LIMIT)
        spans = [(s * sc + off, e * sc + off, ko) for s, e, ko in ref if 0 <= s * sc + off <= lim]
        rs = np.array(sorted(x[0] for x in spans if not S.is_filler_ko(x[2])))
        words = fix_word_times([(w["s"], w["e"], w["w"]) for w in kept_words(d)])
        lines = [l for l in S.words_to_lines(words, 0.0) if not S.is_moan(l[2]) and l[0] <= lim]
        ls = np.array(sorted(l[0] for l in lines))
        for r in rs:
            dmin = float(np.min(np.abs(ls - r))) if len(ls) else 99
            hit05 += dmin <= 0.5
            hit1 += dmin <= 1.0
        nref += len(rs)
        for s, e, t in lines:
            dn = float(np.min(np.abs(rs - s))) if len(rs) else 99
            inside = any(a - 0.3 <= s <= b + 0.3 for a, b, _ in spans)
            c = "matched" if dn <= 1 else ("inside" if inside else ("off1-3" if dn <= 3 else "none"))
            cat[c] += 1
            if c == "none":
                none_txt[re.sub(r"[\s、。．,.!！?？…]", "", t)] += 1
    n = sum(cat.values())
    print(f"lines {n}")
    for k in ("matched", "inside", "off1-3", "none"):
        print(f"  {k:8s} {cat[k]:5d} {cat[k] / n:6.1%}")
    print(f"human lines hit: <=0.5s {hit05 / nref:.1%}  <=1s {hit1 / nref:.1%}")
    print("none - most frequent:", ", ".join(f"{k}({v})" for k, v in none_txt.most_common(20)))


if __name__ == "__main__":
    main()
