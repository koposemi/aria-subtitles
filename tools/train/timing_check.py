"""How close do our subtitle start times land to the human subtitles? Old vs new timing.

    python tools/train/timing_check.py

Uses the saved Groq results (cache/train/asr) of the 26 collected videos:
  old = the server's previous method: Groq segment times, segments merged into sentences
  new = word times (fix_word_times) -> words_to_lines, moans/back-channel lines dropped
The human subtitles (shifted by each video's global offset from align.py) are the reference:
for every human line, the nearest of our line starts within 3 s is its match.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A                                                              # noqa: E402
from realtime_subtitles.youtube_subs import server as S                      # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import fix_word_times       # noqa: E402

T = ROOT / "cache" / "train"
SKIP = {"STAR-774"}                        # subtitle made for another release (0 verified pairs)


class Seg:
    def __init__(self, s, e, t):
        self.start, self.end, self.text = s, e, t


def stats(starts, refs):
    errs = []
    for r in refs:
        best = min(starts, key=lambda s: abs(s - r), default=None)
        if best is not None and abs(best - r) <= 3.0:
            errs.append(best - r)
    if not errs:
        return None
    a = [abs(x) for x in errs]
    return {"matched": len(errs), "refs": len(refs), "median": statistics.median(a),
            "le05": sum(x <= 0.5 for x in a) / len(a), "le1": sum(x <= 1.0 for x in a) / len(a),
            "bias": statistics.median(errs)}


def main() -> None:
    rows, all_old, all_new = [], [], []
    for sub in sorted((T / "subs").iterdir()):
        code = sub.stem
        if code in SKIP:
            continue
        d = json.loads((T / "asr" / f"{code}.json").read_text(encoding="utf-8"))
        ref = A.load_ref(sub)
        segs_f, dur = A.load_asr(T / "asr" / f"{code}.json")
        sc, off, _ = A.best_timing(ref, segs_f, dur)
        refs = [s * sc + off for s, e, ko in ref
                if 0 <= s * sc + off <= min(dur, A.LIMIT) and not S.is_filler_ko(ko)]
        # old: segment times, merged like the old server did
        old_segs = [Seg(s["s"], s["e"], s["ja"]) for s in d["segments"] if s["ja"].strip()]
        old_lines = [l for l in S.JobManager._merge(old_segs, 0.0) if not S.is_moan(l[2])]
        # new: word times
        words = fix_word_times([(w["s"], w["e"], w["w"]) for w in d["words"]])
        new_lines = [l for l in S.words_to_lines(words, 0.0) if not S.is_moan(l[2])]
        o, n = stats([l[0] for l in old_lines], refs), stats([l[0] for l in new_lines], refs)
        if not o or not n:
            continue
        rows.append((code, o, n))
        for r, lines in ((o, all_old), (n, all_new)):
            lines.append(r)
    print(f"{'video':10s} | {'old median':>10s} {'<=0.5s':>6s} {'<=1s':>5s} | {'new median':>10s} {'<=0.5s':>6s} {'<=1s':>5s} | bias old/new")
    for code, o, n in rows:
        print(f"{code:10s} | {o['median']:9.2f}s {o['le05']:6.0%} {o['le1']:5.0%} | {n['median']:9.2f}s "
              f"{n['le05']:6.0%} {n['le1']:5.0%} | {o['bias']:+.2f}/{n['bias']:+.2f}")
    def pooled(rs, k):
        tot = sum(r["matched"] for r in rs)
        return sum(r[k] * r["matched"] for r in rs) / tot
    print(f"{'ALL':10s} | {statistics.median([r['median'] for r in all_old]):9.2f}s {pooled(all_old, 'le05'):6.0%} "
          f"{pooled(all_old, 'le1'):5.0%} | {statistics.median([r['median'] for r in all_new]):9.2f}s "
          f"{pooled(all_new, 'le05'):6.0%} {pooled(all_new, 'le1'):5.0%}")


if __name__ == "__main__":
    main()
