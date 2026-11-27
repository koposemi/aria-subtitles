"""Step 6: before/after translation quality on the hold-out videos.

    python tools/train/evaluate.py [lines_per_video]

For each hold-out video, the recognised Japanese of N aligned lines is translated twice, in
batches of 8 like the server does: once with the plain prompt, once with style rules + similar
human translations (examples.py). Each output is scored against the human subtitle with chrF
(character n-gram F-score, 0-100; higher = closer to the human translation).
Results: cache/train/eval.json and eval_report.txt.
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from realtime_subtitles.youtube_subs import examples                     # noqa: E402
from realtime_subtitles.youtube_subs.server import is_moan, is_filler_ko  # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import GroqPool, GroqUnavailable  # noqa: E402

T = ROOT / "cache" / "train"
BATCH = 8


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    h, r = re.sub(r"[\s/\-.,!?…~]", "", hyp), re.sub(r"[\s/\-.,!?…~]", "", ref)
    precs, recs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        m = sum((hc & rc).values())
        precs.append(m / sum(hc.values()))
        recs.append(m / sum(rc.values()))
    if not precs:
        return 0.0
    p, rr = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + rr == 0 else 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def translate_all(pool, items, use_examples: bool) -> tuple:
    examples.ENABLED = use_examples
    out, models, ctx = [], Counter(), []
    for i in range(0, len(items), BATCH):
        batch = [p["ja"] for p in items[i:i + BATCH]]
        while True:
            try:
                res = pool.translate(batch, ctx[-3:])
                break
            except GroqUnavailable as e:
                print("  waiting:", e, flush=True)
                time.sleep(30)
        models[pool.last_model] += 1
        res = [r or "" for r in res]
        out += res
        ctx += list(zip(batch, res))
        time.sleep(2.5)                      # stay under the per-minute limits
    return out, models


def main() -> None:
    per_video = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    src = next(T / n for n in ("pairs_merged.json", "pairs_verified.json", "pairs.json") if (T / n).exists())
    pairs = json.loads(src.read_text(encoding="utf-8"))
    print("pairs from", src.name)
    hold = (T / "holdout.txt").read_text(encoding="utf-8").split()
    pool = GroqPool()
    results, report = {}, []
    tot = {"base": [], "new": []}
    for code in hold:
        # moans / back-channel sounds ("응") are never translated, so they are not scored either
        items = [p for p in pairs.get(code, []) if not is_moan(p["ja"]) and not is_filler_ko(p["ko"])]
        if not items:
            report.append(f"{code}: no pairs")
            continue
        step = max(1, len(items) // per_video)
        items = items[::step][:per_video]
        base, mb = translate_all(pool, items, False)
        new, mn = translate_all(pool, items, True)
        rows = []
        for p, b, n in zip(items, base, new):
            rows.append({"t": p["t"], "ja": p["ja"], "ref": p["ko"], "base": b, "new": n,
                         "s_base": round(chrf(b, p["ko"]), 1), "s_new": round(chrf(n, p["ko"]), 1)})
        results[code] = rows
        sb = sum(r["s_base"] for r in rows) / len(rows)
        sn = sum(r["s_new"] for r in rows) / len(rows)
        tot["base"] += [r["s_base"] for r in rows]
        tot["new"] += [r["s_new"] for r in rows]
        report.append(f"{code}: {len(rows)} lines  chrF before {sb:.1f} -> after {sn:.1f}  "
                      f"(models before {dict(mb)}, after {dict(mn)})")
        print(report[-1], flush=True)
    if tot["base"]:
        b, n = sum(tot["base"]) / len(tot["base"]), sum(tot["new"]) / len(tot["new"])
        wins = sum(1 for x, y in zip(tot["base"], tot["new"]) if y > x + 2)
        loss = sum(1 for x, y in zip(tot["base"], tot["new"]) if x > y + 2)
        report.append(f"ALL: {len(tot['base'])} lines  chrF before {b:.1f} -> after {n:.1f}  "
                      f"(better {wins}, worse {loss}, same {len(tot['base']) - wins - loss})")
    (T / "eval.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (T / "eval_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[-1:]))


if __name__ == "__main__":
    main()
