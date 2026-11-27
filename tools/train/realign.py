"""Re-time each reference subtitle to its audio with alass (handles different cuts: piecewise
offsets), then build pairs from the re-timed subtitles.

    python tools/train/realign.py      # cache/train/{subs,asr,audio} -> cache/train/alass/*.srt
                                       #                               -> cache/train/pairs_alass.json

alass: https://github.com/kaegi/alass (v2.0.0 Windows build in cache/tools/alass).
The global offset from align.py is applied first so alass only has to fix the local splits,
and lines far outside the collected 25 minutes are left out.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A                                                  # noqa: E402

T = ROOT / "cache" / "train"
ALASS = ROOT / "cache" / "tools" / "alass" / "x" / "alass-windows64" / "bin" / "alass-cli.exe"
SPLIT_PENALTY = os.environ.get("ALASS_SPLIT_PENALTY", "7")


def ts(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues, path: Path) -> None:
    path.write_text("\n".join(f"{i}\n{ts(s)} --> {ts(e)}\n{t.replace(' / ', chr(10))}\n"
                              for i, (s, e, t) in enumerate(cues, 1)), encoding="utf-8")


def main() -> None:
    out_dir = T / "alass"
    out_dir.mkdir(exist_ok=True)
    env = dict(os.environ, ALASS_FFMPEG_PATH=shutil.which("ffmpeg") or "",
               ALASS_FFPROBE_PATH=shutil.which("ffprobe") or "")
    report, allpairs = [], {}
    for sub in sorted((T / "subs").iterdir()):
        code = sub.stem
        asr = T / "asr" / f"{code}.json"
        if not asr.exists():
            continue
        ref = A.load_ref(sub)
        segs, dur = A.load_asr(asr)
        sc, off, score = A.best_timing(ref, segs, dur)
        pre = [(s * sc + off, e * sc + off, t) for s, e, t in ref]
        pre = [c for c in pre if -60 < c[0] < A.LIMIT + 120]
        inp, outp = out_dir / f"{code}.in.srt", out_dir / f"{code}.srt"
        write_srt(pre, inp)
        r = subprocess.run([str(ALASS), str(T / "audio" / f"{code}.m4a"), str(inp), str(outp),
                            "--split-penalty", SPLIT_PENALTY, "--encoding-inc", "utf-8"],
                           capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not outp.exists():
            report.append(f"{code:10s} alass failed: {(r.stderr or r.stdout)[-200:]}")
            continue
        timed = A.parse_srt(outp.read_text(encoding="utf-8", errors="replace"))
        moved = sum(1 for a, b in zip(pre, timed) if abs(a[0] - b[0]) > 1.0)
        pairs = A.pair(timed, segs, 1.0, 0.0, dur)
        allpairs[code] = pairs
        report.append(f"{code:10s} lines={len(pre):4d} moved>1s={moved:4d} pairs={len(pairs):4d}")
        print(report[-1], flush=True)
    (T / "pairs_alass.json").write_text(json.dumps(allpairs, ensure_ascii=False, indent=0), encoding="utf-8")
    (T / "realign_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("pairs:", sum(len(v) for v in allpairs.values()))


if __name__ == "__main__":
    main()
