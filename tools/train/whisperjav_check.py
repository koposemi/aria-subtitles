"""WhisperJAV-style clean-up steps, measured against the human subtitles of the collected videos.

    python tools/train/whisperjav_check.py

Per video: saved Groq words -> subtitle lines (server's words_to_lines) -> optional steps:
  align  : the server's current start alignment to loud regions (_align_cue_starts)
  repair : a line much longer than its text needs gets its start pulled in (end kept)
  snap   : start moved to the nearest speech onset found by the speech detector
  clean  : repetition collapse, impossible-speed lines dropped, known hallucinations dropped
Metrics (human lines = reference, after each video's global offset):
  hit<=0.5s / hit<=1s : share of human lines with one of our line starts that close
  precision<=1s       : share of OUR lines that start within 1 s of a human line (fewer ghost lines -> higher)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import align as A                                                              # noqa: E402
from realtime_subtitles.youtube_subs import server as S                      # noqa: E402
from realtime_subtitles.youtube_subs.groq_client import fix_word_times, HALLUCINATION_RE  # noqa: E402

T = ROOT / "cache" / "train"
SR = 16000
SKIP = {"STAR-774", "MIMK-031"}            # wrong release / machine-made subtitle
WIN = 30.0


def load_audio(code: str) -> np.ndarray:
    raw = T / "asr" / f"{code}.f32"
    if not raw.exists():
        subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-loglevel", "error", "-y", "-i",
                        str(T / "audio" / f"{code}.m4a"), "-ac", "1", "-ar", str(SR), "-f", "f32le", str(raw)],
                       check=True)
    return np.fromfile(raw, dtype=np.float32)


def core(t: str) -> str:
    return re.sub(r"[\s、。．,.!！?？…・「」『』()（）ー〜～]", "", t)


# ---------------------------------------------------------------- candidate steps
def step_align(lines, audio):
    out = []
    for w in sorted({int(s // WIN) for s, _, _ in lines}):
        chunk = audio[int(w * WIN * SR):int((w + 1) * WIN * SR)]
        mine = [l for l in lines if int(l[0] // WIN) == w]
        out += S.JobManager._align_cue_starts(mine, chunk, w * WIN)
    return out


def step_repair(lines, per_char=0.2, base=0.6, slack=0.8):
    out = []
    for s, e, t in lines:
        need = base + per_char * len(core(t))
        if e - s > need + slack:
            s = e - need
        out.append((s, e, t))
    return out


def onsets_of(audio, min_silence_ms=250, pad_ms=60):
    out = []
    for w in range(int(len(audio) / SR // WIN) + 1):
        chunk = audio[int(w * WIN * SR):int((w + 1) * WIN * SR)]
        if len(chunk) < SR:
            continue
        for r in S.speech_regions(chunk, min_silence_ms=min_silence_ms, pad_ms=pad_ms):
            out.append((w * WIN + r["start"] / SR, w * WIN + r["end"] / SR))
    return out


class Regions(list):
    silero: list = []


def step_snap(lines, regions, before=0.8, after=0.6):
    starts = np.array([a for a, _ in regions]) if regions else np.array([])
    out = []
    for s, e, t in lines:
        if len(starts):
            cand = starts[(starts >= s - before) & (starts <= s + after) & (starts < e - 0.15)]
            if len(cand):
                s = float(cand[np.argmin(np.abs(cand - s))])
        out.append((s, e, t))
    return out


REPEAT_RE = re.compile(r"(.{1,4}?)\1{2,}")


def step_clean(lines):
    out, prev = [], None
    for s, e, t in lines:
        t = REPEAT_RE.sub(r"\1\1", t)                     # "やばいやばいやばいやばい" -> twice
        c = core(t)
        if not c or S.is_moan(t) or HALLUCINATION_RE.search(t):
            continue
        if e - s < 0.25 and len(c) > 6:                   # cannot be said that fast
            continue
        if prev is not None and c == core(prev[2]) and s - prev[1] < 2.0:
            continue                                      # immediate repeat of the same line
        out.append((s, e, t))
        prev = (s, e, t)
    return out


# ---------------------------------------------------------------- scoring
def score(lines, refs):
    ls = np.array(sorted(l[0] for l in lines))
    rs = np.array(sorted(refs))
    def nearest(a, b):
        idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
        return np.minimum(np.abs(a - b[idx - 1]), np.abs(a - b[idx]))
    hr, ho = nearest(rs, ls), nearest(ls, rs)
    return (hr <= 0.5).mean(), (hr <= 1.0).mean(), (ho <= 1.0).mean(), len(ls), len(rs)


def step_ghost(lines, regions, min_cover=0.2):
    """Drop lines that hardly overlap any detected speech (sound over moans/music)."""
    out = []
    for s, e, t in lines:
        cover = sum(max(0.0, min(e, b) - max(s, a)) for a, b in regions)
        if cover >= min_cover * max(e - s, 0.3):
            out.append((s, e, t))
    return out


def silero(code):
    f = T / "vad" / f"{code}.json"
    return [tuple(x) for x in json.loads(f.read_text())] if f.exists() else []


def main() -> None:
    variants = {
        "words only": lambda L, a, R: L,
        "+align (now)": lambda L, a, R: step_align(L, a),
        "+repair": lambda L, a, R: step_repair(L),
        "+snap": lambda L, a, R: step_snap(L, R),
        "+repair+snap": lambda L, a, R: step_snap(step_repair(L), R),
        "+repair+snap+clean": lambda L, a, R: step_clean(step_snap(step_repair(L), R)),
        "+align+repair+clean": lambda L, a, R: step_clean(step_repair(step_align(L, a))),
        "+snap(silero)": lambda L, a, R: step_snap(L, R.silero),
        "+snap(silero wide)": lambda L, a, R: step_snap(L, R.silero, 1.2, 0.8),
        "+ghost(silero)": lambda L, a, R: step_ghost(L, R.silero),
        "+snap+ghost(silero)": lambda L, a, R: step_ghost(step_snap(L, R.silero), R.silero),
        "+snap+ghost+clean(silero)": lambda L, a, R: step_clean(step_ghost(step_snap(L, R.silero), R.silero)),
    }
    tot = {k: np.zeros(5) for k in variants}
    for sub in sorted((T / "subs").iterdir()):
        code = sub.stem
        if code in SKIP:
            continue
        d = json.loads((T / "asr" / f"{code}.json").read_text(encoding="utf-8"))
        ref = A.load_ref(sub)
        segs, dur = A.load_asr(T / "asr" / f"{code}.json")
        sc, off, _ = A.best_timing(ref, segs, dur)
        refs = [s * sc + off for s, e, ko in ref
                if 0 <= s * sc + off <= min(dur, A.LIMIT) and not S.is_filler_ko(ko)]
        audio = load_audio(code)
        regions = Regions(onsets_of(audio))
        regions.silero = silero(code)
        words = fix_word_times([(w["s"], w["e"], w["w"]) for w in d["words"]])
        base = [l for l in S.words_to_lines(words, 0.0) if not S.is_moan(l[2])]
        for name, fn in variants.items():
            h05, h1, p1, nl, nr = score(fn(base, audio, regions), refs)
            tot[name] += np.array([h05 * nr, h1 * nr, p1 * nl, nl, nr])
        print(code, "done", flush=True)
    print(f"\n{'variant':22s} {'hit<=0.5s':>9s} {'hit<=1s':>8s} {'precision<=1s':>14s} {'lines':>6s}")
    for name, v in tot.items():
        print(f"{name:22s} {v[0] / v[4]:9.1%} {v[1] / v[4]:8.1%} {v[2] / v[3]:14.1%} {int(v[3]):6d}")


if __name__ == "__main__":
    main()
