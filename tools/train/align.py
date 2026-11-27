"""Steps 3-4: line up each reference subtitle with the recognised Japanese and build ja-ko pairs.

    python tools/train/align.py      # cache/train/{subs,asr} -> cache/train/pairs.json + align_report.txt

Reference subtitles come from other video releases, so each file gets its own timing fix:
a scale (frame-rate conversions) and an offset that best line up "someone is talking"
between the subtitle cues and the recognised speech.
"""
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from realtime_subtitles.youtube_subs.server import is_moan, is_filler_ko  # noqa: E402

T = ROOT / "cache" / "train"
RES = 0.1                        # seconds per activity bin
LIMIT = 1500.0                   # only the first 25 min of audio was collected
SCALES = [1.0, 25 / 23.976, 23.976 / 25, 1.001, 1 / 1.001]
HALLU_RE = re.compile(r"(スパイシー|ご視聴|チャンネル登録|お菓子|字幕)")


# ---------------------------------------------------------------- reference subtitles
def _clean(text: str) -> str:
    text = html.unescape(re.sub(r"<br\s*/?>", "\n", text, flags=re.I))
    text = re.sub(r"<[^>]+>|\{[^}]*\}", "", text).replace(" ", " ")
    lines = [l.strip() for l in re.split(r"\n| / ", text) if l.strip()]
    return " / ".join(lines)


def parse_srt(t: str) -> list:
    cues = []
    for m in re.finditer(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)[^\n]*\n(.*?)(?=\n\s*\n|\Z)",
                         t, re.S):
        g = [int(x) for x in m.groups()[:8]]
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        txt = _clean(m.group(9))
        if txt:
            cues.append((s, e, txt))
    return cues


def parse_smi(t: str) -> list:
    syncs = [(int(m.group(1)) / 1000, m.group(2))
             for m in re.finditer(r"<SYNC\s+Start\s*=\s*\"?(\d+)\"?[^>]*>(.*?)(?=<SYNC|</BODY|\Z)", t, re.S | re.I)]
    cues = []
    for i, (s, body) in enumerate(syncs):
        txt = _clean(body)
        if not txt:
            continue
        e = syncs[i + 1][0] if i + 1 < len(syncs) else s + 3
        cues.append((s, min(e, s + 8), txt))
    return cues


def parse_ass(t: str) -> list:
    cues = []
    for m in re.finditer(r"^Dialogue:[^,]*,(\d+):(\d+):(\d+)\.(\d+),(\d+):(\d+):(\d+)\.(\d+),(?:[^,]*,){6}(.*)$", t, re.M):
        g = [int(x) for x in m.groups()[:8]]
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 100
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 100
        txt = _clean(m.group(9).replace("\\N", "\n"))
        if txt:
            cues.append((s, e, txt))
    return cues


def load_ref(path: Path) -> list:
    t = path.read_text(encoding="utf-8", errors="replace")
    fn = {".srt": parse_srt, ".smi": parse_smi, ".ass": parse_ass}[path.suffix.lower()]
    cues = sorted(fn(t))
    # drop title/credit cards: no Hangul, or credit words
    return [c for c in cues if re.search(r"[가-힣]", c[2]) and not re.search(r"자막|제작|번역|싱크|^[A-Z]+-\d+", c[2])]


# ---------------------------------------------------------------- recognised speech
def load_asr(path: Path) -> tuple:
    d = json.loads(path.read_text(encoding="utf-8"))
    segs = [(s["s"], s["e"], s["ja"].strip()) for s in d["segments"]
            if s["ja"].strip() and not is_moan(s["ja"]) and not HALLU_RE.search(s["ja"])]
    return segs, d["duration"]


def onsets(starts, length: float) -> np.ndarray:
    """Speech-start pattern: a small bump at every line start. Dialogue rhythm is distinctive,
    so this lines up much more sharply than 'talking or not' blocks do."""
    a = np.zeros(int(length / RES) + 1, np.float32)
    for s in starts:
        i = int(max(s, 0) / RES)
        if i < len(a):
            a[i] = 1.0
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)          # ~0.25 s blur
    return np.convolve(a, k, mode="same")


# Words that almost always come out the same way: a Japanese line with the left side and a
# Korean line with the right side are probably the same line. Rare words count for more.
KEYS = [
    (r"大丈夫", r"괜찮"), (r"緊張", r"긴장"), (r"ありがと", r"고마|고맙|감사"), (r"美味し|おいし", r"맛있|맛나"),
    (r"気持ちい|きもちい", r"기분 ?좋|기분좋|느낌 ?좋"), (r"恥ずかし", r"부끄|창피"), (r"可愛|かわい", r"귀여|예쁘|이쁘"),
    (r"痛い|いたい", r"아파|아프"), (r"ごめん", r"미안"), (r"すごい|凄い", r"대단|굉장|엄청|대박"),
    (r"やばい|ヤバい", r"대박|미쳤|위험|큰일"), (r"ダメ|だめ", r"안 ?돼|안 ?되"), (r"お願い", r"부탁"),
    (r"好き", r"좋아해|좋아요|좋아"), (r"初めまして", r"처음 뵙"), (r"おはよう", r"안녕"), (r"こんにちは", r"안녕"),
    (r"いただきます", r"잘 ?먹겠"), (r"お疲れ", r"수고"), (r"イク|イッ|イキ|いく|行く", r"가요|간다|갈 ?것|가버|갈게|가도"),
    (r"出る|出そう|出ちゃ|出す", r"싸|나와|나온"), (r"キス", r"키스|뽀뽀"), (r"おっぱい|胸", r"가슴|젖"),
    (r"本当|ほんと", r"정말|진짜"), (r"何", r"뭐|무슨"), (r"雨", r"비"), (r"車", r"차"), (r"カメラ", r"카메라"),
    (r"写真", r"사진"), (r"お風呂", r"목욕|욕조|탕"), (r"ホテル", r"호텔"), (r"旦那|夫", r"남편"), (r"先生", r"선생"),
    (r"彼氏", r"남자친구|남친"), (r"結婚", r"결혼"), (r"仕事", r"일|회사"), (r"名前", r"이름"), (r"歳", r"살"),
]


def best_timing(ref: list, segs: list, dur: float) -> tuple:
    """(scale, offset, score): ref time t maps to t*scale + offset.

    Votes from lines that share a keyword (see KEYS): each such Japanese/Korean pair votes for
    the gap between their start times. Recognised start times are coarse, so votes are counted in
    a 3 s window. score = peak / average (noise ~1)."""
    bins = np.arange(-300, 300.5, 0.5)
    best = (1.0, 0.0, 0.0)
    for sc in SCALES:
        h = np.zeros(len(bins) - 1)
        for jre, kre in KEYS:
            A = np.array([s for s, _, ja in segs if re.search(jre, ja)])
            R = np.array([s * sc for s, _, ko in ref if re.search(kre, ko)])
            if not len(A) or not len(R):
                continue
            d = (A[:, None] - R[None, :]).ravel()
            d = d[np.abs(d) < 300]
            hk, _ = np.histogram(d, bins)
            h += hk / np.sqrt(len(A) * len(R))
        hs = np.convolve(h, np.ones(6), mode="same")
        k = int(np.argmax(hs))
        score = float(hs[k] / (hs.mean() + 1e-9))
        if score > best[2]:
            best = (sc, float(bins[k] + 0.25), score)
    # refine: median of the keyword votes near the peak
    sc, off, score = best
    near = []
    for jre, kre in KEYS:
        A = [s for s, _, ja in segs if re.search(jre, ja)]
        R = [s * sc for s, _, ko in ref if re.search(kre, ko)]
        near += [x - y - off for x in A for y in R if abs(x - y - off) < 3.0]
    if len(near) >= 5:
        off += float(np.median(near))
    return sc, off, score


# ---------------------------------------------------------------- pairing
def pair(ref: list, segs: list, scale: float, offset: float, dur: float) -> list:
    lines = [(s * scale + offset, e * scale + offset, ko) for s, e, ko in ref]
    owner = {}
    for g in segs:
        c = (g[0] + g[1]) / 2
        best, bd = None, 3.0
        for i, (s2, e2, _) in enumerate(lines):
            d = 0.0 if s2 - 0.5 <= c <= e2 + 0.5 else min(abs(c - s2), abs(c - e2))
            if d < bd:
                best, bd = i, d
        if best is not None:
            owner.setdefault(best, []).append(g)
    out = []
    for i, (s2, e2, ko) in enumerate(lines):
        if i not in owner or s2 < 0 or e2 > min(dur, LIMIT):
            continue
        ja = "".join(g[2] for g in owner[i]).strip()
        if is_moan(ja) or is_filler_ko(ko):              # never translated, never scored
            continue
        ko_len, ja_len = len(re.sub(r"[\s/-]", "", ko)), len(re.sub(r"\s", "", ja))
        if not (0.3 <= ja_len / max(ko_len, 1) <= 3.0):        # one side clearly incomplete
            continue
        out.append({"t": round(s2, 2), "e": round(e2, 2), "ja": ja, "ko": ko})
    return out


def main() -> None:
    report, allpairs = [], {}
    for sub in sorted((T / "subs").iterdir()):
        a = T / "asr" / f"{sub.stem}.json"
        if not a.exists():
            continue
        ref = load_ref(sub)
        segs, dur = load_asr(a)
        sc, off, score = best_timing(ref, segs, dur)
        pairs = pair(ref, segs, sc, off, dur)
        ok = score >= 4.0 and len(pairs) >= 20
        report.append(f"{sub.stem:10s} ref={len(ref):4d} asr={len(segs):4d} scale={sc:.4f} "
                      f"offset={off:+7.1f}s score={score:.2f} pairs={len(pairs):4d} {'OK' if ok else 'SKIP'}")
        if ok:
            allpairs[sub.stem] = pairs
    (T / "pairs.json").write_text(json.dumps(allpairs, ensure_ascii=False, indent=0), encoding="utf-8")
    (T / "align_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("videos kept:", len(allpairs), "pairs:", sum(len(v) for v in allpairs.values()))


if __name__ == "__main__":
    main()
