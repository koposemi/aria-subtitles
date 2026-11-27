"""Reference translations for the translation prompt.

examples.json holds Japanese/Korean subtitle pairs taken from human-made subtitles (built by
tools/train/). For each batch of lines to translate, the most similar Japanese lines are found
(character-bigram TF-IDF, numpy-free so it runs on Termux) and their human translations are
shown to the model as style/term examples. style.txt (optional) adds the translator's style
rules. Off unless ARIA_EXAMPLES=1 (measured gain is small; see tools/train/evaluate.py).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENABLED = os.environ.get("ARIA_EXAMPLES", "0") == "1"
PER_LINE = 2
MAX_TOTAL = 6

_bank: list | None = None
_index: dict = {}
_idf: dict = {}
_norm: list = []
_style: str | None = None


def _grams(text: str) -> list:
    t = re.sub(r"[\s、。．,.!！?？…・「」『』()（）]", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)] or ([t] if t else [])


def _load() -> None:
    global _bank, _style
    if _bank is not None:
        return
    _bank = []
    path = HERE / "examples.json"
    if path.exists():
        try:
            _bank = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _bank = []
    df = defaultdict(int)
    grams = []
    for item in _bank:
        g = set(_grams(item["ja"]))
        grams.append(g)
        for x in g:
            df[x] += 1
    n = max(len(_bank), 1)
    _idf.update({x: math.log(1 + n / c) for x, c in df.items()})
    for i, g in enumerate(grams):
        for x in g:
            _index.setdefault(x, []).append(i)
        _norm.append(math.sqrt(sum(_idf[x] ** 2 for x in g)) or 1.0)
    style = HERE / "style.txt"
    _style = style.read_text(encoding="utf-8").strip() if style.exists() else ""


def similar(line: str, k: int = PER_LINE, min_score: float = 0.5) -> list:
    _load()
    g = set(_grams(line))
    if not g or not _bank:
        return []
    qn = math.sqrt(sum(_idf.get(x, 0.0) ** 2 for x in g)) or 1.0
    score = defaultdict(float)
    for x in g:
        w = _idf.get(x)
        if w:
            for i in _index[x]:
                score[i] += w * w
    ranked = sorted(((s / (qn * _norm[i]), i) for i, s in score.items()), reverse=True)
    return [_bank[i] for s, i in ranked[:k] if s >= min_score]


def prompt_text(lines: list) -> str:
    """Extra system-prompt text for these lines: style rules + similar human translations."""
    if not ENABLED:
        return ""
    _load()
    picked, seen = [], set()
    for line in lines:
        for item in similar(line):
            if item["ja"] not in seen:
                seen.add(item["ja"])
                picked.append(item)
    picked = picked[:MAX_TOTAL]
    out = ""
    if _style:
        out += "\n\n번역 스타일 (사람이 만든 자막 기준):\n" + _style
    if picked:
        out += ("\n\n참고 번역 예시 (비슷한 대사를 사람이 번역한 것. 말투와 용어를 참고하되 그대로 베끼지 말 것):\n"
                + "\n".join(f"- {p['ja']} => {p['ko']}" for p in picked))
    return out
