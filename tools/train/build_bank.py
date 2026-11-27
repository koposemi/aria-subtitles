"""Step 5: example bank (+ style rules) from the training pairs.

    python tools/train/build_bank.py            # pairs.json minus holdout -> src/.../examples.json
    python tools/train/build_bank.py --style    # also ask the LLM for style rules -> style.txt

Hold-out videos (cache/train/holdout.txt) never go into the bank, so the evaluation is fair.
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from realtime_subtitles.youtube_subs.server import is_moan, is_filler_ko  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import KEYS                                            # noqa: E402

T = ROOT / "cache" / "train"
PKG = ROOT / "src" / "realtime_subtitles" / "youtube_subs"


def clean_ko(ko: str) -> str:
    parts = [re.sub(r"^\s*-\s*", "", p).strip() for p in ko.split(" / ")]
    return " ".join(p for p in parts if p)


def main() -> None:
    src = next(T / n for n in ("pairs_merged.json", "pairs_verified.json", "pairs.json") if (T / n).exists())
    pairs = json.loads(src.read_text(encoding="utf-8"))
    print("pairs from", src.name)
    hold = set((T / "holdout.txt").read_text(encoding="utf-8").split())
    bank, seen = [], set()
    for code, items in pairs.items():
        if code in hold:
            continue
        for p in items:
            ja, ko = p["ja"].strip(), clean_ko(p["ko"])
            if not (3 <= len(ja) <= 60 and 2 <= len(ko) <= 60) or is_moan(ja) or is_filler_ko(ko) or ja in seen:
                continue
            if not 0.5 <= len(ja) / len(re.sub(r"\s", "", ko)) <= 2.2 or re.search(r"[\[\]()（）]", ko):
                continue
            # a word that almost always shows up in the translation is missing: probably misaligned
            if any(re.search(j, ja) and not re.search(k, ko) for j, k in KEYS):
                continue
            seen.add(ja)
            bank.append({"ja": ja, "ko": ko})
    (PKG / "examples.json").write_text(json.dumps(bank, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"examples.json: {len(bank)} pairs from {len([c for c in pairs if c not in hold])} videos")

    if "--style" in sys.argv:
        from realtime_subtitles.youtube_subs.groq_client import GroqPool
        random.seed(1)
        sample = random.sample(bank, min(180, len(bank)))
        prompt = ("아래는 사람이 만든 일본 성인영상 한국어 자막의 일본어 원문(음성 인식) => 한국어 번역 쌍입니다. "
                  "이 번역가의 스타일을 다른 번역가가 따라 할 수 있게 한국어 규칙 6~8개로 정리하세요. "
                  "말투(존댓말/반말 선택 기준), 감탄사·신음 처리, 성적 표현의 수위와 단어 선택, 호칭, 문장 길이, "
                  "자주 쓰는 표현을 구체적으로. 각 규칙은 '- '로 시작하는 한 줄. 규칙만 출력.\n\n"
                  + "\n".join(f"{p['ja']} => {p['ko']}" for p in sample))
        pool = GroqPool()
        body = {"model": "openai/gpt-oss-120b", "temperature": 0.2, "max_tokens": 1200,
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": prompt}]}
        data = pool.call(body["model"], "/chat/completions", json.dumps(body).encode(), "application/json")
        rules = data["choices"][0]["message"]["content"].strip()
        rules = "\n".join(l for l in rules.splitlines() if l.strip().startswith("-"))
        (PKG / "style.txt").write_text(rules + "\n", encoding="utf-8")
        print("style.txt:\n" + rules)


if __name__ == "__main__":
    main()
