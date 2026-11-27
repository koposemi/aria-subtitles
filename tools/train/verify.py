"""Check every ja-ko pair with an LLM: is the Korean subtitle a translation of this Japanese?

    python tools/train/verify.py      # pairs.json -> verify.json (cache) -> pairs_verified.json

Resumable (verdicts are cached per batch). Uses gpt-oss-120b first so the qwen quota stays free
for watching; falls back to qwen when gpt-oss gives nothing usable.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from realtime_subtitles.youtube_subs.groq_client import GroqPool, GroqUnavailable  # noqa: E402

T = ROOT / "cache" / "train"
BATCH = 40
MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
PROMPT = (
    "아래는 일본 영상의 '음성 인식 일본어'와 '사람이 만든 한국어 자막'을 시간으로 짝지은 목록입니다. "
    "짝짓기가 틀린 것을 찾아야 합니다. 각 번호마다 한국어 자막이 그 일본어 대사의 번역이면 1, "
    "다른 대사(앞뒤 줄과 섞임, 내용이 다름)이거나 한쪽 내용 대부분이 빠졌으면 0. "
    "의역, 말투 차이, 음성 인식의 사소한 오타는 1로 봅니다. 성인 영상이라 노골적 표현이 있어도 판정만 하세요. "
    '출력은 JSON 객체 하나만: {"1": 1, "2": 0, ...}'
)


def judge(pool, items) -> dict:
    text = "\n".join(f"{i + 1}. 일본어: {p['ja']}\n   한국어: {p['ko']}" for i, p in enumerate(items))
    for model in MODELS:
        body = {"model": model, "temperature": 0, "max_tokens": 1500,
                "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]}
        body.update({"reasoning_effort": "none"} if model.startswith("qwen/") else {"reasoning_effort": "low"})
        for _ in range(20):
            try:
                data = pool.call(model, "/chat/completions", json.dumps(body).encode(), "application/json",
                                 wait=True)
                break
            except GroqUnavailable as e:
                if "400" in str(e):            # bad request for this model: try the next one
                    data = None
                    break
                print("  waiting:", e, flush=True)
                time.sleep(30)
        else:
            continue
        if data is None:
            continue
        try:
            c = data["choices"][0]["message"]["content"] or ""
            obj = json.loads(c[c.find("{"):c.rfind("}") + 1])
            verdict = {i: int(obj[str(i + 1)]) for i in range(len(items)) if str(i + 1) in obj}
            if len(verdict) >= len(items) * 0.9:
                return verdict
        except (KeyError, IndexError, ValueError, TypeError):
            pass
    return {}


def _key(p) -> str:
    import hashlib
    return hashlib.sha1((p["ja"] + "|" + p["ko"]).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else T / "pairs.json"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else T / "pairs_verified.json"
    pairs = json.loads(src.read_text(encoding="utf-8"))
    vpath = T / "verdicts.json"                       # verdict per pair text: reused across runs
    verdicts = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    old = T / "verify.json"                           # first run cached by batch position
    if old.exists() and not verdicts:
        base = json.loads((T / "pairs.json").read_text(encoding="utf-8"))
        for key, v in json.loads(old.read_text(encoding="utf-8")).items():
            code, i = key.rsplit(":", 1)
            for j, p in enumerate(base[code][int(i):int(i) + BATCH]):
                if str(j) in v:
                    verdicts[_key(p)] = v[str(j)]
    todo = [p for items in pairs.values() for p in items if _key(p) not in verdicts]
    print(f"{len(todo)} pairs to check ({len(verdicts)} verdicts reused)", flush=True)
    pool = GroqPool()
    for n in range(0, len(todo), BATCH):
        chunk = todo[n:n + BATCH]
        v = judge(pool, chunk)
        for j, p in enumerate(chunk):
            if j in v:
                verdicts[_key(p)] = v[j]
        vpath.write_text(json.dumps(verdicts), encoding="utf-8")
        print(f"[{n // BATCH + 1}/{(len(todo) + BATCH - 1) // BATCH}] {sum(v.values())}/{len(v)} ok", flush=True)
        time.sleep(2)
    out, kept, total = {}, 0, 0
    for code, items in pairs.items():
        out[code] = [p for p in items if verdicts.get(_key(p)) == 1]
        kept += len(out[code])
        total += len(items)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"VERIFY DONE: kept {kept}/{total} -> {dst.name}")


if __name__ == "__main__":
    main()
