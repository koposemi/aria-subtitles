"""Speech regions with the Silero VAD neural detector (onnxruntime), for testing timing ideas.

    cache/tools/vadenv/Scripts/python tools/train/silero_regions.py
Needs cache/train/asr/<code>.f32 (made by whisperjav_check.py) and cache/tools/silero/silero_vad.onnx.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "cache" / "train"
SR, N, CTX = 16000, 512, 64


def regions(audio, sess, on=0.5, off=0.35, min_sil=0.25, min_speech=0.15):
    state = np.zeros((2, 1, 128), np.float32)
    ctx = np.zeros((1, CTX), np.float32)
    sr = np.array(SR, dtype=np.int64)
    probs = []
    for i in range(0, len(audio) - N, N):
        x = np.concatenate([ctx, audio[None, i:i + N]], axis=1)
        p, state = sess.run(None, {"input": x, "state": state, "sr": sr})
        ctx = x[:, -CTX:]
        probs.append(float(p[0][0]))
    out, start, quiet = [], None, 0
    step = N / SR
    for k, p in enumerate(probs):
        t = k * step
        if start is None and p >= on:
            start, quiet = t, 0
        elif start is not None:
            if p < off:
                quiet += 1
                if quiet * step >= min_sil:
                    end = t - quiet * step + step
                    if end - start >= min_speech:
                        out.append((round(start, 3), round(end, 3)))
                    start, quiet = None, 0
            else:
                quiet = 0
    if start is not None:
        out.append((round(start, 3), round(len(probs) * step, 3)))
    return out


def main():
    sess = ort.InferenceSession(str(ROOT / "cache" / "tools" / "silero" / "silero_vad.onnx"))
    for f in sorted((T / "asr").glob("*.f32")):
        dst = T / "vad" / f"{f.stem}.json"
        if dst.exists():
            continue
        audio = np.fromfile(f, dtype=np.float32)
        r = regions(audio, sess)
        dst.write_text(json.dumps(r))
        print(f.stem, len(r), "regions", flush=True)


if __name__ == "__main__":
    main()
