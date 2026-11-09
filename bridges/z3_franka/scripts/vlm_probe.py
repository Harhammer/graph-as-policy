"""Send one real camera image to an OpenRouter model, like GaP's VLM step does.

    OPENROUTER_API_KEY=... .venv/bin/python scripts/vlm_probe.py MODEL [MODEL ...]

Prints the HTTP status, the provider and the answer (or the error body) per model — the
error body says whether a 429 is the daily free quota or an upstream overload.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

FRAME = "/tmp/bridge_frame.npz"  # written by z3-gap-fake-server --save-frame


def probe(model: str, image_b64: str, key: str) -> None:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List the objects on the table in one short line."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 80,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        msg = data["choices"][0]["message"]
        # Reasoning models can leave ``content`` null and answer in ``reasoning`` — GaP's
        # VLM tool reads ``content``, so such a model is unusable for it as-is.
        answer = msg.get("content") or f"[content empty; reasoning: {msg.get('reasoning')}]"
        answer = answer.strip().replace("\n", " ")
        print(f"{model}: OK {time.time() - t:.1f}s via {data.get('provider')}: {answer[:120]}")
    except urllib.error.HTTPError as e:
        print(f"{model}: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")


def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    rgb = np.load(FRAME)["rgb"]
    ok, jpg = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    image_b64 = base64.b64encode(jpg.tobytes()).decode()
    for model in sys.argv[1:] or ["google/gemma-4-31b-it:free"]:
        probe(model, image_b64, key)


if __name__ == "__main__":
    main()
