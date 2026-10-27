"""Gemini review of every composite. Resumable: results are appended to data/review/gemini.jsonl."""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .review_prompt import PROMPT

MODELS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite"]
OUT = config.REPO_ROOT / "data/review/gemini.jsonl"


def call(model, key, jpeg_b64):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_b64}}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def main():
    key = os.environ["GEMINI_API_KEY"]
    index = json.loads((config.REPO_ROOT / "cache/composites/index.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["key"] for l in OUT.read_text().splitlines() if l.strip()}
    model_i = 0
    with OUT.open("a") as f:
        for row in index:
            if row["key"] in done:
                continue
            b64 = base64.b64encode((config.REPO_ROOT / row["file"]).read_bytes()).decode()
            for attempt in range(8):
                try:
                    res = call(MODELS[model_i], key, b64)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and model_i + 1 < len(MODELS):
                        model_i += 1
                        print("switching model to", MODELS[model_i], flush=True)
                        continue
                    if e.code in (429, 500, 503):
                        time.sleep(min(60, 2 ** attempt))
                        continue
                    raise
                except Exception:
                    time.sleep(min(60, 2 ** attempt))
            else:
                print("gave up on", row["key"], flush=True)
                continue
            rec = {"key": row["key"], "model": MODELS[model_i], "verdict": res.get("verdict"), "reason": res.get("reason")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(len(done) + 1, rec["key"], rec["verdict"], flush=True)
            done.add(row["key"])


if __name__ == "__main__":
    main()
