"""Static features and zero-shot CNN scores for all 1,412 paired captures; resumable."""
import json
from pathlib import Path

from . import config
from .nn import ChangeNet, score_pair
from .zoo import features_from_arrays, load_pair
import torch

OUT = config.RESULTS / "full_features.jsonl"


def main():
    idx = json.loads((config.REPO_ROOT / "cache/composites/index.json").read_text())
    model = ChangeNet()
    model.load_state_dict(torch.load(config.REPO_ROOT / "cache/changenet_synthetic.pt"))
    model.eval()
    done = set()
    if OUT.exists():
        done = {json.loads(l)["key"] for l in OUT.read_text().splitlines() if l.strip()}
    with OUT.open("a") as f:
        for n, x in enumerate(idx, 1):
            if x["key"] in done:
                continue
            a, b = load_pair(x["ref"], x["cand"])
            rec = {"key": x["key"], **features_from_arrays(a, b), "cnn_zero": score_pair(model, a, b)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(n, x["key"], flush=True)


if __name__ == "__main__":
    main()
