"""Compute every static-pair feature for the 108 human-labelled pairs and cache them."""
import json
import sys

from . import config
from .zoo import static_features
from pathlib import Path


def main(root_arg=None):
    root = config.captures_root(root_arg)
    labels = json.loads(Path(config.DEFAULT_LABELS).read_text())
    orig = set(json.loads(Path(config.DEFAULT_ORIG76).read_text()))
    rows = []
    for key, v in labels.items():
        page, platform, theme = key.split("/")
        ref, cand = config.pair_paths(root, platform, f"{page}_{theme}", "png")
        feats = static_features(ref, cand)
        rows.append({"id": key, "y": int(v["label"] == "DIFFERENT"), "orig76": key in orig, **feats})
        print(len(rows), key, flush=True)
    (config.RESULTS / "static_features.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
