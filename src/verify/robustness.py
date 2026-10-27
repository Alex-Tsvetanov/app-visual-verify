"""Shift robustness: shift the candidate screenshot right by 0-4 px (replicated border) and
recompute every static layer, so the false-alarm and detection rates can be read per shift."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config
from .zoo import features_from_arrays, load_pair

SHIFTS = [0, 1, 2, 3, 4]


def shifted(img, dx):
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def main(root_arg=None):
    root = config.captures_root(root_arg)
    labels = json.loads(Path(config.DEFAULT_LABELS).read_text())
    orig = set(json.loads(Path(config.DEFAULT_ORIG76).read_text()))
    rows = []
    for key, v in labels.items():
        page, platform, theme = key.split("/")
        ref, cand = config.pair_paths(root, platform, f"{page}_{theme}", "png")
        a, b = load_pair(ref, cand)
        for dx in SHIFTS:
            feats = features_from_arrays(a, shifted(b, dx))
            rows.append({"id": key, "y": int(v["label"] == "DIFFERENT"), "orig76": key in orig,
                         "shift": dx, **feats})
        print(len(rows) // len(SHIFTS), key, flush=True)
    (config.RESULTS / "shift_features.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)


def analyse():
    """False-alarm and detection rates per shift, with each layer's threshold fitted at shift 0."""
    from sklearn.metrics import roc_auc_score
    from .evaluate import fit_threshold
    from .study import CLASSICAL, NAMES

    rows = json.loads((config.RESULTS / "shift_features.json").read_text())
    base = [r for r in rows if r["shift"] == 0]
    y0 = np.array([r["y"] for r in base])
    out = {}
    lines = [r"\begin{tabular}{l" + "cc" * len(SHIFTS) + "}", r"\hline",
             "Метод & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d} px}}" for d in SHIFTS) + r" \\",
             " & " + " & ".join("ЛА & ОД" for _ in SHIFTS) + r" \\", r"\hline"]
    for k in CLASSICAL:
        s0 = np.array([r[k] for r in base])
        t = fit_threshold(y0, s0)
        res = {}
        for d in SHIFTS:
            rs = [r for r in rows if r["shift"] == d]
            y = np.array([r["y"] for r in rs])
            s = np.array([r[k] for r in rs])
            fa = float((s[y == 0] >= t).mean())   # SAME pairs flagged (lower is better)
            det = float((s[y == 1] >= t).mean())  # DIFFERENT pairs flagged (higher is better)
            res[d] = {"false_alarm": fa, "detect": det, "auc": float(roc_auc_score(y, s))}
        out[k] = {"threshold": float(t), "by_shift": res}
        cells = " & ".join(f"{res[d]['false_alarm']:.2f} & {res[d]['detect']:.2f}" for d in SHIFTS)
        lines.append(f"{NAMES[k]} & {cells} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (config.REPO_ROOT / "paper/tables/shift.tex").write_text("\n".join(lines) + "\n")
    (config.RESULTS / "shift_study.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "analyse":
    analyse()
