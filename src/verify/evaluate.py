"""Evaluate the layers against the labels. Writes data/results/ and config/thresholds.json.

Static: 108 human labels (MINOR counts as SAME), layer 'pixel'.
  - AUC over all 108 (LOO: the score is not fitted, so LOO AUC is the AUC).
  - Held-out: orig76 is the training split, new32 the test split. The verdict
    threshold is fitted on orig76 and scored on new32.
  - Threshold LOO: refit the threshold on 107 labels and classify the one left out.
GIF: 34 ground-truth pairs (MINOR counts as SAME), layer 'temporal'. Same procedure.
Bootstrap: 2000 resamples, percentile 95% CI on AUC.

The verdict threshold is the cut that maximises balanced accuracy. The MINOR
band is [0.5 * diff, diff): there is no label data to fit it, so it is a guess.

Reported-only numbers (Siamese head, Grok v2 hand features) are copied from
their result files. Their features are not in this repo, so they are not reproduced here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from . import SCHEMA_VERSION, config
from .layers import pixel, temporal

N_BOOT = 2000
SEED = 0
MINOR_FRACTION = 0.5

REPORTED = {
    "static_siamese_head_LOO108": {"auc": 0.695, "ci": [0.62, 0.78],
                                   "source": "data/results/train_log.txt E2 LOO 108 (ResNet-18 features, not in repo)"},
    "static_siamese_head_LOO108_rn": {"auc": 0.664, "ci": [0.59, 0.74],
                                      "source": "data/results/train2_rn.txt (features not in repo)"},
    "static_grok_v2_LOO108": {"auc": 0.909, "ci": [0.85, 0.96],
                              "source": "data/labels/port/cpp/tools/parity/ml/data/v3_results.txt LOO all v2 (features not in repo)"},
    "gif_dtw_AUC34": {"auc": 0.558, "ci": None,
                      "source": "src/gif/all_dtw.json reproduced here by temporal layer"},
}


def auc_ci(y, s, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    boots = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if len(np.unique(y[b])) == 2:
            boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(roc_auc_score(y, s)), "ci": [float(lo), float(hi)],
            "n": int(len(y)), "pos": int(y.sum()), "neg": int(len(y) - y.sum())}


def fit_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    finite = np.isfinite(thr)
    bal = (tpr + (1 - fpr)) / 2
    return float(thr[finite][np.argmax(bal[finite])])


def accuracy(y, s, t):
    pred = s >= t
    tp = np.sum(pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    return {"acc": float((pred == y).mean()),
            "bal_acc": float((tp / max(1, y.sum()) + tn / max(1, (1 - y).sum())) / 2)}


def threshold_loo(y, s):
    preds = np.zeros(len(y), bool)
    for i in range(len(y)):
        mask = np.ones(len(y), bool)
        mask[i] = False
        t = fit_threshold(y[mask], s[mask])
        preds[i] = s[i] >= t
    return {"acc": float((preds == y).mean()),
            "bal_acc": float((preds[y == 1].mean() + (~preds[y == 0]).mean()) / 2)}


def evaluate_split(y, s, held_mask):
    """Train/test on a fixed split, plus LOO over all labels."""
    t_train = fit_threshold(y[~held_mask], s[~held_mask])
    return {
        "train_threshold": t_train,
        "heldout_auc": auc_ci(y[held_mask], s[held_mask]),
        "heldout_threshold_acc": accuracy(y[held_mask], s[held_mask], t_train),
        "train_auc": auc_ci(y[~held_mask], s[~held_mask]),
    }


def report_block(y, s, held_mask=None):
    out = {"all": auc_ci(y, s), "threshold_loo": threshold_loo(y, s),
           "threshold_all_in_sample": fit_threshold(y, s)}
    out["threshold_all_in_sample_acc"] = accuracy(y, s, out["threshold_all_in_sample"])
    if held_mask is not None:
        out["split"] = evaluate_split(y, s, held_mask)
    return out


def load_static(root):
    labels = json.loads(Path(config.DEFAULT_LABELS).read_text())
    orig = set(json.loads(Path(config.DEFAULT_ORIG76).read_text()))
    rows = []
    for key, v in labels.items():
        page, platform, theme = key.split("/")
        ref, cand = config.pair_paths(root, platform, f"{page}_{theme}", "png")
        if not (ref.exists() and cand.exists()):
            continue
        r = pixel.compare(ref, cand)
        rows.append({"id": key, "y": int(v["label"] == "DIFFERENT"), "in_orig76": key in orig,
                     "score": r["score"], "global": r["details"]["global_dissimilarity"],
                     "size_mismatch": r["details"]["size_mismatch"]})
    return rows


def load_gif(root):
    truth = json.loads(Path(config.DEFAULT_TRUTH).read_text())
    prior = {r["id"]: r for r in json.loads(Path(config.DEFAULT_GIF_DTW).read_text())}
    rows, max_repro_diff = [], 0.0
    for key, label in truth.items():
        platform, name = key.split("/")
        ref, cand = config.pair_paths(root, platform, name, "gif")
        if not (ref.exists() and cand.exists()):
            continue
        r = temporal.compare(ref, cand)
        if key in prior:
            max_repro_diff = max(max_repro_diff, abs(r["score"] - prior[key]["dtw"]))
        rows.append({"id": key, "y": int(label == "DIFFERENT"), "score": r["score"],
                     "static_full": r["details"]["static_full"]})
    return rows, max_repro_diff


def main(argv=None):
    p = argparse.ArgumentParser(prog="verify.evaluate")
    p.add_argument("--captures", help="captures root (or set VERIFY_CAPTURES)")
    p.add_argument("--results", type=Path, default=config.RESULTS)
    args = p.parse_args(argv)
    root = config.captures_root(args.captures)
    args.results.mkdir(parents=True, exist_ok=True)

    # Static layer
    srows = load_static(root)
    y = np.array([r["y"] for r in srows])
    s = np.array([r["score"] for r in srows])
    held = np.array([not r["in_orig76"] for r in srows])
    static = {"n_labels": len(srows), "n_orig76": int((~held).sum()), "n_new32": int(held.sum()),
              "size_mismatch_pairs": [r["id"] for r in srows if r["size_mismatch"]],
              "pixel_max_tile": report_block(y, s, held),
              "pixel_global_dissimilarity": report_block(y, np.array([r["global"] for r in srows]), held)}

    # GIF layer
    grows, repro = load_gif(root)
    gy = np.array([r["y"] for r in grows])
    gs = np.array([r["score"] for r in grows])
    gif = {"n_labels": len(grows), "n_pos": int(gy.sum()),
           "max_abs_diff_vs_all_dtw_json": repro,
           "temporal_dtw": report_block(gy, gs),
           "temporal_static_full": report_block(gy, np.array([r["static_full"] for r in grows]))}

    # Thresholds fitted on all labels (in-sample), written for the CLI.
    diff_static = static["pixel_max_tile"]["threshold_all_in_sample"]
    diff_gif = gif["temporal_dtw"]["threshold_all_in_sample"]
    thresholds = {
        "schema_version": SCHEMA_VERSION,
        "note": "fitted on all labels (in-sample); MINOR band width is a guess",
        "static": {"score": "pixel.score", "diff": diff_static, "minor": MINOR_FRACTION * diff_static},
        "gif": {"score": "temporal.score", "diff": diff_gif, "minor": MINOR_FRACTION * diff_gif},
    }
    config.THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    config.THRESHOLDS.write_text(json.dumps(thresholds, indent=2) + "\n")

    out = {"schema_version": SCHEMA_VERSION, "static": static, "gif": gif,
           "reported_not_reproduced": {k: v for k, v in REPORTED.items() if k != "gif_dtw_AUC34"}}
    (args.results / "eval.json").write_text(json.dumps(out, indent=2) + "\n")
    (args.results / "static_pairs.json").write_text(json.dumps(srows, indent=1) + "\n")
    (args.results / "gif_pairs.json").write_text(json.dumps(grows, indent=1) + "\n")
    print(json.dumps({"static_pixel_auc": static["pixel_max_tile"]["all"],
                      "gif_dtw_auc": gif["temporal_dtw"]["all"],
                      "gif_max_repro_diff": repro}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
