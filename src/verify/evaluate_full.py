"""Evaluate every layer on the full 1,412-pair set, with the consensus and human subsets reported separately.

Labels: data/labels/full_v1.json. A pair is DIFFERENT when its label is DECLINED.
  - human: 204 pairs where Gemini and Claude disagreed and a human chose the verdict.
  - consensus: 1,208 pairs where Gemini and Claude agreed; their shared verdict is used, not human-checked.
Unsupervised layers are scored directly. The stacker uses stratified 5-fold cross-validation.
AUC CIs: 2,000-resample percentile bootstrap.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config
from .evaluate import accuracy, auc_ci, fit_threshold
from .study import CLASSICAL, NAMES

LAYERS = CLASSICAL + ["cnn_zero"]


def load():
    labels = json.loads((config.REPO_ROOT / "data/labels/full_v1.json").read_text())
    feats = {}
    for line in (config.RESULTS / "full_features.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            feats[r["key"]] = r
    keys = [k for k in labels if k in feats]
    y = np.array([labels[k]["verdict"] == "DECLINED" for k in keys]).astype(int)
    src = np.array([labels[k]["source"] for k in keys])
    X = {name: np.array([feats[k][name] for k in keys]) for name in LAYERS}
    return keys, y, src, X


def block(y, s, mask):
    return {"all": auc_ci(y[mask], s[mask]) if mask.all() else auc_ci(y, s)}


def subset_auc(y, s, src):
    out = {}
    out["full"] = auc_ci(y, s)
    out["consensus"] = auc_ci(y[src == "consensus"], s[src == "consensus"])
    out["human"] = auc_ci(y[src == "human"], s[src == "human"])
    return out


def stack_oof(X, y, folds=5, seed=0):
    Xs = np.column_stack([X[n] for n in LAYERS])
    oof = np.zeros(len(y))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, y):
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xs[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
    return oof


def figure_roc(y, src, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    panels = [("всички 1412 двойки", np.ones(len(y), bool)), ("човешки проверени 204", src == "human")]
    for ax, (title, mask) in zip(axes, panels):
        for label, s in curves.items():
            fpr, tpr, _ = roc_curve(y[mask], s[mask])
            ax.plot(fpr, tpr, label=f"{label} ({roc_auc_score(y[mask], s[mask]):.2f})", lw=1.2)
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Фалшиво положителни (FPR)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("Истински положителни (TPR)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    keys, y, src, X = load()
    res = {"n": len(y), "n_declined": int(y.sum()), "n_human": int((src == "human").sum()),
           "n_consensus": int((src == "consensus").sum()), "layers": {}}
    for name in LAYERS:
        s = X[name]
        thr = fit_threshold(y, s)
        res["layers"][name] = {**subset_auc(y, s, src),
                               "threshold": float(thr), "in_sample": accuracy(y, s, thr)}
    oof = stack_oof(X, y)
    res["layers"]["stack"] = {**subset_auc(y, oof, src), "threshold": float(fit_threshold(y, oof)),
                              "in_sample": accuracy(y, oof, fit_threshold(y, oof))}
    (config.RESULTS / "full_study.json").write_text(json.dumps(res, indent=2) + "\n")

    rows = [r"\begin{tabular}{lccc}", r"\hline",
            r"Метод & AUC (1412) & AUC (консенсус, 1208) & AUC (човешки, 204) \\", r"\hline"]
    for name in LAYERS + ["stack"]:
        d = res["layers"][name]
        label = NAMES.get(name, "Логистичен стекинг (5-кратно CV)")
        cells = " & ".join(f"{d[k]['auc']:.3f} [{d[k]['ci'][0]:.2f}, {d[k]['ci'][1]:.2f}]"
                           for k in ["full", "consensus", "human"])
        rows.append(f"{label} & {cells} \\\\")
    rows += [r"\hline", r"\end{tabular}"]
    (config.REPO_ROOT / "paper/tables/full_static.tex").write_text("\n".join(rows) + "\n")

    curves = {"SSIM плочки": X["pixel_ssim_tile"], "LBP": X["lbp_chi2"],
              "ResNet-18 топ-5": X["resnet_tile_top5"], "DINOv2 топ-5": X["dino_tile_top5"],
              "CNN синтетичен": X["cnn_zero"], "Стекинг (CV)": oof}
    figure_roc(y, src, curves, config.REPO_ROOT / "paper/figures/roc_static.pdf")
    print(json.dumps({k: round(v["full"]["auc"], 3) for k, v in res["layers"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
