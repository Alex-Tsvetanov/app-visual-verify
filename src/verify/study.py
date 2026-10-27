"""Evaluate every layer against the labels; write data/results/study.json,
paper/tables/*.tex and paper/figures/*.pdf.

Protocols (all AUC values with 2000-resample percentile 95% CIs):
  all108    AUC over the 108 static labels (MINOR counts as SAME). Training-free
            and zero-shot scores are not fitted, so this is their leave-one-out AUC.
  heldout32 AUC on the 32 new labels; orig76 is the training split for fitted parts.
  loo_bal   leave-one-out balanced accuracy of a threshold fitted on the other labels.
  stack     logistic regression over the layer scores: leave-one-out (pooled AUC) and
            orig76 -> new32.
  cnn_cv    fine-tuned CNN, 4-fold stratified cross-validation (out-of-fold AUC).
GIF pairs: 34 truth pairs with 4 DIFFERENT. Their CIs are very wide and are reported as such.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config
from .evaluate import accuracy, auc_ci, fit_threshold, threshold_loo
from .layers import pixel, temporal
from .nn import ChangeNet, finetune_mil, score_pair, train_synthetic, window_scores, to_input
from .zoo import load_pair

RES = config.RESULTS
PAPER = config.REPO_ROOT / "paper"
CLASSICAL = ["pixel_ssim_tile", "gmsd", "chamfer_edges", "colour_hellinger", "lbp_chi2",
             "orb_homography", "segment_unmatched", "resnet_tile_max", "resnet_tile_top5",
             "dino_tile_max", "dino_tile_top5"]
NAMES = {
    "pixel_ssim_tile": "SSIM по плочки (базова)", "gmsd": "GMSD", "chamfer_edges": "Chamfer по контури",
    "colour_hellinger": "Цветови хистограми", "lbp_chi2": "LBP текстура", "orb_homography": "ORB + RANSAC",
    "segment_unmatched": "Сегменти (k-means)", "resnet_tile_max": "ResNet-18 плочки (max)",
    "resnet_tile_top5": "ResNet-18 плочки (топ-5)", "dino_tile_max": "DINOv2 плочки (max)",
    "dino_tile_top5": "DINOv2 плочки (топ-5)", "cnn_zero": "CNN, обучен синтетично",
    "cnn_cv": "CNN, дообучен (4-кратно CV)", "stack_loo": "Логистичен стекинг (LOO)",
}


GIF_NAMES = {"dtw": "DTW по SSIM", "pixel_frame0": "SSIM на първия кадър",
             "set_chamfer": "хамфър по кадри", "cnn_set": "CNN по кадри (насищане)"}


def fmt_ci(d):
    return f"{d['auc']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def heldout_auc(y, s, held):
    return auc_ci(y[held], s[held])


def evaluate_score(y, s, held):
    out = {"all": auc_ci(y, s), "heldout": heldout_auc(y, s, held)}
    thr_loo = threshold_loo(y, s)
    out["loo_bal_acc"] = thr_loo["bal_acc"]
    out["loo_acc"] = thr_loo["acc"]
    t_train = fit_threshold(y[~held], s[~held])
    out["train_threshold_acc_heldout"] = accuracy(y[held], s[held], t_train)["bal_acc"]
    return out


# ---- CNN scores ---------------------------------------------------------

def static_pairs(root):
    rows = json.loads((RES / "static_features.json").read_text())
    pairs = []
    for r in rows:
        page, platform, theme = r["id"].split("/")
        ref, cand = config.pair_paths(root, platform, f"{page}_{theme}", "png")
        pairs.append(load_pair(ref, cand))
    return rows, pairs


def cnn_zero(model, pairs):
    return np.array([score_pair(model, a, b) for a, b in pairs])


def cnn_cv(pairs, y, base, folds=4, seed=0):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        m = finetune_mil(base, [pairs[i] for i in tr], y[tr], epochs=2, seed=seed + k)
        for i in te:
            oof[i] = score_pair(m, *pairs[i])
        print(f"cnn_cv fold {k} done", flush=True)
    return oof


def prob_map(model, a, b):
    """Full-resolution change probability (window average), used for the figure."""
    x = to_input(a, b)
    h, w = x.shape[2:]
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    import torch
    from .nn import window_origins, WIN
    model.eval()
    origins = window_origins(h, w)
    with torch.no_grad():
        for y0, x0 in origins:
            p = torch.sigmoid(model(x[:, :, y0:y0 + WIN, x0:x0 + WIN]))[0].numpy()
            acc[y0:y0 + WIN, x0:x0 + WIN] += p
            cnt[y0:y0 + WIN, x0:x0 + WIN] += 1
    return acc / np.maximum(cnt, 1)


# ---- GIF layers ---------------------------------------------------------

def gif_frames_rgb(path, n=6):
    with Image.open(path) as im:
        total = im.n_frames
        idx = np.linspace(0, total - 1, min(n, total)).round().astype(int)
        out = []
        for i in idx:
            im.seek(int(i))
            out.append(np.asarray(im.convert("RGB")))
    return out


def gif_set_chamfer(ref, cand):
    """Order-free SSIM set distance between the motion crops of two GIFs (cost matrix as in temporal)."""
    A, B = temporal.frames(ref), temporal.frames(cand)
    boxes = [b for b in (temporal.motion_box(A), temporal.motion_box(B)) if b]
    if boxes:
        y0 = min(b[0] for b in boxes)
        y1 = max(b[1] for b in boxes)
        x0 = min(b[2] for b in boxes)
        x1 = max(b[3] for b in boxes)
        H, W = A.shape[1:]
        y0, y1 = max(0, min(y0, H - 32)), min(H, max(y1, y0 + 32))
        x0, x1 = max(0, min(x0, W - 32)), min(W, max(x1, x0 + 32))
        A, B = A[:, y0:y1, x0:x1], B[:, y0:y1, x0:x1]
    C = temporal.cost_matrix(A, B)
    return float(0.5 * (C.min(1).mean() + C.min(0).mean()))


def gif_set_cnn(model, ref, cand):
    fa, fb = gif_frames_rgb(ref), gif_frames_rgb(cand)
    S = np.array([[score_pair(model, a, b) for b in fb] for a in fa])
    return float(max(S.min(1).max(), S.min(0).max()))


def gif_stage(root, model):
    cache = RES / "gif_features.json"
    if cache.exists():
        return json.loads(cache.read_text())
    truth = json.loads(config.DEFAULT_TRUTH.read_text())
    rows = []
    for key, label in truth.items():
        platform, name = key.split("/")
        ref, cand = config.pair_paths(root, platform, name, "gif")
        rows.append({
            "id": key, "y": int(label == "DIFFERENT"),
            "dtw": temporal.compare(ref, cand)["score"],
            "pixel_frame0": pixel.compare(ref, cand)["score"],
            "set_chamfer": gif_set_chamfer(ref, cand),
            "cnn_set": gif_set_cnn(model, ref, cand),
        })
        print("gif", key, rows[-1], flush=True)
    cache.write_text(json.dumps(rows, indent=1) + "\n")
    return rows


# ---- figures ------------------------------------------------------------

def figure_roc(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for label, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{label} ({roc_auc_score(y, s):.2f})", lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("Фалшиво положителни (FPR)")
    ax.set_ylabel("Истински положителни (TPR)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def figure_heatmap(model, pairs, rows, path, picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 2.6 * len(picks)))
    for r, i in enumerate(picks):
        a, b = pairs[i]
        pm = prob_map(model, a, b)
        for c, (img, title) in enumerate([(a, "референция"), (b, "кандидат"), (pm, "карта на промяната")]):
            ax = axes[r, c]
            if c < 2:
                ax.imshow(img)
            else:
                ax.imshow(pm, cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{title}: {rows[i]['id']}" if c == 0 else title, fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---- main ---------------------------------------------------------------

def main(root_arg=None):
    root = config.captures_root(root_arg)
    rows, pairs = static_pairs(root)
    y = np.array([r["y"] for r in rows])
    held = np.array([not r["orig76"] for r in rows])
    feats = {k: np.array([r[k] for r in rows]) for k in CLASSICAL}

    model = train_synthetic(root)

    zero = RES / "cnn_zero.json"
    s_zero = np.array(json.loads(zero.read_text())) if zero.exists() else cnn_zero(model, pairs)
    zero.write_text(json.dumps(s_zero.tolist()) + "\n")
    feats["cnn_zero"] = s_zero

    cv = RES / "cnn_cv.json"
    if cv.exists():
        s_cv = np.array(json.loads(cv.read_text()))
    else:
        s_cv = cnn_cv(pairs, y, model)
        cv.write_text(json.dumps(s_cv.tolist()) + "\n")
    feats["cnn_cv"] = s_cv

    single = {k: evaluate_score(y, feats[k], held) for k in feats}

    # stacking
    X = np.column_stack([feats[k] for k in CLASSICAL + ["cnn_zero"]])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    loo = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Xs[m], y[m])
        loo[i] = clf.predict_proba(Xs[i:i + 1])[0, 1]
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Xs[~held], y[~held])
    ho = clf.predict_proba(Xs[held])[:, 1]
    single["stack_loo"] = {"all": auc_ci(y, loo), "heldout": auc_ci(y[held], ho),
                           "loo_bal_acc": accuracy(y, loo, 0.5)["bal_acc"]}

    # GIF
    gif = gif_stage(root, model)
    gy = np.array([r["y"] for r in gif])
    gif_res = {}
    for k in ["dtw", "pixel_frame0", "set_chamfer", "cnn_set"]:
        s = np.array([r[k] for r in gif])
        gif_res[k] = {"all": auc_ci(gy, s), "loo_bal_acc": threshold_loo(gy, s)["bal_acc"]}

    study = {"n_static": int(len(y)), "n_pos": int(y.sum()), "n_heldout": int(held.sum()),
             "static": single, "gif": {"n": len(gy), "n_pos": int(gy.sum()), **gif_res}}
    (RES / "study.json").write_text(json.dumps(study, indent=2, ensure_ascii=False) + "\n")

    # tables
    tdir = PAPER / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lccc}", r"\hline",
             r"Метод & AUC (108) & AUC (32, нов) & LOO точност \\", r"\hline"]
    for k in CLASSICAL + ["cnn_zero", "cnn_cv", "stack_loo"]:
        d = single[k]
        lines.append(f"{NAMES[k]} & {d['all']['auc']:.3f} [{d['all']['ci'][0]:.2f}, {d['all']['ci'][1]:.2f}] & "
                     f"{d['heldout']['auc']:.3f} [{d['heldout']['ci'][0]:.2f}, {d['heldout']['ci'][1]:.2f}] & "
                     f"{d['loo_bal_acc']:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (tdir / "static.tex").write_text("\n".join(lines) + "\n")

    gl = [r"\begin{tabular}{lccc}", r"\hline", r"Слой & AUC (34) & 95\% ДИ & LOO точност \\", r"\hline"]
    for k, v in gif_res.items():
        gl.append(f"{GIF_NAMES[k]} & {v['all']['auc']:.3f} & [{v['all']['ci'][0]:.2f}, {v['all']['ci'][1]:.2f}] & "
                  f"{v['loo_bal_acc']:.3f} \\\\")
    gl += [r"\hline", r"\end{tabular}"]
    (tdir / "gif.tex").write_text("\n".join(gl) + "\n")

    # figures
    curves = {"SSIM плочки": (y, feats["pixel_ssim_tile"]), "LBP": (y, feats["lbp_chi2"]),
              "ResNet-18 топ-5": (y, feats["resnet_tile_top5"]), "CNN синтетичен": (y, s_zero),
              "CNN дообучен (CV)": (y, s_cv), "Стекинг (LOO)": (y, loo)}
    figure_roc(curves, PAPER / "figures" / "roc_static.pdf")
    order = np.argsort(-s_cv)
    picks = [int(order[0]), int(order[-1])]
    neg_idx = [i for i in order if y[i] == 0][:1]
    picks = [int(order[0]), int(neg_idx[0])] if neg_idx else picks
    figure_heatmap(model, pairs, rows, PAPER / "figures" / "heatmap.pdf", picks)

    print(json.dumps({k: (v["all"]["auc"] if "all" in v else v) for k, v in single.items()}, indent=1, ensure_ascii=False))
    print(json.dumps(gif_res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
