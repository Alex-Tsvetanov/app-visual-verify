"""Figure for the paper: GIF pairs with sampled frames, the SSIM cost matrix and the DTW path."""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from . import config
from .layers import temporal
from .study import gif_frames_rgb


def cost_and_path(ref, cand):
    """Same cropping and cost as temporal.compare, but returns the matrix and the path."""
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
    total, path = temporal.dtw(C)
    return C, path, float(total / len(path))


def strip(ax_row, frames, label):
    for k, f in enumerate(frames):
        ax = ax_row[k]
        ax.imshow(f)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label} {k + 1}", fontsize=6)


def frame_at(path, idx):
    with Image.open(path) as im:
        im.seek(int(idx))
        return np.asarray(im.convert("RGB"))


def build(root, pairs, path, steps=6):
    """One row pair per GIF: reference and candidate frames at evenly spaced DTW path steps."""
    n = len(pairs)
    fig = plt.figure(figsize=(7.4, 3.0 * n))
    outer = fig.add_gridspec(n, 1, hspace=0.9)
    for r, (key, verdict_label) in enumerate(pairs):
        platform, name = key.split("/")
        ref, cand = config.pair_paths(root, platform, name, "gif")
        C, dtw_path, score = cost_and_path(ref, cand)
        pick = np.linspace(0, len(dtw_path) - 1, steps).round().astype(int)
        inner = outer[r].subgridspec(2, steps + 1, width_ratios=[1] * steps + [1.6], hspace=0.45, wspace=0.35)
        for k, s in enumerate(pick):
            i, j = dtw_path[s]
            # the path indexes the motion-cropped, half-scale frames; map to the full frame index
            fi = int(round(i * (Image.open(ref).n_frames - 1) / max(1, C.shape[0] - 1)))
            fj = int(round(j * (Image.open(cand).n_frames - 1) / max(1, C.shape[1] - 1)))
            ax = fig.add_subplot(inner[0, k])
            ax.imshow(frame_at(ref, fi))
            ax.axis("off")
            ax.set_title(f"еталон {fi}", fontsize=6)
            ax = fig.add_subplot(inner[1, k])
            ax.imshow(frame_at(cand, fj))
            ax.axis("off")
            ax.set_title(f"порт {fj}", fontsize=6)
        ax = fig.add_subplot(inner[:, steps])
        ax.imshow(C, cmap="magma", origin="upper")
        pi, pj = zip(*dtw_path)
        ax.plot(pj, pi, color="cyan", lw=1.2)
        ax.set_xlabel("кадри на порта", fontsize=6)
        ax.set_ylabel("кадри на еталона", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_title(f"DTW = {score:.3f}\n{verdict_label}", fontsize=6, pad=6)
        box = outer[r].get_position(fig)
        fig.text(box.x0, box.y1 + 0.03, key, fontsize=7, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = config.captures_root(sys.argv[1] if len(sys.argv) > 1 else None)
    picks = [
        ("android/swipe_refresh_dark", "DIFFERENT: вярно"),
        ("android/selection_synchronization_dark", "SAME: грешен положителен"),
    ]
    build(root, picks, config.REPO_ROOT / "paper/figures/gif_example.pdf")
    print("ok")
