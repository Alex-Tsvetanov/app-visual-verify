"""Layer 'temporal': DTW alignment of two animation GIFs over 1-SSIM frame costs.

Same algorithm and constants as src/gif/gifdtw.py (which it replaces as the
source of truth), so results can be compared with the earlier numbers. Score is
the path-normalised DTW cost; higher means more different. DTW absorbs timing
differences by design, which is why the temporal score is not the whole story
for animations whose timing is part of what must match.
"""
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

SCALE = 0.5   # downscale frames for speed
TAU = 0.04    # motion threshold on |I_t - I_0| (grayscale 0..1)
MIN_WINDOW = 32  # SSIM uses 7x7 windows; tiny motion boxes are widened to this


def frames(path):
    im = Image.open(path)
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        g = im.convert("L")
        g = g.resize((int(g.width * SCALE), int(g.height * SCALE)), Image.BILINEAR)
        out.append(np.asarray(g, dtype=np.float32) / 255)
    return np.stack(out)


def motion_box(F):
    d = np.abs(F - F[0]).max(0) > TAU
    ys, xs = np.where(d)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def cost_matrix(A, B):
    C = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = 1 - ssim(A[i], B[j], data_range=1.0)
    return C


def dtw(C):
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return D[n, m], path


def compare(ref_path, cand_path):
    A, B = frames(ref_path), frames(cand_path)
    Afull, Bfull = A, B
    # Full-frame static term: catches differences outside the motion box.
    static_full = float(cost_matrix(Afull, Bfull).min(1).mean())

    boxes = [b for b in (motion_box(A), motion_box(B)) if b]
    if not boxes:
        # Neither clip moves: the only evidence is the static full-frame term.
        return {"available": True, "score": static_full,
                "details": {"static_full": static_full, "motion": False,
                            "frames": [len(A), len(B)]}}

    y0 = min(b[0] for b in boxes)
    y1 = max(b[1] for b in boxes)
    x0 = min(b[2] for b in boxes)
    x1 = max(b[3] for b in boxes)
    H, W = A.shape[1:]
    y0, y1 = max(0, min(y0, H - MIN_WINDOW)), min(H, max(y1, y0 + MIN_WINDOW))
    x0, x1 = max(0, min(x0, W - MIN_WINDOW)), min(W, max(x1, x0 + MIN_WINDOW))
    A = A[:, y0:y1, x0:x1]
    B = B[:, y0:y1, x0:x1]

    C = cost_matrix(A, B)
    total, path = dtw(C)
    n, m = C.shape
    lag = np.array([i / n - j / m for i, j in path])
    return {
        "available": True,
        "score": float(total / len(path)),
        "details": {"dtw": float(total / len(path)), "static_full": static_full,
                    "motion": True, "lag_max": float(np.abs(lag).max()),
                    "frames": [n, m], "path_len": len(path)},
    }
