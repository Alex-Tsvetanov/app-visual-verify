# DTW alignment of two sparse animation GIFs using 1-SSIM frame costs.
import os, sys, json, glob, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

CAP = os.environ["VERIFY_CAPTURES"]  # captures root, see src/verify/config.py
SCALE = 0.5   # downscale frames for speed
TAU = 0.04    # motion threshold on |I_t - I_0| (grayscale 0..1)

def frames(path):
    im = Image.open(path); out = []
    for i in range(im.n_frames):
        im.seek(i)
        g = im.convert("L")
        g = g.resize((int(g.width * SCALE), int(g.height * SCALE)), Image.BILINEAR)
        out.append(np.asarray(g, dtype=np.float32) / 255)
    return np.stack(out)

def motion_box(F):
    d = np.abs(F - F[0]).max(0) > TAU
    ys, xs = np.where(d)
    if len(ys) == 0: return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

def cost_matrix(A, B):
    C = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = 1 - ssim(A[i], B[j], data_range=1.0)
    return C

def dtw(C):
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    # backtrack path
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if k == 0: i, j = i - 1, j - 1
        elif k == 1: i -= 1
        else: j -= 1
    path.reverse()
    return D[n, m], path

def analyse(maui_path, cpp_path):
    A, B = frames(maui_path), frames(cpp_path)
    # crop both to the union of their motion boxes so static chrome does not dilute the cost
    boxes = [b for b in (motion_box(A), motion_box(B)) if b]
    if not boxes: return None
    y0 = min(b[0] for b in boxes); y1 = max(b[1] for b in boxes)
    x0 = min(b[2] for b in boxes); x1 = max(b[3] for b in boxes)
    # enforce a minimum window so SSIM (7x7 windows) is defined for tiny motion boxes
    H, W = A.shape[1:]
    y0, y1 = max(0, min(y0, H - 32)), min(H, max(y1, y0 + 32))
    x0, x1 = max(0, min(x0, W - 32)), min(W, max(x1, x0 + 32))
    Afull, Bfull = A, B
    A = A[:, y0:y1, x0:x1]; B = B[:, y0:y1, x0:x1]
    C = cost_matrix(A, B)
    # full-frame static term: catches differences outside the motion box (e.g. a filled field)
    Cfull = cost_matrix(Afull, Bfull)
    static_full = float(Cfull.min(1).mean())
    total, path = dtw(C)
    norm = total / len(path)
    n, m = C.shape
    lag = np.array([i / n - j / m for i, j in path])     # diagonal deviation along the path
    lockstep = np.mean([C[i, i] for i in range(min(n, m))]) if n == m else np.nan
    best_static = C.min(1).mean()                          # no ordering constraint
    return dict(static_full=static_full, dtw=float(norm), lockstep=float(lockstep), best_static=float(best_static),
                lag_mean=float(lag.mean()), lag_max=float(np.abs(lag).max()),
                n=n, m=m, n_path=len(path))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "labeled":
        L = json.load(open(sys.argv[2]))
        rows = []
        for k, v in L.items():
            page, plat, theme = k.split("/")
            a = f"{CAP}/{plat}/maui/{page}_{theme}.gif"; b = f"{CAP}/{plat}/cpp/{page}_{theme}.gif"
            if os.path.exists(a) and os.path.exists(b):
                r = analyse(a, b)
                if r: rows.append(dict(id=k, label=v["label"], **r)); print(k, v["label"], {x: round(y, 3) if isinstance(y, float) else y for x, y in r.items()}, flush=True)
        json.dump(rows, open(sys.argv[3], "w"), indent=1)
    elif mode == "all":
        rows = []
        for a in sorted(glob.glob(f"{CAP}/*/maui/*.gif")):
            plat = a.split("/")[-3]; name = os.path.basename(a)
            b = os.path.join(os.path.dirname(os.path.dirname(a)), "cpp", name)
            if not os.path.exists(b): continue
            try: r = analyse(a, b)
            except Exception as e: print("ERR", a, e, flush=True); continue
            if r: rows.append(dict(id=f"{plat}/{name[:-4]}", **r)); print(plat, name, round(r["dtw"], 3), flush=True)
        json.dump(rows, open(sys.argv[2], "w"), indent=1)
