"""Static-pair layers beyond the primary pixel layer: classical image-processing
measures (filtering, edges, colour, texture, keypoints, segmentation) and
frozen pretrained CNN/ViT tile features. Every score is higher = more different.
"""
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from skimage.feature import local_binary_pattern

from . import config
from .layers import pixel

GRID = 16


def _load(path, frame=0):
    with Image.open(path) as im:
        im.seek(frame)
        return np.asarray(im.convert("RGB"))


def load_pair(ref_path, cand_path, frame=0):
    """RGB uint8 arrays with the candidate resized to the reference size if needed."""
    a = _load(ref_path, frame)
    b = _load(cand_path, frame)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return a, b


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _tile_slices(h, w, grid=GRID):
    rows = np.array_split(np.arange(h), min(grid, h))
    cols = np.array_split(np.arange(w), min(grid, w))
    return [(r[0], r[-1] + 1, c[0], c[-1] + 1) for r in rows for c in cols]


# --- filtering: gradient magnitude similarity deviation (Xue et al., 2014)
def gmsd(ga, gb):
    ga = ga.astype(np.float32)
    gb = gb.astype(np.float32)
    kx = np.array([[1, 0, -1]] * 3, np.float32) / 3
    ky = kx.T

    def grad(x):
        return np.sqrt(cv2.filter2D(x, -1, kx) ** 2 + cv2.filter2D(x, -1, ky) ** 2)

    ma, mb = grad(ga), grad(gb)
    c = 170.0
    gms = (2 * ma * mb + c) / (ma ** 2 + mb ** 2 + c)
    return float(gms.std())


# --- edges: symmetric chamfer distance between Canny edge maps (Borgefors, 1988)
def chamfer(ga, gb):
    ea = cv2.Canny(ga, 50, 150) > 0
    eb = cv2.Canny(gb, 50, 150) > 0
    if not ea.any() and not eb.any():
        return 0.0
    if not ea.any() or not eb.any():
        return 1.0
    da = cv2.distanceTransform(np.where(ea, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform(np.where(eb, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    d = 0.5 * (da[eb].mean() + db[ea].mean())
    return float(d / np.hypot(*ga.shape))


# --- colour: per-tile RGB histogram, Hellinger distance, worst tile
def colour_hellinger(a, b, bins=4):
    step = 256 // bins
    qa = (a[..., 0] // step) * bins * bins + (a[..., 1] // step) * bins + a[..., 2] // step
    qb = (b[..., 0] // step) * bins * bins + (b[..., 1] // step) * bins + b[..., 2] // step
    worst = 0.0
    for y0, y1, x0, x1 in _tile_slices(*a.shape[:2]):
        ha = np.bincount(qa[y0:y1, x0:x1].ravel(), minlength=bins ** 3) / ((y1 - y0) * (x1 - x0))
        hb = np.bincount(qb[y0:y1, x0:x1].ravel(), minlength=bins ** 3) / ((y1 - y0) * (x1 - x0))
        worst = max(worst, np.sqrt(max(0.0, 1 - np.sqrt(ha * hb).sum())))
    return float(worst)


# --- texture: uniform LBP (P=8, R=1) histograms per tile, chi-square, worst tile (Ojala et al., 2002)
def lbp_chi2(ga, gb):
    la = local_binary_pattern(ga, 8, 1, "uniform").astype(int)
    lb = local_binary_pattern(gb, 8, 1, "uniform").astype(int)
    worst = 0.0
    for y0, y1, x0, x1 in _tile_slices(*ga.shape):
        ha = np.bincount(la[y0:y1, x0:x1].ravel(), minlength=10).astype(float)
        hb = np.bincount(lb[y0:y1, x0:x1].ravel(), minlength=10).astype(float)
        ha /= ha.sum()
        hb /= hb.sum()
        worst = max(worst, 0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-9)))
    return float(worst)


# --- model fitting: ORB keypoints, ratio test, RANSAC homography (Rublee et al., 2011; Fischler & Bolles, 1981)
def orb_homography(ga, gb):
    orb = cv2.ORB_create(nfeatures=1500)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 1.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 1.0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if mask is None:
        return 1.0
    inliers = float(mask.sum())
    return float(np.clip(1 - inliers / min(len(ka), len(kb)), 0, 1))


# --- segmentation: joint k-means colour regions, unmatched component area (Gonzalez & Woods, ch. 10)
def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def segment_unmatched(a, b, k=6, max_side=400, min_frac=0.001, seed=0):
    scale = min(1.0, max_side / max(a.shape[:2]))
    if scale < 1:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    px = np.vstack([a.reshape(-1, 3), b.reshape(-1, 3)]).astype(np.float32)
    rng = np.random.default_rng(seed)
    sub = px[rng.choice(len(px), min(len(px), 20000), replace=False)]
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(sub, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)

    def labels(img):
        d = ((img.reshape(-1, 1, 3).astype(np.float32) - centers[None]) ** 2).sum(-1)
        return d.argmin(1).reshape(img.shape[:2])

    def components(lab):
        out = []
        min_area = min_frac * lab.size
        for c in range(k):
            n, _, stats, _ = cv2.connectedComponentsWithStats((lab == c).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area >= min_area:
                    out.append((c, (x, y, w, h), area))
        return out

    A = components(labels(a))
    B = components(labels(b))
    total = sum(x[2] for x in A) + sum(x[2] for x in B)
    if total == 0:
        return 0.0
    used = set()
    unmatched = 0
    for ca, ba, aa in A:
        best, best_iou = None, 0.5
        for j, (cb, bb, _) in enumerate(B):
            if j in used or cb != ca:
                continue
            iou = _iou(ba, bb)
            if iou >= best_iou:
                best, best_iou = j, iou
        if best is None:
            unmatched += aa
        else:
            used.add(best)
    unmatched += sum(area for j, (_, _, area) in enumerate(B) if j not in used)
    return float(unmatched / total)


# --- pretrained tile features (frozen; no training)
_ROOT = config.HUB_CACHE
_models = {}


def _resnet():
    if "rn" not in _models:
        m = torchvision.models.resnet18(weights=None)
        m.load_state_dict(torch.load(_ROOT / "checkpoints/resnet18-f37072fd.pth", map_location="cpu"))
        _models["rn"] = m.eval()
    return _models["rn"]


def _dino():
    if "dino" not in _models:
        m = torch.hub.load(str(_ROOT / "facebookresearch_dinov2_main"), "dinov2_vits14",
                           source="local", pretrained=False)
        m.load_state_dict(torch.load(_ROOT / "checkpoints/dinov2_vits14_pretrain.pth", map_location="cpu"))
        _models["dino"] = m.eval()
    return _models["dino"]


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_tensor(rgb, size):
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255
    return (t - MEAN) / STD


@torch.no_grad()
def _tile_tokens(name, rgb):
    x = _to_tensor(rgb, 512 if name == "rn" else 448)
    if name == "rn":
        m = _resnet()
        f = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
    else:
        f = _dino().forward_features(x)["x_norm_patchtokens"]
        f = f.transpose(1, 2).reshape(1, -1, 32, 32)
    f = torch.nn.functional.adaptive_avg_pool2d(f, GRID)
    return torch.nn.functional.normalize(f.flatten(2)[0].T, dim=-1)  # (256, C)


def deep_tiles(name, a, b):
    """Per-tile cosine dissimilarity -> worst tile and mean of the 5 worst tiles."""
    d = 1 - (_tile_tokens(name, a) * _tile_tokens(name, b)).sum(-1)
    top5 = torch.topk(d, 5).values.mean().item()
    return {"max": float(d.max()), "top5": float(top5)}


def features_from_arrays(a, b, ref_rgb01=None, cand_rgb01=None):
    """All static layers for RGB uint8 arrays a (reference) and b (candidate), already the same size."""
    ga, gb = _gray(a), _gray(b)
    rn = deep_tiles("rn", a, b)
    dino = deep_tiles("dino", a, b)
    pix = pixel.compare_arrays(a.astype(np.float32) / 255, b.astype(np.float32) / 255)
    return {
        "pixel_ssim_tile": pix["score"],
        "gmsd": gmsd(ga, gb),
        "chamfer_edges": chamfer(ga, gb),
        "colour_hellinger": colour_hellinger(a, b),
        "lbp_chi2": lbp_chi2(ga, gb),
        "orb_homography": orb_homography(ga, gb),
        "segment_unmatched": segment_unmatched(a, b),
        "resnet_tile_max": rn["max"],
        "resnet_tile_top5": rn["top5"],
        "dino_tile_max": dino["max"],
        "dino_tile_top5": dino["top5"],
    }


def static_features(ref_path, cand_path):
    a, b = load_pair(ref_path, cand_path)
    return features_from_arrays(a, b)
