"""Trained change detector (early-fusion, FC-EF style; Daudt et al., 2018) with a
small U-Net-like encoder/decoder (Ronneberger et al., 2015).

Input: reference and candidate stacked as 6 channels. Output: a per-pixel change
logit map. Score for an image pair = the largest window-level change probability
(mean over 8x8 blocks), so a single changed region is enough.

Training data (never the labelled pages):
  - synthetic: patches of MAUI reference captures that are not in any label file,
    perturbed with rectangle fills, shifts, copies, recolours and strokes (positives)
    or with photometric jitter and tiny colour deltas (negatives).
  - real (fine-tuning): labelled static pairs with image-level labels only, using
    multiple-instance learning over sliding windows.
"""
import json
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import config

WIN = 128
STRIDE = 96
PATCH = 256
CACHE = config.REPO_ROOT / "cache"
torch.set_num_threads(8)


class ChangeNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.enc1 = block(6, 16)
        self.down = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(block(32, 32), block(32, 32))
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec = nn.Sequential(block(32, 16), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        e = self.enc1(x)
        m = self.mid(self.down(e))
        return self.dec(torch.cat([e, self.up(m)], 1))[:, 0]


def to_input(a, b):
    """uint8 HxWx3 pair -> float tensor 1x6xHxW in [0, 1]."""
    x = np.concatenate([a, b], -1).astype(np.float32) / 255
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def window_origins(h, w):
    ys = list(range(0, max(h - WIN, 0) + 1, STRIDE))
    xs = list(range(0, max(w - WIN, 0) + 1, STRIDE))
    if ys[-1] != max(h - WIN, 0):
        ys.append(max(h - WIN, 0))
    if xs[-1] != max(w - WIN, 0):
        xs.append(max(w - WIN, 0))
    return [(y, x) for y in ys for x in xs]


def window_scores(model, x, batch=32):
    """Differentiable per-window scores for a 1x6xHxW tensor (window max of 8x8 mean)."""
    _, _, h, w = x.shape
    if h < WIN or w < WIN:
        pad = (0, max(0, WIN - w), 0, max(0, WIN - h))
        x = F.pad(x, pad)
        h, w = x.shape[2:]
    origins = window_origins(h, w)
    scores = []
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        xb = torch.cat([x[:, :, y:y + WIN, xx:xx + WIN] for y, xx in chunk], 0)
        p = torch.sigmoid(model(xb)).unsqueeze(1)
        scores.append(F.avg_pool2d(p, 8).flatten(1).amax(1))
    return torch.cat(scores)


@torch.no_grad()
def score_pair(model, a, b):
    model.eval()
    return float(window_scores(model, to_input(a, b)).max())


# ---- synthetic data ------------------------------------------------------

def _rect(rng, h, w):
    rh, rw = rng.randint(8, 72), rng.randint(8, 72)
    y, x = rng.randint(0, h - rh), rng.randint(0, w - rw)
    return y, x, rh, rw


def make_pair(patch, rng):
    """Return (changed uint8 patch, binary change mask) for a clean uint8 patch."""
    h, w, _ = patch.shape
    c = patch.astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    if rng.random() < 0.5:
        # negative: photometric jitter, noise, and a sub-visible colour delta
        c = c * rng.uniform(0.97, 1.03) + rng.uniform(-4, 4)
        c += np.random.randn(h, w, 1).astype(np.float32) * rng.uniform(0, 2)
        y, x, rh, rw = _rect(rng, h, w)
        c[y:y + rh, x:x + rw] += rng.uniform(-6, 6)
    else:
        for _ in range(rng.randint(1, 3)):
            y, x, rh, rw = _rect(rng, h, w)
            op = rng.randint(0, 4)
            if op == 0:
                c[y:y + rh, x:x + rw] = [rng.randint(0, 255) for _ in range(3)]
                mask[y:y + rh, x:x + rw] = 1
            elif op == 1:
                c[y:y + rh, x:x + rw] = 255 - c[y:y + rh, x:x + rw]
                mask[y:y + rh, x:x + rw] = 1
            elif op == 2:
                dy, dx = rng.randint(2, 12) * rng.choice([-1, 1]), rng.randint(2, 12) * rng.choice([-1, 1])
                region = c[y:y + rh, x:x + rw].copy()
                c[y:y + rh, x:x + rw] = np.median(c.reshape(-1, 3), 0)
                ny, nx = min(max(y + dy, 0), h - rh), min(max(x + dx, 0), w - rw)
                c[ny:ny + rh, nx:nx + rw] = region
                mask[y:y + rh, x:x + rw] = 1
                mask[ny:ny + rh, nx:nx + rw] = 1
            elif op == 3:
                sy, sx = rng.randint(0, h - rh), rng.randint(0, w - rw)
                c[y:y + rh, x:x + rw] = c[sy:sy + rh, sx:sx + rw].copy()
                mask[y:y + rh, x:x + rw] = 1
            else:
                canvas = np.ascontiguousarray(c.copy())
                for _ in range(rng.randint(3, 6)):
                    p1 = (x + rng.randint(0, rw), y + rng.randint(0, rh))
                    p2 = (x + rng.randint(0, rw), y + rng.randint(0, rh))
                    col = tuple(rng.randint(0, 255) for _ in range(3))
                    cv2.line(canvas, p1, p2, col, rng.randint(1, 2))
                c = canvas.astype(np.float32)
                mask[y:y + rh, x:x + rw] = 1
    return np.clip(c, 0, 255).astype(np.uint8), mask


def _labelled_bases():
    """(platform, page) pairs that appear in any label file; never used as synthetic bases."""
    excluded = set()
    for key in json.loads(config.DEFAULT_LABELS.read_text()):
        page, platform, _ = key.split("/")
        excluded.add((platform, page))
    for key in json.loads(config.DEFAULT_TRUTH.read_text()):
        platform, name = key.split("/")
        excluded.add((platform, name.rsplit("_", 1)[0]))
    return excluded


def build_patch_pool(root, n_images=400, per_image=3, seed=0):
    path = CACHE / "synthetic_patches.npz"
    if path.exists():
        return np.load(path)["patches"]
    rng = random.Random(seed)
    excluded = _labelled_bases()
    files = []
    for p in sorted(root.glob("*/maui/*.png")):
        platform = p.parts[-3]
        page = p.stem.rsplit("_", 1)[0]
        if (platform, page) not in excluded:
            files.append(p)
    rng.shuffle(files)
    patches = []
    for p in files[:n_images]:
        img = np.asarray(Image.open(p).convert("RGB"))
        h, w, _ = img.shape
        if h < PATCH or w < PATCH:
            continue
        for _ in range(per_image):
            y, x = rng.randint(0, h - PATCH), rng.randint(0, w - PATCH)
            patches.append(img[y:y + PATCH, x:x + PATCH])
    arr = np.stack(patches)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, patches=arr)
    return arr


def _batch(patches, rng, bs=32):
    xs, ys = [], []
    for _ in range(bs):
        p = patches[rng.randint(0, len(patches) - 1)]
        c, m = make_pair(p, rng)
        # half of the crops centred on a change so positives are common
        if m.any() and rng.random() < 0.5:
            yy, xx = np.nonzero(m)
            k = rng.randint(0, len(yy) - 1)
            cy, cx = yy[k], xx[k]
            y0 = min(max(cy - WIN // 2 + rng.randint(-40, 40), 0), PATCH - WIN)
            x0 = min(max(cx - WIN // 2 + rng.randint(-40, 40), 0), PATCH - WIN)
        else:
            y0, x0 = rng.randint(0, PATCH - WIN), rng.randint(0, PATCH - WIN)
        a = p[y0:y0 + WIN, x0:x0 + WIN]
        b = c[y0:y0 + WIN, x0:x0 + WIN]
        xs.append(to_input(a, b)[0])
        ys.append(torch.from_numpy(m[y0:y0 + WIN, x0:x0 + WIN]))
    return torch.stack(xs), torch.stack(ys)


def _loss(logits, target):
    pos_w = torch.tensor(3.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_w)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def train_synthetic(root, steps=2000, lr=1e-3, seed=0, path=None):
    path = path or CACHE / "changenet_synthetic.pt"
    if path.exists():
        m = ChangeNet()
        m.load_state_dict(torch.load(path))
        return m.eval()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    np.random.seed(seed)
    patches = build_patch_pool(root)
    model = ChangeNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for step in range(steps):
        x, y = _batch(patches, rng)
        loss = _loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            print(f"synthetic step {step} loss {loss.item():.4f}", flush=True)
    CACHE.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval()


def finetune_mil(model, pairs, labels, epochs=2, lr=1e-4, seed=0):
    """pairs: list of (ref_rgb, cand_rgb) uint8; labels: 0/1 image-level. Returns a new model."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = ChangeNet()
    m.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()
    order = list(range(len(pairs)))
    for ep in range(epochs):
        rng.shuffle(order)
        for i in order:
            a, b = pairs[i]
            s = window_scores(m, to_input(a, b)).max().clamp(1e-4, 1 - 1e-4)
            y = float(labels[i])
            loss = -(y * torch.log(s) + (1 - y) * torch.log(1 - s))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()
