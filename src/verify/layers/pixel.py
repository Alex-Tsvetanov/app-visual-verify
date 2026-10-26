"""Layer 'pixel': per-tile SSIM on the first frame, RGB, on a fixed grid.

Score is the worst tile's dissimilarity (1 - mean SSIM inside the tile), so one
badly wrong region is not averaged away. Higher means more different.
"""
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

GRID = 16


def load_rgb(path, frame=0):
    with Image.open(path) as im:
        im.seek(frame)
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255


def tile_means(a, grid):
    rows = np.array_split(np.arange(a.shape[0]), min(grid, a.shape[0]))
    cols = np.array_split(np.arange(a.shape[1]), min(grid, a.shape[1]))
    return np.array([[a[np.ix_(r, c)].mean() for c in cols] for r in rows])


def compare(ref_path, cand_path, grid=GRID):
    ref = load_rgb(ref_path)
    cand = load_rgb(cand_path)
    size_mismatch = ref.shape != cand.shape
    if size_mismatch:
        # Compare on the reference grid; the mismatch is reported, not hidden.
        h, w = ref.shape[:2]
        resized = Image.fromarray((cand * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        cand = np.asarray(resized, dtype=np.float32) / 255
    mean_ssim, smap = ssim(ref, cand, data_range=1.0, channel_axis=2, full=True)
    tiles = tile_means(1 - smap.mean(axis=2), grid)
    return {
        "available": True,
        "score": float(tiles.max()),
        "details": {
            "max_tile": float(tiles.max()),
            "mean_tile": float(tiles.mean()),
            "global_dissimilarity": float(1 - mean_ssim),
            "grid": grid,
            "size_mismatch": bool(size_mismatch),
            "ref_size": [ref.shape[1], ref.shape[0]],
            "cand_size": [cand.shape[1], cand.shape[0]],
        },
    }
