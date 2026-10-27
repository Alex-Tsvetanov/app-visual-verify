"""Where inputs live. Nothing here is tied to one machine.

Captures are read in place from the MAUI repo. Set VERIFY_CAPTURES (or pass
--captures to the evaluation) to the directory that holds
{platform}/{maui,cpp}/{page}_{theme}.{png,gif}.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LABELS = REPO_ROOT / "data/labels/port/cpp/tools/parity/ml/data/labels.json"
DEFAULT_TRUTH = REPO_ROOT / "data/labels/private_truth.json"
DEFAULT_MODEL_GIF_LABELS = REPO_ROOT / "src/gif/gif_labels_claude.json"
DEFAULT_GIF_DTW = REPO_ROOT / "src/gif/all_dtw.json"
DEFAULT_ORIG76 = REPO_ROOT / "data/labels/port/cpp/tools/parity/ml/data/orig76.json"
THRESHOLDS = REPO_ROOT / "config/thresholds.json"
RESULTS = REPO_ROOT / "data/results"

ENV_CAPTURES = "VERIFY_CAPTURES"
HUB_CACHE = Path.home() / ".cache/torch/hub"  # pretrained ResNet-18 and DINOv2 weights


def captures_root(override=None):
    raw = override or os.environ.get(ENV_CAPTURES)
    if not raw:
        raise SystemExit(f"captures directory not set: pass --captures or export {ENV_CAPTURES}")
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise SystemExit(f"captures directory not found: {root}")
    return root


def pair_paths(root, platform, name, ext):
    """Reference (maui) and candidate (cpp) capture for one page/theme, e.g. name='button_light'."""
    ref = root / platform / "maui" / f"{name}.{ext}"
    cand = root / platform / "cpp" / f"{name}.{ext}"
    return ref, cand
