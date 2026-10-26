# app-visual-verify

Multi-layer verification of application screenshots and screen recordings:
a reference app (C# MAUI, `maui`) against a copycat implementation (the C++ port, `cpp`).

`verify <reference> <candidate>` prints per-layer scores and a
SAME / MINOR / DIFFERENT verdict as JSON.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
export VERIFY_CAPTURES=/path/to/maui/port/cpp/docs/comparison/captures

.venv/bin/python -m verify verify \
  $VERIFY_CAPTURES/ios/maui/button_light.png $VERIFY_CAPTURES/ios/cpp/button_light.png
.venv/bin/python -m verify verify --out out.json \
  $VERIFY_CAPTURES/ios/maui/switch_light.gif $VERIFY_CAPTURES/ios/cpp/switch_light.gif
.venv/bin/python -m verify.evaluate          # reproduces data/results/
.venv/bin/python -m pytest -q tests
```

Two PNGs give kind `static`; two GIFs give kind `gif`. Mixing kinds is an error.

## Layers

| Layer | Input | Status |
|---|---|---|
| `pixel` | first frame, RGB, per-tile SSIM on a 16x16 grid | implemented; primary for static (score = worst tile) |
| `temporal` | GIF, DTW over 1-SSIM frame costs | implemented; primary for gif (same algorithm as `src/gif/gifdtw.py`) |
| `layout` | layout trees exported by the port harness | stub, reports unavailable until the format exists |

Verdict thresholds are in `config/thresholds.json`, written by the evaluation.
They are fitted in-sample. The MINOR band `[0.5·diff, diff)` is a guess: there are
too few MINOR labels to fit it.

## Evaluation

`python -m verify.evaluate` scores each layer against the labels: AUC with
bootstrap CI, leave-one-out thresholds, and a fixed held-out split (orig76 train /
new32 test). Output goes to `data/results/` (`eval.json`, `SUMMARY.md`, per-pair
scores). Current numbers are in `data/results/SUMMARY.md`.

## Configuration

- `VERIFY_CAPTURES` (or `--captures` for the evaluation): the captures root,
  `{platform}/{maui,cpp}/{page}_{theme}.{png,gif}`. Captures are read in place from the
  MAUI repo and are not copied here.
- Labels and truth files default to paths under `data/`, relative to this repo.
- `src/siamese/*.py` and `src/gif/gifdtw.py` read `VERIFY_CAPTURES`. The Siamese
  scripts also need `SCRATCH` (features and labels in the layout of the MAUI scratchpad).

## Layout

- `src/verify/` the package: `cli.py`, `verdict.py`, `evaluate.py`, `layers/`, `config.py`.
- `data/labels/` human labels (`labels.json`, from the parity labeller) and the
  blind-judge truth file (`private_truth.json`, outside any judge's view).
- `data/results/` evaluation output and the earlier Siamese training logs.
- `src/gif/` GIF DTW prototype and its model-assigned label file (not human labels).
- `src/siamese/` frozen-encoder feature extraction and Siamese head training (prototype).
- `config/thresholds.json` verdict thresholds (generated).
- `PLAN.md` the open design questions.

## Provenance

- `labels.json`, `orig76.json`, `v2_results.txt`, `v3_results.txt` come from
  MAUI repo commit 1015d15 (path `port/cpp/tools/parity/ml/data/`).
- `src/gif/gif_labels_claude.json` are model-assigned labels, not human labels.
