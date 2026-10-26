# Plan: multi-layer verification

## Facts so far
- Static Siamese head on frozen ResNet-18 features: LOO AUC 0.66–0.70 on 108 labels. Grok v2 hand features: LOO 0.91 (held-out 0.85).
- Training-free top-5 tile cosine (ResNet-18, 16x16 grid): AUC 0.79. Not yet verified on a held-out set.
- GIF DTW over 1-SSIM frame costs: AUC 0.56 on 34 pairs with 4 positives. Does not validate. DTW absorbs easing and timing; mean SSIM dilutes small changes.
- Gemini advice: deterministic clock in the C++ harness, then per-frame comparison; layout-tree JSON diff for geometry.

## Open questions
1. Ground truth: are the 14 user labels plus the 20 model labels enough? Need human re-labelling for the GIF set.
2. Judge comparison: compare Haiku, Sonnet and Opus verdicts with human labels and with each other (agreement, kappa). Verdict files will be written to the scratchpad and need copying here.
3. Layers: layout-tree JSON diff (needs harness export), pixel/SSIM with per-tile minimum, learned perceptual (DINOv2/LPIPS), temporal alignment (deterministic clock preferred).
4. Interface: CLI that takes (reference, candidate) and emits per-layer scores plus a verdict.

## Next
- Make data paths configurable; remove hard-coded `/Users/...` paths.
- Re-check the GIF labels with a second human pass.
