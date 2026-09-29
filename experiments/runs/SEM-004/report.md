# SEM-004 — restricted first-order observation-direction residual

- Status: **completed; drop/unproven on the main fold**.
- Phase: R2/S3 single-fold screen.
- Parent: frozen `SEM-001`; `SEM-004-DIAG` supplies the authenticated 7,069-anchor mask but is not a quality result.
- Run commit/worktree evidence: `39cd07de5b60646e0c37db5f02562a10ccad0602` / `b731ee6567007bb92923bba4754967a48dbd0d8cfd4fb13186bd9b1d08806ca6`.
- External result/output-manifest SHA-256: `23752cb5f6df886ebde277ddd61a06b33d4144f3fcea70611c56f4cbbe2daf99` / `f984b1dd0bbfb1e8f7251dc523190cd5b8605713ab008e6df37e0186636d12da`.

## Unique variable

For each enabled anchor, three zero-initialized coefficients receive the official-frame direction `normalize(anchor_position - camera_center)`. The capped first-order scalar subtracts from the frozen background logit and adds to the frozen `stay_cable` logit. The frozen static logits for deck, tower and foundation are directly unchanged; all RGB and static semantic parameters remain immutable.

The cap (`1.0`), L2 (`1e-4`), AdamW learning rate (`0.01`), seed (`20260731`), SEM-001 sampler, 10,000-iteration budget and keep/drop gate were fixed before any validation result was observed. Checkpoints and 74-view results were saved at 1k/3k/5k/10k; no intermediate result changed the run.

## Actual run and engineering evidence

The single continuous run completed in 3,294.01 s (0.9150 GPU h) on RTX 5070. Peak PyTorch allocation was 822 MiB. The final residual checkpoint is 144,829 bytes, SHA-256 `5d598f08559458907dcefb90660a4124735ffefccc5508db3f36b0dadb6e3b1a`, and reloaded bit-identically.

All RGB artifact and rendered-PNG hashes remained equal; RGB maximum absolute pixel difference was `0`. Frozen SEM-001 logits were bit-identical after training. The official external camera interface passed `CAMERA_FROM_WORLD`, identity Sim(3), input-derived 1320×989 size, RGB, five logits and uint8 0–4 semantic PNG. No guard-band, unlabeled RGB, validation-selected hyperparameter or hidden test input was used.

At 10k all 7,069 anchors had nonzero coefficient norms. Observed absolute delta had mean `0.1780`, p95 `0.8723` and maximum `1.0000`; nevertheless the rendered class decision changed too little. Direct deck/tower/foundation residual logits remained exactly zero by construction.

## Fixed 74-view result

| Metric | SEM-001 | SEM-004 | Delta |
|---|---:|---:|---:|
| mIoU with background | 0.616103 | 0.616226 | +0.000123 (+0.0123 pp) |
| mIoU without background | 0.563337 | 0.563406 | +0.000070 |
| background IoU | 0.827169 | 0.827506 | +0.000338 |
| deck IoU | 0.900999 | 0.900990 | −0.000009 |
| stay_cable IoU | 0.068759 | 0.068957 | +0.000197 |
| tower IoU | 0.834870 | 0.834897 | +0.000028 |
| foundation IoU | 0.448718 | 0.448781 | +0.000063 |
| background→stay_cable | 13,166,331 | 13,137,079 | −29,252 (−0.2222%) |
| worst 10% non-bg mIoU | 0.423440 | 0.423471 | +0.000031 |
| 140–146 non-bg mIoU | 0.601213 | 0.601593 | +0.000380 |

Stay-cable precision/recall are `0.078200 / 0.368429`, both only marginally above SEM-001. In the paired audit, 53/74 views improved total mIoU and 54/74 reduced background→stay_cable, but the magnitudes were negligible. The 1k/3k/5k/10k mIoU values were 0.616034/0.616149/0.616239/0.616226; the registered final comparison remains the 10k result, not the post-hoc best 5k snapshot.

## Decision

`drop_unproven_main_fold`. Eight quality/engineering checks passed, including stay_cable non-regression, all three protected classes, worst/hard views, RGB=0, reload and external camera. Two required checks failed:

- required mIoU gain was at least +1.0 pp; observed gain was only +0.0123 pp;
- required clear background→stay_cable reduction was at least 1%; observed reduction was 0.2222%.

The training-side direction correlation therefore did not translate into a meaningful single-fold semantic gain under this restricted causal intervention. This is a negative hypothesis result, not a CUDA/interface/code failure. SEM-001 remains frozen. Per the pre-registration, cap, mask, L2, learning rate and seed will not be changed or rerun, and S4, pseudo-labeling, three-fold/three-seed confirmation and GeometryPrior were not started.
