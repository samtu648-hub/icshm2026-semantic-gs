# SEM-022 — relative-maximum contribution-aware visibility supervision

- Status: completed; **drop for S1 confirmation**.
- Parent: `SEM-001`, `SEM-020-EQ`, `SEM-020-DIAG`, `SEM-022-PF`.
- Code freeze: `56695181985332e55f030567f3293758638d2ef9`.

This completed the 0→10,000 single-seed main-fold comparison with only the relative-maximum stop-gradient contribution gate (threshold 0.05) changed.

## Execution note

Three service-launch attempts exited before iteration 1: first a missing Git-ignored output directory, then two invocations whose `PATH` omitted the R1 environment's existing `ninja`. These are retained at the start of the external stdout/stderr log. The successful retry used the same frozen config, code commit and R1 CUDA-extension cache, and begins at its recorded step 1; no failed attempt produced a checkpoint, metric or quality result.

## Evidence

- Result artifact: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/sem022_full/result.json`, SHA-256 `7049f0751f5da570fd2eff41966a34a2d66f67bb8fa6117cd1057176b53b6067`.
- Semantic checkpoint: 3,784,168 bytes, SHA-256 `af3248d471f544b378871c661c5eeb96fc90cb7192ae7f8eb64130d686233cb4`; reload passed.
- Training: 10,000 iterations, 2,515.12 seconds / 0.69864 GPU-hours; peak PyTorch allocation 843 MiB.
- RGB isolation: frozen artifact hashes and RGB PNG hashes identical; RGB maximum absolute pixel difference `0`.
- External official `CAMERA_FROM_WORLD` camera: identity Sim(3), RGB `1320×989`, five logits `5×989×1320`, uint8 semantic PNG in `0..4`; no test RGB or labels used.

## Fixed 74-view local result

| Metric | SEM-001 | SEM-022 | Delta |
|---|---:|---:|---:|
| mIoU, including background | 0.61610 | 0.63121 | +1.51 pp |
| mIoU, excluding background | 0.56334 | 0.57977 | +1.64 pp |
| stay_cable IoU | 0.06876 | 0.07751 | +0.87 pp |
| stay_cable precision / recall | 0.07796 / 0.36804 | 0.08790 / 0.39601 | +0.99 / +2.80 pp |
| background→stay_cable | 13,166,331 / 15.0076% | 12,405,892 / 14.1408% | −760,439 px / −0.867 pp |
| worst 10% mIoU excluding background | 0.42344 | 0.42831 | +0.49 pp |
| 140–146 mIoU excluding background | 0.60121 | 0.61275 | +1.15 pp |

Other class IoU changes versus SEM-001 are deck −0.37 pp, tower −0.89 pp and foundation +6.96 pp. The relative-max gate retained 91.26% contribution mass and 44.14% contributors during training; its direct frozen training-view diagnostic retained 89.90% semantic probability-gradient mass and 93.85% anchor support.

## Decision

SEM-022 improves the target error and clears the +1.0 pp mIoU and nondecreasing stay_cable IoU conditions. It nevertheless violates the pre-registered protected-class condition because tower IoU falls 0.887 pp, exceeding the allowed 0.5 pp decline. It therefore does **not** enter three-fold/three-seed confirmation and cannot authorize SEM-023, S2, or S3.
