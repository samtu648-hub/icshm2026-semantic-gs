# SEM-021 — cumulative contribution-aware visibility supervision

- Status: completed; **drop for S1 confirmation**.
- Parent: `SEM-001`, `SEM-020-EQ`, `SEM-020-DIAG`, `SEM-021-PF`.
- Code freeze: `56695181985332e55f030567f3293758638d2ef9`.
- Output: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/sem021_full/`.

This completed the pre-registered 0→10,000 single-seed main-fold comparison. It changes only the stop-gradient cumulative contribution gate (threshold 0.95); RGB geometry, opacity, color, MLPs and optimizer state remained frozen.

## Evidence

- Result artifact: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/sem021_full/result.json`, SHA-256 `936014854090eb58193e31a8bb499498a50dbc18577b61a370a3c03890a4a03c`.
- Semantic checkpoint: 3,784,168 bytes, SHA-256 `9236016cb0ea57dd0d26d0550c09acb7a7f772eeeee8c448648489765f381a45`; reload passed.
- Training: 10,000 iterations, 2,413.91 seconds / 0.67053 GPU-hours; peak PyTorch allocation 843 MiB.
- RGB isolation: frozen artifact hashes and RGB PNG hashes identical; RGB maximum absolute pixel difference `0`.
- External official `CAMERA_FROM_WORLD` camera: identity Sim(3), RGB `1320×989`, five logits `5×989×1320`, uint8 semantic PNG in `0..4`; no test RGB or labels used.

## Fixed 74-view local result

| Metric | SEM-001 | SEM-021 | Delta |
|---|---:|---:|---:|
| mIoU, including background | 0.61610 | 0.62411 | +0.80 pp |
| mIoU, excluding background | 0.56334 | 0.57296 | +0.96 pp |
| stay_cable IoU | 0.06876 | 0.07396 | +0.52 pp |
| stay_cable precision / recall | 0.07796 / 0.36804 | 0.08340 / 0.39517 | +0.54 / +2.71 pp |
| background→stay_cable | 13,166,331 / 15.0076% | 13,129,314 / 14.9654% | −37,017 px / −0.042 pp |
| worst 10% mIoU excluding background | 0.42344 | 0.43128 | +0.78 pp |
| 140–146 mIoU excluding background | 0.60121 | 0.60965 | +0.84 pp |

The cumulative gate retained 95.54% contribution mass and 65.06% contributors over validation; the direct frozen training-view diagnostic retained 95.63% semantic probability-gradient mass and 96.44% anchor support.

## Decision

The direction is positive, but the pre-registered main-fold gate requires at least +1.0 pp mIoU. SEM-021 reaches only +0.80 pp including background and +0.96 pp excluding background; tower IoU also changes −0.54 pp, beyond the −0.5 pp protection line. It therefore does **not** enter three-fold/three-seed confirmation and does not authorize SEM-023, S2, or S3.
