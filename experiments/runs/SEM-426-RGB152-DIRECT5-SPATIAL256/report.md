# SEM-426-RGB152-DIRECT5-SPATIAL256

Status: completed, independently audited, closed and not promoted.

## Question

Can the smallest native 3D semantic representation—five trainable logits per
frozen RGB152 Gaussian—generalize across the same spatial holdout as the
matched SEM396 two-dimensional Mask2Former baseline?

## Fixed protocol

- Parent engineering gate: SEM425, 22/22 independent checks passed.
- Reconstruction: frozen RGB152, 789,281 Gaussians.
- Split: `spatial_confirmation_v1`, with 256 semantic training views, 30
  spatial confirmation views and 17 excluded buffer views.
- Optimization: 6,000 updates, Adam learning rate 0.05, median-frequency
  weighted NLL, and exactly the SEM396 crop/view/flip trace.
- Endpoint selection: step 6,000 fixed before confirmation access; step 3,000
  is retained only as a descriptive checkpoint.
- Trainable state: 3,946,405 semantic parameters. RGB geometry, opacity, SH,
  scale, rotation and exposure remained frozen.

## Execution evidence

Training completed all 6,000 updates in 399.631 seconds and accessed all 256
training views. Before checkpoint sealing it accessed zero confirmation RGB,
zero confirmation labels and zero buffer data. Peak reserved GPU memory was
6,586 MiB. The step-6,000 checkpoint SHA-256 is
`31ba2e6fc6f6ccfc799946b2d6a760a136a0e766be66dd8bbae3dfb5bc597968`.

Checkpoint replay was exact. RGB parameter hashes were unchanged, and a fresh
RGB render before/after training was bitwise equal with maximum absolute pixel
difference zero. The sealed checkpoint hash was unchanged by confirmation
inference.

## Result

| split | method | five-class mIoU | foreground mIoU | stay_cable IoU | stay_cable boundary F1 |
|---|---|---:|---:|---:|---:|
| train | direct 3D logits | 65.454847% | 61.794288% | 29.796455% | 13.991218% |
| spatial confirmation | direct 3D logits | 49.914356% | 41.946686% | 9.395042% | 12.750150% |
| train | SEM396 2D baseline | 95.375428% | 94.367004% | 95.395787% | 88.187853% |
| spatial confirmation | SEM396 2D baseline | 89.204606% | 86.667377% | 89.355622% | 79.519325% |

Relative to SEM396 on the same 30 confirmation views, direct 3D logits change
five-class mIoU by -39.290250 pp, foreground mIoU by -44.720691 pp,
`stay_cable` IoU by -79.960580 pp and 2 px boundary F1 by -66.769176 pp.

The failure is spatially concentrated but not confined to one region. Region-1
foreground mIoU is 25.123766% and `stay_cable` IoU is 1.569984%; region-2 is
59.223733% / 13.464524%; region-3 is 55.394908% / 11.017696%. The worst six
confirmation views are 153, 157, 158, 159, 160 and 161, with mean foreground
mIoU 13.856129%.

The `stay_cable` field has low precision (10.482504%) and moderate recall
(47.523850%), so the main failure is not merely missing thin cables: the shared
Gaussian labels also spread cable identity into large non-cable regions. That
is consistent with view-dependent occlusion and mixed ray responsibility being
collapsed into one context-free label distribution per Gaussian.

## Independent audit and decision

`scripts/audit_sem426_direct5_spatial.py` independently re-read all 286 saved
predictions, rebuilt every train/holdout/region confusion matrix, checked all
prediction hashes and role counts, and verified checkpoint/RGB isolation. All
20 checks passed.

This experiment is closed without promotion. It establishes that native 3D
semantic rendering is technically feasible, but the minimal five-logit field
is scientifically inadequate and must not replace, fuse with or tune against
the retained two-dimensional system. No second seed, second fold, learning-rate
search, class-weight search or post-processing rescue is authorized under this
ID. A future 3D attempt must change the representation or responsibility unit
(for example view-conditioned features or cross-view surface identity) and use
a new registered experiment.

Evidence:

- `outputs/quality/SEM-426/r1/training/result.json`
- `outputs/quality/SEM-426/r1/inference/result.json`
- `outputs/quality/SEM-426/r1/audit.json`
- `docs/experiments/sem426_direct5_spatial_protocol.md`
