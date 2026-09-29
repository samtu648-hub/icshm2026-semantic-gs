# SEM-434-RGB152-MASK2FORMER-LONG12K

Status: completed on 2026-09-22 as a historically exposed development curve;
not a blind or hidden-test result. The authoritative attempt is `r2`.

## Recovery and execution

SEM396's step-6000 file contains model weights but no AdamW, GradScaler, torch
RNG or sampler RNG state, so it cannot be resumed reliably. The initial r1 run
used the same ADE initialization and exact 6000-sample prefix but differed from
SEM396 in 625/650 tensors at step 6000 (maximum absolute difference 0.0530541;
all-parameter mean absolute difference 0.000412042) and stopped at the original
over-strict historical-equality gate. Two earlier zero-update environment failures
and this r1 training failure are preserved under `outputs/quality/SEM-434/r1`.

r2 therefore uses its own sealed step 6000 as the paired control and continues in
the same process, optimizer, AMP and sampler state. Steps 1--6000 use the original
poly schedule; steps 6001--12000 use fixed per-group LR `1e-5` / backbone `1e-6`.
The actual 12000-sample trace matches the registered plan and its first 6000
entries match SEM396's plan. RGB hashes are unchanged.

Training completed 12000 successful updates in 3111.272 seconds with 12 AMP
skips. Peak allocated/reserved memory was 8084.516/8564 MiB. Four model
checkpoints and full resume checkpoints at 6000/12000 were saved. Four-endpoint
inference took 774.104 seconds with 1968 MiB peak reserved. The whole SEM434
directory, including preserved failures, uses 9.5 GiB.

## Full v1 curve (%)

| Step | Train mIoU | Confirm mIoU | Train FG | Confirm FG | mIoU gap | FG gap | Cable boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6000 | 95.382456 | 89.774856 | 94.375736 | 87.362879 | 5.607599 | 7.012857 | 79.509264 |
| 8000 | 95.466139 | 90.151646 | 94.476668 | 87.821090 | 5.314494 | 6.655578 | 81.434597 |
| 10000 | 95.515005 | 89.976609 | 94.536702 | 87.600187 | 5.538397 | 6.936516 | 79.814495 |
| 12000 | 95.612647 | 90.099000 | 94.654627 | 87.752179 | 5.513647 | 6.902447 | 80.765547 |

| Step | Cable IoU | Precision | Recall | FP | FN | Deck IoU | Tower IoU | Foundation IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6000 | 89.149770 | 96.958735 | 91.714402 | 38,640 | 111,290 | 91.282705 | 88.715127 | 80.303916 |
| 8000 | 90.104557 | 96.981547 | 92.704370 | 38,755 | 97,993 | 91.634157 | 89.076029 | 80.469619 |
| 10000 | 89.760197 | 97.087748 | 92.243819 | 37,165 | 104,179 | 91.641094 | 88.900458 | 80.098998 |
| 12000 | 89.795242 | 96.982663 | 92.375969 | 38,603 | 102,404 | 92.067919 | 88.940887 | 80.204669 |

## Regions and overfitting

| Step | Region 1 mIoU / boundary | Region 2 mIoU / boundary | Region 3 mIoU / boundary | Worst view FG mIoU |
|---:|---:|---:|---:|---:|
| 6000 | 86.370480 / 77.547747 | 90.761821 / 72.898550 | 91.480469 / 92.311799 | 161.png: 73.940891 |
| 8000 | 87.621225 / 80.829518 | 90.750276 / 74.646821 | 91.377579 / 92.397071 | 160.png: 75.288692 |
| 10000 | 87.216215 / 79.886969 | 90.698170 / 71.509615 | 91.568285 / 92.009356 | 160.png: 74.710678 |
| 12000 | 87.336917 / 79.842457 | 91.120576 / 76.579935 | 90.955234 / 88.376023 | 160.png: 74.841254 |

Training mIoU rises monotonically, but confirmation peaks at 8000 and falls at
10000; 12000 only partly recovers. The first overfitting signal is therefore after
8000. At 8000, cable FN decreases by 13,297 while FP rises by only 115, so the
gain is overwhelmingly reduced omission. All deck/tower/foundation IoUs improve.
At 12000 the overall score remains above 6000, but region-3 mIoU falls 0.525235pp
and its boundary F1 falls 3.935776pp, so the late endpoint is not region-uniform.

Relative to 6000, 8000 changes 61,340 confirmation pixels, with 39,366 corrected
and 20,122 harmed (net +19,244). On the 97 available released-unlabeled views it
changes 0.180294% of pixels and adds 42,012 cable pixels; without GT this is only
a distribution-change observation, not evidence of improvement.

The qualitative review at
`outputs/quality/SEM-434/r2/review/unlabeled_long_training/index.html` shows all
301--400 identifiers, including explicit missing panels for the three RGB152
views not registered by COLMAP. It compares frozen RGB, 6000 and 8000 and must
not be read as a scored evaluation.

Independent PNG re-reading reproduced all aggregate, regional, per-view and
boundary counts in 1184 checks. Result hashes: training
`247452d34617a73864928eed3377b6b28df79447040867288979c08a20b81b72`,
inference `86f6971ae7f1de3f059509a6f245d3d7df90c9ea00633c8ce55927001105a505`,
audit `cc531015e16f64c7ebe9a00c3e8e09e630b4dd6b0aa7635a102a3115ea710d1f`.
