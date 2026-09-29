# SEM-435-RGB154-MASK2FORMER-LONG12K

Status: completed on 2026-09-22 as the fixed v2 development replication; not a
blind or hidden-test result. The authoritative attempt is `r2`; the prepare-only
`r1` directory is preserved from before the narrow historical-equality correction.

## Execution

The recipe is identical to SEM434 regardless of its v1 result: fresh ADE Swin-L,
seed 20260805, original loss/augmentation/AdamW/AMP, original poly schedule through
6000, then fixed per-group LR `1e-5` / backbone `1e-6` through 12000. The new run's
own step 6000 is the paired control because historical SEM412 is model-only and
not a resumable or bitwise trajectory anchor.

Training completed 12000 successful updates in 3210.523 seconds with 9 AMP skips.
Peak allocated/reserved memory was 8084.516/8564 MiB. The full actual trace and
its historical 6000-sample prefix match, all endpoint/resume files are finite, and
RGB hashes are unchanged. Four-endpoint inference took 830.861 seconds with 1968
MiB peak reserved. SEM435 uses 8.7 GiB including the preserved prepare-only r1.

## Full v2 curve (%)

| Step | Train mIoU | Confirm mIoU | Train FG | Confirm FG | mIoU gap | FG gap | Cable boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6000 | 95.432715 | 90.039723 | 94.435926 | 87.715142 | 5.392992 | 6.720784 | 75.770647 |
| 8000 | 95.494444 | 89.906581 | 94.508104 | 87.551317 | 5.587862 | 6.956788 | 75.977734 |
| 10000 | 95.647082 | 89.970813 | 94.696193 | 87.635594 | 5.676269 | 7.060599 | 75.588990 |
| 12000 | 95.716999 | 90.352399 | 94.779562 | 88.103798 | 5.364601 | 6.675764 | 76.609421 |

| Step | Cable IoU | Precision | Recall | FP | FN | Deck IoU | Tower IoU | Foundation IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6000 | 84.066139 | 90.236689 | 92.477604 | 129,866 | 97,634 | 95.421114 | 89.205265 | 82.168051 |
| 8000 | 83.925203 | 90.539639 | 91.992209 | 124,757 | 103,934 | 95.373269 | 89.248084 | 81.658710 |
| 10000 | 83.571270 | 89.857933 | 92.275125 | 135,176 | 100,262 | 95.430715 | 89.225737 | 82.314656 |
| 12000 | 84.421645 | 90.532199 | 92.596796 | 125,686 | 96,087 | 95.497017 | 89.819183 | 82.677346 |

## Regions and endpoint behavior

| Step | Region 1 mIoU / boundary | Region 2 mIoU / boundary | Region 3 mIoU / boundary | Worst view FG mIoU |
|---:|---:|---:|---:|---:|
| 6000 | 79.914428 / 72.611017 | 92.930248 / 89.190682 | 89.130737 / 69.139502 | 057.png: 52.399710 |
| 8000 | 79.169888 / 73.395263 | 93.088169 / 90.907099 | 88.957312 / 67.997670 | 057.png: 52.629239 |
| 10000 | 79.398367 / 73.422673 | 93.331200 / 90.380426 | 88.924202 / 67.291553 | 057.png: 51.878348 |
| 12000 | 80.545895 / 74.249121 | 93.234999 / 90.265990 | 89.375544 / 69.253117 | 057.png: 54.603131 |

Training mIoU again rises monotonically. Unlike v1, confirmation and the gaps
worsen at 8000/10000 before all principal metrics reach their best value at 12000.
Thus v2 has a transient overfitting signal at 8000--10000 followed by late recovery,
not the v1 peak-at-8000 shape. At 12000, cable FP decreases by 4,180 and FN by
1,547 relative to 6000; the larger absolute change is reduced false positives,
with both precision and recall improving. Deck, tower and foundation all improve,
all three regions improve in mIoU, and the worst view improves 2.203421pp FG mIoU.

Relative to 6000, 12000 changes 47,741 confirmation pixels, with 25,976 corrected
and 18,321 harmed (net +7,655). On the 94 available released-unlabeled views it
changes 0.176642% of pixels and removes 4,982 cable pixels. Those views lack GT;
the change direction cannot be called a quality gain.

The shared qualitative review at
`outputs/quality/SEM-434/r2/review/unlabeled_long_training/index.html` shows all
301--400 identifiers, including explicit missing panels for the six RGB154 views
not registered by COLMAP. It compares frozen RGB, 6000 and 12000 and contains no
GT-derived quality claim.

Independent PNG re-reading reproduced 1180 aggregate, regional, per-view and
boundary checks. Result hashes: training
`ad25075736d94a85949cbd55fa163d395178f4f540d8360b93a8ac990c2ac566`,
inference `f66119dd06f6fb9747d9594fa9711e32fd73858beff9821fc9055f7b0b0a5d00`,
audit `b4d7ad7ceef9bfbbacf2b0bf41bac6e39ceb81e76b6a422c2f91fb3d4bb7e787`.
