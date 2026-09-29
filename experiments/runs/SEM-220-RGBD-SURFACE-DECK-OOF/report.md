# SEM-220-RGBD-SURFACE-DECK-OOF — early RGB/surface fusion

## Status

Completed across 2026-09-06/07. No calibration threshold passed; check and the
reused 74-view development set remained unopened.

## Question and single change

SEM-200/201/210/211 show that frozen geometry is not precise enough to recover
deck by a late hard decision, even when intersected with SEM-160 appearance.
SEM-220 instead supplies geometry while the high-resolution binary model learns
the deck boundary. The only primary change from SEM-160 is four label-free input
channels: robust log depth and the camera-space XYZ normal of the frozen
RGB-030-FULL visible surface.

The pretrained RGB patch weights are copied exactly and the four new channel
weights start at zero. The SegFormer-B2 architecture, binary target, class
weights, 960 native crop, augmentation, optimizer, seed, 3,000 iterations per
fold, spatial OOF split and final-checkpoint rule remain matched to SEM-160.

All 219 OOF probabilities are produced before calibration. The model may only
add deck where SEM-160 and both foundation provenance maps remain background;
all existing deck, `stay_cable`, tower and foundation pixels remain exact.
Check opens only after a calibration pass, and the reused 74-view development
set remains closed.

## Result

All 219 four-channel geometry tensors were built without labels and sealed.
Three matched 3,000-step folds trained for 516.95/504.86/512.95 seconds
(0.42632 GPU hours total), peaked at 5,606 MiB and strictly reloaded their final
checkpoints. All 219 OOF probability maps were present before calibration.

The best threshold was again 0.99. Relative to frozen SEM-160 it added 8,497
pixels at 58.997% true-deck precision and changed:

- mIoU +0.00502 pp
- deck IoU +0.02190 pp
- deck recall +0.06677 pp
- final deck precision 95.588%

The direction is slightly positive but far below the registered +0.1/+0.2 pp
IoU/recall gates, so it is not a promotable improvement. Geometry feature,
OOF probability, prediction and result hashes are `1846ae41...c18`,
`62f052df...b73`, `cf33e41a...e04` and `b8603bc3...d42`.
