# SEM-140-CABLE-BINARY-OOF — native-resolution cable binary OOF

## Status

Completed on 2026-09-04; the training-only OOF gate passed. This experiment
did not read the reused 74-view development labels and cannot replace
`SEM-111` or the independent `SEM-001` fallback by itself. Its passing result
freezes threshold `0.70` and permits one separately registered SEM-141 final
checkpoint.

## Question

Does removing the deck/tower/foundation classification burden from the learned
render-domain branch, while increasing its native-pixel crop from 768 to 960,
improve `stay_cable` separation? At fusion time the learned branch may emit
only background or `stay_cable`; every SEM-001 deck, tower and foundation
prediction is copied byte-for-byte.

## Frozen protocol

- Reuse the exact 219 frozen Scaffold RGB renderings, labels, spatial OOF folds,
  pretrained SegFormer-B2 revision, seed, optimiser, 3,000-step budget and
  label-independent SEM-110 calibration/check partition.
- Reinitialize a two-class head for `non_stay_cable` and `stay_cable`. Train on
  native 960-pixel crops without resizing, using only weighted two-class cross
  entropy and horizontal flips.
- Produce exactly one spatially out-of-fold cable probability for every
  training view. Compare the fixed 0.05–0.95 threshold grid on the 108-view
  calibration half only.
- A threshold must match or improve SEM-110 calibration mIoU, cable IoU and
  recall while staying below 1.25× its background false positives. Among
  passers, use the already documented SEM-120 score; check labels never select
  another threshold.
- On the 111-view check half, require no mIoU or recall loss, at least +0.5 pp
  cable IoU, at most 1.25× background false positives, and exact protected
  pixels. Only a passing result permits a separately registered SEM-141 final
  model.

## Decision boundary

Any OOF failure is recorded and stops this recipe before final training or
74-view developer evaluation. A pass freezes the training recipe and threshold
for one separately registered all-219 final checkpoint. No threshold, loss,
crop, seed or checkpoint may be chosen from check or 74-view results.

## Completed evidence

The 8-step GPU smoke passed at 5,570 MiB peak allocated memory, with strict
checkpoint reload. The three fixed folds then completed 3,000 steps each and
produced 95, 84 and 40 mutually exclusive OOF probability maps. Training took
493.95, 489.45 and 490.21 seconds (0.4093 GPU-hours total); OOF inference took
28.00, 24.85 and 11.56 seconds. All three checkpoints reloaded, all inputs were
frozen rendered RGB, and fixed validation/unlabelled/real RGB remained closed.

On the 108-view calibration half, thresholds 0.55 through 0.85 passed the
registered baseline constraints. The unchanged developer score selected 0.70:
mIoU `0.807648 → 0.813038`, cable IoU `0.871964 → 0.896361`, cable recall
`0.927498 → 0.949307`, and background→cable pixels `674,104 → 609,942`.

After freezing 0.70, the 111-view check also passed: mIoU
`0.807461 → 0.813876` (+0.641 pp), cable IoU `0.871190 → 0.899532`
(+2.834 pp), recall `0.945650 → 0.959316` (+1.367 pp), precision
`0.917110 → 0.935209`, and background→cable pixels
`1,113,641 → 850,139` (-23.66%). SEM-001 deck/tower/foundation predictions
were exactly unchanged. The summary result SHA-256 is
`6ef7394c5e258e052004d59968a6f29ea850b402f588f853364efaf5f56674d1`.

## Outcome

The OOF gate passed and SEM-141 may train exactly one all-219 final checkpoint
with the same two-class recipe and threshold 0.70. No OOF result authorises a
new threshold, crop, loss, seed or checkpoint choice.
