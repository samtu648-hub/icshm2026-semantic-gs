# SEM-160-DECK-BINARY-OOF — native-resolution deck recovery OOF

## Status

Completed on 2026-09-04; the training-only OOF gate passed. Threshold 0.99 and
the entire recipe are frozen for one separately registered SEM-161 final model.
The reused 74-view development set remained unavailable.

## Question

Can a dedicated high-resolution rendered-RGB deck classifier recover the
fragmented deck pixels that the frozen SEM-001 3D sidecar emits as background,
without modifying any accepted semantic prediction?

## Frozen protocol

- Reuse SEM-140's SegFormer-B2 revision, seed, three spatial OOF folds, native
  960 crop, 960/480 sliding inference, weighted binary CE, optimiser and 3,000
  steps per fold. The only learned change is the binary target: non-deck/deck.
- Start from SEM-150's training OOF candidate. A pixel may become deck only if
  it is background both before and after the foundation gate. This prevents
  the deck branch from relabelling a foundation pixel that SEM-150 rejected.
- Existing deck is never deleted. `stay_cable`, tower and foundation remain
  exact. No transition other than background to deck is allowed.
- Compare the registered threshold grid on the 108-view calibration half. A
  candidate needs at least +0.1 pp deck IoU and recall, no mIoU loss, at least
  95% aggregate deck precision, no more than 1.25x background-to-deck errors,
  and exact protected predictions. The selected threshold then faces the same
  independent 111-view check gate.

## Boundary

A check pass permits one separately registered all-219 SEM-161 checkpoint. It
does not permit threshold, crop, loss, seed or checkpoint selection from check
or the reused 74-view development set. Failure stops this branch and leaves
SEM-151 frozen.

## Completed evidence

The 8-step GPU smoke passed with strict reload at 5,570 MiB peak allocation.
Three final-3000 folds then produced 95, 84 and 40 mutually exclusive OOF
probability maps. Training took 1,461.07 seconds in total (0.40585 GPU-hours),
OOF inference took 64.82 seconds, and all three checkpoints reloaded strictly.

On the 108-view calibration half, thresholds 0.97, 0.98, 0.99 and 0.995 passed
all registered gates. Maximum deck IoU selected 0.99. Relative to SEM-150,
deck IoU rose from 0.941511 to 0.945301 (+0.379 pp), recall from 0.980876 to
0.988000 (+0.712 pp), total mIoU by 0.082 pp, and precision remained 0.956281.

With 0.99 frozen, the 111-view check also passed: deck IoU
`0.935087→0.939457` (+0.437 pp), recall `0.977237→0.985647` (+0.841 pp),
total mIoU `0.867385→0.868315` (+0.093 pp), and precision remained 0.952487.
The branch added 94,125 deck pixels: 63,324 true deck, 29,250 background,
1,526 `stay_cable`, 11 tower and 14 foundation ground-truth pixels. All
existing predictions and all foundation-gate rejections remained exact.

The OOF result SHA-256 is
`1b805dba1bd114686825da448e1ac978471562b1d9d5258d9eaa84cfd5020fcb`;
the 219-file selected prediction manifest SHA-256 is
`129d82d6ee46c85632f9bd393a36d7b3f719272d8fa537f93fe2541ff25b3444`.

## Outcome

SEM-160 qualifies the frozen 0.99 recipe for one SEM-161 all-training final
checkpoint and one developer audit. No threshold, model or training parameter
may change in response to check or future development results.
