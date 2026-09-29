# SEM-170-DECK-ARBITRATION-OOF — frozen deck-lane arbitration

## Status

Completed on 2026-09-06. All 12 frozen calibration thresholds failed, so the
111-view check and reused 74-view development labels remained unopened.

## Question

SEM-161 can only add deck, so it cannot remove existing blue tails or let the
dedicated binary model repair both sides of a ragged background/deck boundary.
Can the already frozen spatial-OOF deck probabilities safely arbitrate both
directions inside the class-0/class-1 lane?

## Frozen protocol

No model is trained and no image is rendered. Reuse the exact SEM-160 OOF
probabilities and SEM-150 training baseline. Existing class 2 (`stay_cable`),
class 3 (tower), class 4 (foundation), and foundation pixels previously
rejected to background remain exact. Other class-0/class-1 pixels become deck
when probability is at least the tested threshold and background otherwise.

The existing 108-view calibration half selects one value from the frozen grid;
the independent 111-view check cannot alter it. Relative to SEM-160's selected
0.99 additive result, both halves require non-negative mIoU and recall changes,
at least +0.2 pp deck IoU and precision, no increase in background-to-deck
pixels, and exact protection.

## Boundary

This is training-only OOF qualification. The reused 74-view development set,
external camera, unlabelled images, hidden inputs and real RGB are unavailable.
Failure stops bidirectional arbitration. Passing permits only a separately
registered final application of the same frozen threshold and SEM-161
checkpoint; it does not modify SEM-161.

## Result

The best calibration deck IoU occurred at threshold 0.70, but it was already
strictly worse than SEM-160's frozen additive 0.99 baseline:

- mIoU: `0.868790 -> 0.866657` (-0.213 pp)
- deck IoU: `0.945301 -> 0.935974` (-0.933 pp)
- deck precision: `0.956281 -> 0.957613` (+0.133 pp; below the +0.2 pp gate)
- deck recall: `0.988000 -> 0.976426` (-1.157 pp)
- background-to-deck false-positive ratio: `1.16010`

The candidate added 220,135 deck pixels but also deleted 242,977 existing deck
pixels. The deletions included 124,445 true-deck pixels, directly confirming
the preregistered recall risk. Class 2/3/4 and foundation-rejected predictions
remained exact.

Result SHA-256:
`d377c2540978687691ca1467781fe5210dc6a1fed84e84cf7367f1dfe3824bbb`.

## Conclusion

Drop bidirectional arbitration. The frozen binary probabilities remain useful
as an additive complement, but are not calibrated to veto existing 3D deck
predictions. Further bridge-deck work may preserve every SEM-160 deck pixel and
test only spatially constrained recovery of internal gaps; the deletion route
is closed and SEM-161 remains frozen.
