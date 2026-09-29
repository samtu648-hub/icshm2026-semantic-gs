# SEM-221-RGBD-DECK-ARBITRATION-OOF — independent deck arbitration

## Status

Completed on 2026-09-07. All twelve calibration thresholds failed; check and
the reused 74-view development set remained unopened.

## Question and boundary

SEM-220's additive-only best candidate slightly improves deck IoU but cannot
reach the recall and IoU gates. This final early-fusion diagnostic tests whether
the independently trained RGB/surface probabilities can also remove false deck
pixels, instead of protecting every SEM-160 deck prediction.

Only the background/deck lane is mutable. Class 2 (`stay_cable`), tower,
foundation, and foundation-rejected background remain exact. The twelve
thresholds, precision/IoU/recall gates and selection rule are fixed before this
evaluation. Calibration is opened first; check remains closed unless a
candidate passes every gate. The reused 74-view development set is unavailable.

## Result

The best calibration deck IoU occurred at threshold 0.70. Relative to SEM-160:

- mIoU -0.17606 pp
- deck IoU -0.76577 pp
- deck precision +0.20642 pp
- deck recall -1.05189 pp
- background-to-deck false-positive ratio 1.1050

It added 133,614 pixels and removed 232,733 existing deck pixels. Fixed panels
show the removals running along long true deck edges, explaining the recall
loss. Result SHA-256 is `3f6789f3...2d2a`. Bidirectional use of SEM-220 is
dropped, and the RGB/surface bridge branch is closed without a final model.
