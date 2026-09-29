# SEM-211-APPEARANCE-XYZ-JOINT-OOF — joint evidence gate

## Status

Completed on 2026-09-06. None of the 30 registered pairs passed calibration;
check and the reused 74-view development set remained unopened.

## Question and protocol

SEM-160 appearance alone has useful recall but its below-0.99 additions are not
precise enough. SEM-210 XYZ alone covers the bridge region but also activates
nearby background surfaces. SEM-211 tests their conjunction without retraining
either parent: an eligible background pixel must pass both registered frozen
probability thresholds before it can become deck.

Thirty threshold pairs are fixed in advance. All source probabilities are
spatial OOF. For each calibration view every pair is constructed before its
label is opened. Selection may use only the deterministic 108-view calibration
partition. The 111-view check opens once only if a pair passes every gate; the
reused 74-view development set remains closed. Existing deck, `stay_cable`,
tower, foundation and foundation-rejected pixels remain exact.

## Result

The diagnostic best pair was appearance 0.98 and XYZ 0.90. It added 5,061
pixels at 48.82% true-deck precision. Relative to SEM-160, mIoU changed by
+0.00037 pp, deck IoU by +0.00029 pp, recall by +0.03291 pp and precision by
-0.03052 pp. The candidate is effectively neutral and far below the
+0.1/+0.2 pp IoU/recall gates.

Prediction/result SHA-256 are `626ce0a3...f95` and `e38697e8...a06`. Late
intersection cannot make either parent's errors sufficiently independent;
close this post-hoc geometry recovery family.
