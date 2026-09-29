# SEM-201-MV-DECK-INTERIOR-GEOMETRY-DIAG — source-interior geometry gate

## Status

Completed on 2026-09-06. The calibration gate failed; check and the reused
74-view development set remained unopened.

## Rationale and single change

SEM-200 recovered real deck but its long-baseline projection expanded labelled
source silhouettes into target background: only 34.14% of its 10,035 eligible
calibration additions were true deck. SEM-201 changes only the source surface
definition. A fixed 8-pixel elliptical erosion, reused from the earlier
pre-registered deck-interior scale, removes uncertain source silhouettes before
3D backprojection.

The sealed SEM-200 RGB depth maps, spatial folds, three-view source
confirmation, eight source cameras, 5% depth tolerance, two-vote target rule,
2-pixel splat, foundation-safe eligibility and gates remain unchanged. No vote
or erosion grid is searched. Check and the reused 74-view set remain closed
unless the single calibration candidate passes.

## Result

The 8-pixel source interior removed most boundary drift and raised addition
precision from SEM-200's 34.14% to 56.25%, but coverage collapsed. Only 464
pixels were added: 261 true deck, 120 background, 82 `stay_cable`, one tower
and no foundation.

Relative to SEM-160, mIoU changed by +0.00021 pp, deck IoU by +0.00088 pp and
deck recall by +0.00348 pp. These are far below the registered +0.1/+0.2 pp
IoU/recall gates. Result SHA-256 is `a20084f3...f8c4`. The result proves that an
interior 3D signal exists, but sparse projection cannot cover the missing deck.
