# SEM-150-FOUNDATION-GATE-OOF — precision-first foundation verification

## Status

Completed on 2026-09-04. The training-only OOF gate passed and froze threshold
0.10 before any 74-view access.

## Question

Can the already frozen SEM-101 spatial-OOF foundation probability remove
SEM-141's excessive foundation-to-background spill without retraining a model
or modifying deck, `stay_cable`, or tower predictions?

## Frozen rule

The training baseline is SEM-001 fused with SEM-140's frozen OOF cable branch
at threshold 0.70. Only pixels already predicted as class 4 are candidates.
They remain foundation when SEM-101's class-4 probability meets a threshold;
otherwise they become background. The rule cannot add foundation or emit any
other class.

Thresholds 0.1 through 0.9 at intervals of 0.1 are evaluated on the existing
label-independent 108-view calibration half. A candidate must improve
foundation IoU by at least 5 pp, precision by at least 10 pp, retain at least
95% foundation recall, not reduce global mIoU, remove at least 10% of
background-to-foundation errors, and preserve deck/`stay_cable`/tower exactly.
The best foundation IoU among passers is frozen before the 111-view check.

## Boundary

No 74-view, hidden, real RGB, unlabelled input, external camera or model
training is available. Check failure stops this mechanism; check success only
permits a separately registered final developer audit. SEM-141 remains intact.

## Result

Threshold 0.10 was the only calibration candidate satisfying the registered
95% recall floor. On 108-view calibration, foundation IoU improved from
0.389989 to 0.655965 (+26.60 pp), precision from 0.391499 to 0.676357
(+28.49 pp), recall remained 0.956056, global mIoU improved 5.49 pp and
background-to-foundation errors fell 70.76%.

After freezing 0.10, the independent 111-view check reproduced the gain:
foundation IoU 0.396854 to 0.654920 (+25.81 pp), precision 0.398136 to
0.661045 (+26.29 pp), recall 0.991956 to 0.986049, global mIoU +5.35 pp and
background-to-foundation errors -66.84%. The gate rejected 1,175,575 existing
foundation pixels, of which 1,159,570 were true background and only 6,945 were
true foundation. Deck, `stay_cable`, and tower predictions remained exact.

The result SHA-256 is
`e29843e21f0f9097b6c78c6ccf3fa0d59b10605ca9abbd8cbaf5320984d991a0`.
SEM-151 is the only authorised final developer conversion.
