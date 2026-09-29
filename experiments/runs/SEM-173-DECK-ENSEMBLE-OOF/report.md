# SEM-173-DECK-ENSEMBLE-OOF — frozen deck-probability ensemble

## Status

Completed on 2026-09-06. All 36 calibration pairs failed, so the 111-view
check and reused 74-view development labels remained unopened.

## Question and fixed rule

SEM-172 is slightly more precise but less complete than SEM-160. Test whether
a convex probability blend reduces their independent ranking errors. No model
is trained: blend `alpha * p160 + (1-alpha) * p172` for the three registered
SEM-160 weights, then apply SEM-160's unchanged SEM-150 background-only fusion.

The 108-view calibration half selects one registered alpha/threshold pair.
Relative to SEM-160, calibration and the one-shot 111-view check require
non-negative mIoU and deck recall, at least +0.05 pp deck IoU, at least 95%
deck precision, no more than 1.10 times background-to-deck false positives,
and exact protection.

The 74-view development set, external camera, real RGB, unlabelled and hidden
inputs remain unavailable. Failure ends this ensemble and retains SEM-161;
passing only permits a separately registered full-data counterpart.

## Result

The maximum calibration deck IoU used SEM-160 weight 0.75 and threshold 0.99.
Relative to SEM-160 it changed:

- mIoU: `0.86879012 -> 0.86878857` (-0.00015 pp)
- deck IoU: `0.94530132 -> 0.94533346` (+0.00321 pp)
- deck precision: `0.95628069 -> 0.95677835` (+0.04977 pp)
- deck recall: `0.98800008 -> 0.98750449` (-0.04956 pp)
- background-to-deck false positives: 0.99205 times SEM-160

The IoU change is far below the registered +0.05 pp gate and is purchased with
lower recall and slightly lower mIoU. Protection remained exact.

Result SHA-256:
`be15a0e3b2edb4fbee556ffff89ff393ecae54e64679b14dced32c4c4cf48019`.

## Conclusion

Drop the ensemble and end the current deck post-processing/same-architecture
loss series. Bidirectional deletion, morphological closing, fixed interior
weighting and frozen probability ensembling all failed before independent
check. SEM-161 remains the strongest qualified developer candidate. Any new
deck effort must change the underlying rendered evidence or representation and
receive a new protocol; alpha, threshold, morphology and loss scans are closed.
