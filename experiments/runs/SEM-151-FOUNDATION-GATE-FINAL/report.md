# SEM-151-FOUNDATION-GATE-FINAL — frozen developer audit

## Status

Completed on 2026-09-04. The frozen threshold 0.10 passed every developer
quality, retention and engineering gate. This is now the strongest local
developer candidate and is recommended for official hidden evaluation.

## Protocol

Start from SEM-141's already materialised 74 developer predictions. For pixels
already predicted as foundation, retain class 4 only when the frozen SEM-102
final class-4 probability is at least 0.10; otherwise emit background. All
deck, `stay_cable`, and tower prediction masks must remain byte-identical.

All 74 outputs and their manifest must be written before labels are opened.
Relative to SEM-141, the developer gate requires at least +3 pp foundation IoU,
+8 pp precision, at least 90% recall, no global mIoU loss, at least 10%
background-to-foundation reduction, no more than 0.5 pp loss on worst-10% or
140–146, exact immutable classes, RGB identity and the official external-camera
contract.

## Boundary

The validation set is explicitly reused. A pass recommends only official
hidden evaluation and cannot trigger threshold, model or checkpoint changes.

## Results

Relative to the frozen SEM-141 baseline:

| Metric | SEM-141 | SEM-151 | Delta |
|---|---:|---:|---:|
| mIoU with background | 0.774899 | 0.830934 | +5.604 pp |
| mIoU without background | 0.722893 | 0.792639 | +6.975 pp |
| foundation IoU | 0.448718 | 0.727703 | +27.898 pp |
| foundation precision | 0.467405 | 0.782758 | +31.535 pp |
| foundation recall | 0.918191 | 0.911866 | -0.633 pp |
| background to foundation pixels | 156,852 | 37,460 | -76.118% |
| worst-10% mIoU without background | 0.445049 | 0.486655 | +4.161 pp |
| views 140--146 mIoU without background | 0.774075 | 0.791585 | +1.751 pp |

The gate rejected 133,722 existing foundation predictions. Of these, 119,392
were true background and became correct, while 1,058 were true foundation and
became false negatives. Foundation false positives fell from 174,992 to 42,328.
Across the 74 developer views, foundation IoU improved in 70, tied in four
views without effective foundation evidence and regressed in none.

The other protected classes are exactly unchanged from SEM-141:

| Class | IoU | Precision | Recall |
|---|---:|---:|---:|
| deck | 0.900999 | 0.957400 | 0.938629 |
| stay_cable | 0.706983 | 0.842841 | 0.814334 |
| tower | 0.834870 | 0.883822 | 0.937786 |

Background IoU/precision/recall are 0.984114/0.991583/0.992404. The final
foundation IoU/precision/recall are 0.727703/0.782758/0.911866.

## Engineering evidence

- All 74 predictions and `prediction_manifest.json` were materialised before
  the explicitly reused developer labels were opened.
- Deck, `stay_cable` and tower masks are byte-exact relative to SEM-141.
- Frozen parent checkpoint reload and RGB byte-identity checks passed.
- The official `CAMERA_FROM_WORLD + SIMPLE_RADIAL` external-camera path passed:
  RGB `[989,1320,3]`, five logits `[5,989,1320]`, and a class-ID PNG in 0--4.
- Result SHA-256:
  `e1d932db9264adf281f56ec03f62db9d6927f8120bf6220b13c18b85b5ed8d3f`.
- Prediction manifest SHA-256:
  `8443adc70cd7837c6ea83b4fb6f17015701799d55abab575d4f94a17950b33c1`.

## Visual review

Panels 059, 081 and 159 use the fixed order RGB / GT / SEM-141 / SEM-151 /
green-corrected-red-harmed. The gate removes broad purple spill around the
ground, deck and shore while retaining the annotated piers and footings. Green
corrections dominate the outcome overlays and only sparse red removals are
visible. This agrees with the aggregate and per-view measurements; it is not a
substitute for hidden evaluation.

## Decision and rollback

Keep SEM-151 as the strongest local developer candidate and submit it to the
official hidden evaluation without changing threshold, parent models or fusion
rules. Preserve SEM-141 as the immediate rollback, SEM-111 as the lower cable
false-positive candidate, and SEM-001 as the independent static 3D fallback.
