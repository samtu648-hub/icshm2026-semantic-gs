# SEM-111-PROTECTED-FUSION-FINAL — fixed protected fusion audit

## Status

Completed; formal retention gate passed. SEM-110 selected `stay_cable` threshold 0.80 using only
its deterministic training calibration half, then passed its independent
training check half. SEM-111 applies exactly that already-frozen decision rule
to the fixed 74 validation predictions: no model is trained, no threshold is
selected and deck/tower/foundation predictions remain the frozen SEM-001 map.

## Access protocol

The runner first validates the SEM-110 result hash, SEM-001 result hash and
SEM-102 prediction manifest hash. It creates all 74 fused predictions and a
manifest before opening fixed labels once. It then produces a separate official
`CAMERA_FROM_WORLD + SIMPLE_RADIAL` external output from frozen semantic arrays
only. Neither stage reads target RGB or labels through the external interface.

## Formal gate

Relative to SEM-001, global mIoU, `stay_cable` IoU and recall must not decline;
deck/tower/foundation must be exactly unchanged; background→`stay_cable` must
decrease by at least 1%; worst 10% and 140–146 mIoU may each fall at most
0.5pp. RGB bytes must remain identical and the external output must be a
1320×989 RGB, float32 five-logit and uint8 0–4 semantic triplet. Failure keeps
`RGB-030-FULL Scaffold-GS + SEM-001` as the only fallback.

## Executed result

All 74 fused predictions were written and hash-manifested before validation
labels were opened. The prediction-manifest SHA-256 is
`c1a8756963aa3ad3389aab503fcc567e453c21a4584247d38941c09115cb3917`; it
records zero real-RGB or unlabelled input, plus exact preservation of all
protected SEM-001 prediction pixels. The final result SHA-256 is
`45d2aa9095815a086a8927c7233bd585f6bb2ebcfe9a1c8b936c3d57915757d9`.

| Metric (74 fixed views) | SEM-001 | SEM-111 | Result |
|---|---:|---:|---|
| mIoU with background | 0.616103 | 0.760635 | +14.45 pp |
| `stay_cable` IoU / recall | 0.068759 / 0.368036 | 0.638137 / 0.711445 | +56.94 / +34.34 pp |
| background→`stay_cable` pixels | 13,166,331 | 347,412 | −97.36% |
| deck IoU | 0.900999 | 0.900999 | exactly unchanged |
| tower IoU | 0.834870 | 0.834870 | exactly unchanged |
| foundation IoU | 0.448718 | 0.448718 | exactly unchanged |
| worst 10% mIoU without background | 0.423440 | 0.428629 | +0.52 pp |
| views 140–146 mIoU without background | 0.601213 | 0.758237 | +15.70 pp |

Every metric retention check passed. The official external output also passed:
it preserves the source frozen RGB byte-for-byte, accepts
`CAMERA_FROM_WORLD + SIMPLE_RADIAL`, emits 1320×989 RGB, float32 five-channel
logits and uint8 0–4 semantic PNG, and retains a valid probability simplex.
It never reads reference RGB or labels. Representative GT / frozen SEM-001 /
fused / error panels are available for `054.png`, `140.png`, and `183.png`.

## Decision

`SEM-111` is the new qualified semantic candidate. Its deployment composition
is frozen `RGB-030-FULL Scaffold-GS`, frozen SEM-001 protected-class map, and
the frozen SEM-102 rendered-RGB probability stream under the fixed 0.80 fusion
rule. `RGB-030-FULL Scaffold-GS + SEM-001` remains the reversible rollback.
No further threshold or model selection is authorized by this result.
