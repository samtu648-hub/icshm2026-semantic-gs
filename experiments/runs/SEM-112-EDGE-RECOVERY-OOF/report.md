# SEM-112-EDGE-RECOVERY-OOF — edge-connected low-confidence cable recovery

## Status

Completed; calibration gate failed. SEM-111 remains the current qualified semantic candidate
and `RGB-030-FULL Scaffold-GS + SEM-001` remains the independent rollback.
SEM-112 is training-only and has no fixed-74-view, unlabelled, real-RGB or
external-camera interface.

## Hypothesis

The fixed 0.80 SEM-110/111 threshold is deliberately conservative, so it may
leave some real thin cable pixels as background. A recovery pixel is permitted
only if three independent, frozen signals agree: SEM-001 already predicted
`stay_cable`; the SEM-101 spatial OOF probability lies in `[0.55, 0.80)`; and
the same-camera frozen RGB contains a strong luminance edge that is within two
pixels of a high-confidence (`>=0.80`) rendered-domain cable seed. This can
only add class-2 pixels in the mutable lane; deck/tower/foundation stay
bit-identical.

## Single change and pre-registered selection

The sole changed inference variable is the rendered-RGB edge-support
quantile, selected only from `0.80/0.85/0.90`. The high threshold 0.80, low
threshold 0.55, two-pixel adjacency, base class-2 requirement, source models,
hash partition and all geometry are fixed. Calibration selects the candidate
with the highest `stay_cable` IoU only when it also improves mIoU by at least
0.2pp, cable IoU by 0.2pp and recall by 0.5pp versus frozen SEM-110, grows
background→`stay_cable` by at most 10%, and still removes at least 80% of the
raw SEM-001 background→`stay_cable` errors. Ties favour higher recall, then
fewer false positives and fewer restored pixels.

No calibration pass means check labels and 74-view remain closed. A failed
check stops SEM-112. A passed check can only justify a separately registered,
fixed-rule final audit; it cannot alter SEM-111 or reopen its 74-view result.

## Inputs and boundaries

- 219 authorised training labels only, partitioned by the existing deterministic
  hash into 108 calibration and 111 check views.
- Hash-validated SEM-001 training class maps, SEM-101 spatial OOF probabilities,
  SEM-101 same-camera frozen rendered RGB, and reconstructed SEM-110 baseline.
- No model fitting, RGB reference imagery, unlabelled image, hidden input,
  actual validation image, or external camera.

## Planned command

```bash
python3 scripts/run_sem112_edge_recovery_oof.py \
  --config experiments/runs/SEM-112-EDGE-RECOVERY-OOF/config.yaml \
  --split experiments/splits/sem_001_scaffold_static_v1.json \
  --oof-split experiments/splits/sem_005_oof_spatial_v1.json \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --sem001-training-root "$ICSHM_OUTPUT_ROOT/r2/sem_110/sem001_training_predictions/full-20260902-r1" \
  --sem101-oof-root "$ICSHM_OUTPUT_ROOT/r2/sem_101/rendered_segformer_oof/full-20260902-r3" \
  --rendered-rgb-root "$ICSHM_OUTPUT_ROOT/r2/sem_101/rendered_training_rgb/full-20260902-r3" \
  --sem110-root "$ICSHM_OUTPUT_ROOT/r2/sem_110/protected_fusion_oof/full-20260902-r1" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_112/edge_recovery_oof/full-20260902-r1"
```

## Result

All frozen source manifests and all 219 hash-addressed files were validated
before any label opened. The training partition remains 108 calibration and 111
check views. No candidate passed calibration, so the check half, fixed 74-view,
unlabelled images, real RGB and external-camera interface all remained closed.

The strongest pre-registered candidate was edge quantile 0.90. It restored
122,598 calibration pixels and kept deck/tower/foundation exactly unchanged.
It raised `stay_cable` recall from 0.927498 to 0.933866 (+0.64pp) while keeping
background→`stay_cable` growth to 7.57% (674,104 to 725,132) and retaining a
95.35% reduction against raw SEM-001. But its mIoU gain was only 0.0396pp
(0.807648 to 0.808044) and cable-IoU gain only 0.1827pp (0.871964 to
0.873791), both below the pre-registered +0.20pp optimisation requirement.

The 0.80 and 0.85 edge quantiles were weaker and also exceeded the 10% maximum
background→`stay_cable` growth (11.94% and 10.15%). Therefore no edge quantile
could be selected; no prediction panels were generated. The result SHA-256 is
`dfbf23d35da79edeeb2370ec6201856877cc16676b283ffcc02aa46c66c1ed6c`.

## Decision

- keep_or_drop: drop_calibration_gate_failed
- fallback: `RGB-030-FULL Scaffold-GS + SEM-001`

The rule is frozen as a negative result. Do not lower its gates, alter the
0.55/0.80 thresholds or 2px adjacency, add edge quantiles, or use training
check/74-view to choose a rescue rule. SEM-111 remains the current candidate.
