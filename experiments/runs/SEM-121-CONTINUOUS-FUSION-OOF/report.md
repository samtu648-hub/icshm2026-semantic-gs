# SEM-121-CONTINUOUS-FUSION-OOF — OOF-calibrated continuous cable fusion

## Status

Completed on 2026-09-03. The independent 111-view training check failed the
pre-registered dominance gate, so the 74-view developer set was not opened.

SEM-121 follows the SEM-120/VIS-004 conclusion that further hard-threshold
sweeping cannot resolve the cable-recall versus yellow-background trade-off.
It keeps the frozen protected fusion sources, but replaces the hard teacher
threshold with one fixed-form, three-coefficient logistic function. Its two
inputs are the frozen SEM-101 OOF teacher's cable-versus-background log odds
and whether frozen SEM-001 already predicted `stay_cable`.

The function is not a new neural model, does not train RGB or semantics, and
cannot change SEM-001 deck, tower or foundation pixels. The coefficients are
fitted once from deterministic, balanced samples in the existing training
calibration half. There is no threshold grid, probability-band selection,
check refit or development refit.

## Flow

1. Hash-check the frozen training SEM-001 maps, SEM-101 OOF probabilities,
   SEM-001 result and SEM-120 result before labels.
2. Fit the one registered calibrator on 108 calibration views.
3. Compare it once on 111 check views to hard fusion 0.30 and 0.20.
4. Only if it dominates 0.30 in mIoU, cable IoU, cable recall, background
   errors and protected exactness, write and score the 74-view development
   candidate. This score remains developer-only.

## Command

```bash
python3 scripts/run_sem121_continuous_fusion_oof.py \
  --config experiments/runs/SEM-121-CONTINUOUS-FUSION-OOF/config.yaml \
  --split experiments/splits/sem_001_scaffold_static_v1.json \
  --oof-split experiments/splits/sem_005_oof_spatial_v1.json \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --sem001-training-root "$ICSHM_OUTPUT_ROOT/r2/sem_110/sem001_training_predictions/full-20260902-r1" \
  --sem101-oof-root "$ICSHM_OUTPUT_ROOT/r2/sem_101/rendered_segformer_oof/full-20260902-r3" \
  --sem101-oof-summary "$ICSHM_OUTPUT_ROOT/r2/sem_101/rendered_segformer_oof_summary/full-20260902-r3/result.json" \
  --sem110-result "$ICSHM_OUTPUT_ROOT/r2/sem_110/protected_fusion_oof/full-20260902-r1/result.json" \
  --sem001-result "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/result.json" \
  --sem001-validation-predictions "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/validation_semantic_png" \
  --sem102-validation-manifest "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1/prediction_manifest.json" \
  --sem102-validation-root "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1" \
  --sem120-result "$ICSHM_OUTPUT_ROOT/r2/sem_120/dev_threshold_sweep/full-20260903-r1/result.json" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_121/continuous_fusion_oof/full-20260903-r1"
```

## Result

The immutable input-access manifest confirms that only the 219 labelled
training views were used: 108 calibration and 111 check. No 74-view labels,
real RGB, unlabelled input, hidden input, or external-camera mode was opened.

The calibration fit used 221,184 mutable `stay_cable` and 221,184 mutable
non-cable samples. It converged in nine Newton steps with coefficients
`intercept=1.018158`, `teacher_log_odds=0.600878`, and
`SEM-001_cable_prior=0.544541` (balanced-sample log loss `0.081903`).

On the independent 111-view check, the frozen hard 0.30 baseline obtained
mIoU `0.790574`, `stay_cable` IoU/recall `0.798087/0.970561`, and
2,812,695 background→`stay_cable` pixels. The continuous candidate obtained
mIoU `0.778500`, `stay_cable` IoU/recall `0.746121/0.975556`, and 3,948,987
background→`stay_cable` pixels. Thus it recovered 0.50 pp recall, but lost
1.21 pp mIoU and 5.20 pp cable IoU while adding 1,136,292 background false
positives. Deck, tower and foundation prediction pixels remained exact.

Four of five check comparisons failed, so this formula is dropped without
retuning its L2, sampling, coefficients, features, or decision rule. The
Git-ignored result is
`$ICSHM_OUTPUT_ROOT/r2/sem_121/continuous_fusion_oof/full-20260903-r1/result.json`
with SHA-256 `31da5c37bc94ff801fbc71eff65bdd9aa71ff44afa0ebd545a486af225f1af07`.
