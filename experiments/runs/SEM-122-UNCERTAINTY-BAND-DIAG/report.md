# SEM-122-UNCERTAINTY-BAND-DIAG — read-only 0.20–0.30 uncertainty-band analysis

## Status

Completed on 2026-09-03 as a developer-only read-only diagnostic.

## Question

`SEM-120` shows that hard 0.20 recovers more `stay_cable` pixels than 0.30,
but `VIS-004` shows part of those additions as broad yellow background. The
failed SEM-121 continuous prior confirms that merely mixing the same two
signals cannot resolve this. Before another inference change, SEM-122 measures
what the exact difference band contains.

The band is fixed as every mutable SEM-001 pixel with frozen teacher
`0.20 <= p(stay_cable) < 0.30`. The runner proves that it equals the
`stay_cable` pixels of hard 0.20 minus hard 0.30, then reports its ground-truth
class mixture, frozen SEM-001 class mixture, fixed probability bins,
8-connected attachment to hard-0.30 seeds, and distance to those seeds.

No class map, probability, RGB output, checkpoint, geometry, model parameter,
or inference rule is changed. This is a developer-only diagnostic on the
explicitly reusable 74 views; it does not select a candidate or support a
formal-replacement claim.

## Command

```bash
python3 scripts/run_sem122_uncertainty_band_diag.py \
  --config experiments/runs/SEM-122-UNCERTAINTY-BAND-DIAG/config.yaml \
  --split experiments/splits/sem_001_scaffold_static_v1.json \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --sem001-result "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/result.json" \
  --sem001-predictions "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/validation_semantic_png" \
  --sem102-manifest "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1/prediction_manifest.json" \
  --sem102-audit-root "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1" \
  --sem120-result "$ICSHM_OUTPUT_ROOT/r2/sem_120/dev_threshold_sweep/full-20260903-r1/result.json" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_122/uncertainty_band_diag/full-20260903-r1"
```

## Result

The input-access manifest was written after frozen-source hash validation and
before labels. It confirms no real RGB, unlabelled input, hidden input or
external-camera mode was opened; no RGB or prediction was modified.

The band is bit-identical to hard-0.20 `stay_cable` minus hard-0.30
`stay_cable`. Across 74 developer views it contains 186,758 pixels: 70,276
(37.63%) are GT `stay_cable`, 114,218 (61.16%) are GT background, and 2,264
(1.21%) are another GT class. Thus an unconstrained return to 0.20 recovers
real cables but adds a background-majority region.

Probability is a weak monotonic ranking: the true-cable fraction rises from
34.61% in `[0.20, 0.22)` to 41.20% in `[0.28, 0.30]`, still below a majority.
Attachment is not discriminative: 180,181 (96.48%) band pixels touch a hard
0.30 seed, but only 37.96% of those pixels are true cable. Distance is more
informative but still insufficient alone: the 1–2 px ring is 44.95% true cable,
falling to 29.04% at 2–4 px and 20.29% at 4–8 px.

The three panels visibly confirm the aggregate: green is recovered true cable,
red is background expansion, orange is another GT class, yellow is the frozen
hard-0.30 seed and blue is the candidate band. The result establishes that
plain seed connectivity cannot safely clean the yellow region; a later
developer sweep may test the fixed joint spatial predicate, but must be a new
experiment rather than a post-hoc modification of SEM-122.

The Git-ignored output root is
`$ICSHM_OUTPUT_ROOT/r2/sem_122/uncertainty_band_diag/full-20260903-r1`;
`result.json` SHA-256 is
`a89eda528810a0c746bf9699e3e5de48277b3b715960c001a04bb2fb706ec0a3`.
