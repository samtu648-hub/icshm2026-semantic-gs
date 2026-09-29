# SEM-123-BAND-RECOVERY-DEV-SWEEP — selective recovery inside the 0.20–0.30 band

## Status

Completed development sweep on 2026-09-03. The 74 labels are explicitly reused
for development only; this result is not blind or a formal replacement claim.

## Rationale

SEM-122 established that the entire hard-0.20 minus hard-0.30 band is 61.16%
background, while the near-seed ring and higher probability end have a larger
true-cable share. It also established that seed attachment alone is not enough.

SEM-123 tests one changed inference variable: start from protected hard 0.30,
then recover a fixed subset of that band. A subset is the conjunction of a
registered probability floor, maximum 0.30-seed distance and frozen SEM-001
prior mode. The 18 fixed predicates plus unchanged hard-0.30 baseline are
ranked on the explicitly reusable development set with the existing SEM-120
score. Deck, tower and foundation always remain bit-identical to SEM-001.

This is a developer sweep, not blind or formal evidence. It cannot alter RGB,
geometry, checkpoints, model weights or source probabilities. The selected
class map will be materialised only to make panels reproducible; it is not a
new final candidate.

## Command

```bash
python3 scripts/run_sem123_band_recovery_dev_sweep.py \
  --config experiments/runs/SEM-123-BAND-RECOVERY-DEV-SWEEP/config.yaml \
  --split experiments/splits/sem_001_scaffold_static_v1.json \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --sem001-result "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/result.json" \
  --sem001-predictions "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/validation_semantic_png" \
  --sem102-manifest "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1/prediction_manifest.json" \
  --sem102-audit-root "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1" \
  --sem120-result "$ICSHM_OUTPUT_ROOT/r2/sem_120/dev_threshold_sweep/full-20260903-r1/result.json" \
  --sem122-result "$ICSHM_OUTPUT_ROOT/r2/sem_122/uncertainty_band_diag/full-20260903-r1/result.json" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_123/band_recovery_dev_sweep/full-20260903-r2"
```

## Result

All 18 predicates and the hard-0.30 baseline completed. The input-access
manifest was written after hash-checking SEM-001, SEM-102, SEM-120 and SEM-122
but before labels. RGB, checkpoints and model weights were untouched; every
candidate retains deck, tower and foundation prediction pixels exactly.

The registered development-score leader is `p20_d02_any`: recover every
`p >= 0.20` band pixel within 2 px of a hard-0.30 seed. Relative to hard 0.30,
it changes mIoU from `0.771317` to `0.771605` (+0.029 pp), `stay_cable` IoU
from `0.690498` to `0.692107` (+0.161 pp), and recall from `0.835102` to
`0.854184` (+1.908 pp). It recovers 58,136 GT cable pixels but adds 73,598 GT
background pixels, taking background→`stay_cable` from 629,688 to 703,286.

The more conservative Pareto candidate is `p20_d04_sem001_stay_cable`:
recover `p >= 0.20` band pixels within 4 px only where SEM-001 had already
predicted `stay_cable`. It has the full table's highest plain mIoU (`0.771991`)
and cable IoU (`0.693816`), with recall `0.845885` and 658,240 background→
`stay_cable` pixels. It adds 32,854 true-cable versus 28,552 background pixels.

Both outputs are developer-only. They are retained as recall-oriented and
conservative visual candidates for a same-pose comparison; neither replaces
historical SEM-111 or the SEM-001 rollback. An initial `r1` attempt wrote only
its pre-label input manifest before the execution window interrupted it; it
produced no metrics or candidate map and was preserved rather than overwritten.
The complete immutable output is
`$ICSHM_OUTPUT_ROOT/r2/sem_123/band_recovery_dev_sweep/full-20260903-r2` with
`result.json` SHA-256
`75146360d3f5da6a6fd63ece2fcfa86efc404ac9668f6e757e14578cf24c0359`.
