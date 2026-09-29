# SEM-120-DEV-THRESHOLD-SWEEP — protected-fusion developer threshold sweep

## Status

Completed development sweep on 2026-09-03. This is an explicit development
experiment authorised by the user after SEM-111/112. The 74 labelled views are
no longer represented as a blind selection set for SEM-120: they are a reusable
development set. SEM-111 retains its historical formal-gate record and SEM-001
retains its independent rollback role.

## Scope

SEM-120 changes only the existing protected-fusion `stay_cable` threshold. It
compares `0.10/0.20/0.30/0.40/0.50/0.60/0.65/0.70/0.75/0.80/0.85/0.90` on all
74 development views. Deck, tower and foundation stay bit-identical to SEM-001
at every threshold; RGB, geometry, opacity, checkpoints and model weights do
not change.

## Development ranking

The displayed leader maximises:

`0.55 × mIoU + 0.30 × stay_cable IoU + 0.15 × stay_cable recall`.

Ties favour fewer background→`stay_cable` pixels, then the higher threshold.
The complete leaderboard also reports precision, background false positives,
worst-10% and 140–146 metrics plus GT / SEM-111 / selected / error panels.
This ranking is for fast iteration and visual review only, not a blind-test
claim or a formal replacement decision.

## Command

```bash
python3 scripts/run_sem120_dev_threshold_sweep.py \
  --config experiments/runs/SEM-120-DEV-THRESHOLD-SWEEP/config.yaml \
  --split experiments/splits/sem_001_scaffold_static_v1.json \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --sem001-result "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/result.json" \
  --sem001-predictions "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/validation_semantic_png" \
  --sem102-manifest "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1/prediction_manifest.json" \
  --sem102-audit-root "$ICSHM_OUTPUT_ROOT/r2/sem_102/validation_audit/final-20260902-r1" \
  --sem111-result "$ICSHM_OUTPUT_ROOT/r2/sem_111/final_protected_fusion/full-20260902-r1/result.json" \
  --sem111-predictions "$ICSHM_OUTPUT_ROOT/r2/sem_111/final_protected_fusion/full-20260902-r1/predictions" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_120/dev_threshold_sweep/full-20260903-r1"
```

## Result

All twelve thresholds completed. Before scoring, the runner hash-checked every
frozen SEM-001, SEM-102 and SEM-111 input, then reconstructed the 0.80 result
bit-exactly from the frozen inputs. At every threshold, deck, tower and
foundation remain bit-identical to SEM-001. No RGB, checkpoint, geometry,
unlabelled image, hidden test, real RGB or external-camera input was opened.

The pre-declared developer score selects **0.20** (score `0.758950`). Relative
to the historical SEM-111 0.80 result, it changes mIoU from `0.760635` to
`0.770685` (+1.005 pp), `stay_cable` IoU from `0.638137` to `0.687827`
(+4.969 pp), and `stay_cable` recall from `0.711445` to `0.858168` (+14.672
pp). Its cost is background-to-`stay_cable` pixels increasing from `347,412`
to `743,906` (2.14x the SEM-111 count). Worst-10% and challenge 140--146
mIoU also improve to `0.460220` and `0.770756`.

Threshold **0.30** is the nearby balanced Pareto alternative: it has the best
plain mIoU (`0.771317`) and the best `stay_cable` IoU (`0.690498`) in this
sweep, while using `629,688` background-to-`stay_cable` pixels and retaining
`0.835102` cable recall. It loses 2.307 pp recall to the score-selected 0.20,
so it is not the pre-declared developer leader; it is retained for visual
comparison rather than silently discarded.

This is a developer ranking only. It does not replace SEM-111's historical
formal result, and it is not evidence of blind-test or hidden-view
generalisation. The full threshold table, selected 0.20 predictions and
GT / SEM-111 / selected / error panels (054, 140 and 183) are stored under the
configured Git-ignored output root. `result.json` SHA-256 is
`2cc91b76d31c4915cc8775376700549f03bc469c327953fd6aad950a276f1798`.
