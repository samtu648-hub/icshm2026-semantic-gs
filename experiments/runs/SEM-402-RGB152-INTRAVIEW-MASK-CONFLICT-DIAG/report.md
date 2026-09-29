# SEM-402-RGB152-INTRAVIEW-MASK-CONFLICT-DIAG

- Status: completed; population gate failed; training not authorized.
- Parent: SEM-398 and SEM-401.
- Variable: none; frozen read-only rasterized attribution diagnostic.
- Confirmation access: forbidden; 201 training labels only.

The fixed protocol is `docs/experiments/sem402_intraview_mask_conflict_protocol.md`. The diagnostic tests a statistic omitted by SEM398 and SEM401: whether one parent Gaussian's rendered footprint receives both `stay_cable` and non-cable supervision inside the same view. Thresholds and all-or-nothing gates were sealed before execution.

## Execution and isolation

Attempt `r1` stopped before scene or label access because a Python 3.11 process could not load the existing CPython 3.12 rasterizer extension. The failed result is preserved. Attempt `r2` changed only to the already-built project Python 3.12/CUDA environment; configuration, code hash, thresholds, and inputs remained fixed. It read exactly 201 training labels and zero confirmation labels. The frozen RGB152 native replay for `001.png` was pixel-exact both before and after the diagnostic, and no model or optimizer update occurred.

For each view, gradients of a unit scalar color probe were summed separately over `stay_cable` and non-cable pixels. These are rasterized alpha-transmittance contribution masses. All raw gradients were non-negative. The evidence therefore measures within-view footprint responsibility rather than projected child centres or a whole-image semantic-loss gradient.

## Results

Of 6,269 sealed eligible parents, 6,268 had at least three visible training views; visible-view quantiles were `6 / 32 / 47 / 92 / 147`. Coverage was therefore effectively complete. Foreground contribution was positive for 5,284 eligible parents and background contribution for 6,268.

Conflict was nearly absent. Covered-parent mean mask-signal quantiles were `0.412450 / 0.955947 / 0.986681 / 0.999877 / 1.000000`, where zero is balanced foreground/background responsibility and one is one-sided. Ambiguous-view counts had quantiles `0 / 0 / 0 / 2 / 95`. Only 2 parents satisfied all preregistered per-parent conditions: at least three visible views, at least three ambiguous views, and mean signal below 0.5. Their median visible-view count was 116, so these two were well observed rather than visibility noise.

| Gate | Observed | Required | Pass |
|---|---:|---:|:---:|
| Qualified within-view conflict parents | 2 | at least 512 | no |
| Qualified-parent median visible views | 116 | at least 10 | yes |
| Different parents in fixed 4,096 budget versus SEM398 | 723 | at least 512 | yes |

The low-signal ranking is non-identical to SEM398: the full selections overlap on 3,373/4,096 parents. But it does not identify a usable conflict population; almost all changed ranks are merely relative ordering among strongly one-sided parents.

## Decision

`training_authorized=false`. SEM398 did omit within-view cancellation, but correcting that statistic does not recover the missing population: only 2/6,269 eligible parents meet the fixed conflict definition, versus the required 512. Combined with SEM400 and SEM401, the evidence now rejects parent selection, displacement direction, projected child responsibility, and within-view rasterized footprint conflict as separate rescue explanations for this post-hoc split family.

No matched training experiment, threshold relaxation, per-pixel contribution cutoff scan, mask-signal variant, or reuse of the opened SEM400 confirmation set is allowed. A future joint reconstruction/semantics experiment must change the representation or supervision unit—such as ray/surface-aware semantic responsibility—rather than re-rank the same parent-Gaussian split. Complete result: `outputs/quality/SEM-402/diagnostic-r2/result.json`; evidence SHA256: `67267fde764dc00c578088d009d6798a41b4a2a320578190d6d5d419961c78a4`.

Repository verification passed after recording the result: upload safety, secret scan, Markdown links, 267 JSON-compatible configurations, shell syntax, and 666 unit tests (87 optional skips). `git diff --check` also passed.
