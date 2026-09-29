# QUAL-018-ABSGS-DIVERGENCE-DIAG

Registered 2026-09-21; completed. This is an optimization-trajectory diagnosis,
not a reconstruction recipe or blind confirmation. RGB159/160 share RGB152's
78,045-point training-only initialization, spatial-v1 353/30/17 roles, seed 0,
native source and 30k settings. Historical RGB152/156/158 remain frozen.
[Protocol](../../../docs/experiments/absgs_divergence_diagnostic_protocol.md).

## Execution

- The 701-step smoke and all six orchestrated stages exited 0. Stage time was
  5,992.106 seconds: train A/B 2,809.654/2,771.492 seconds, evaluate
  129.181/128.899 seconds and analyze 59.386 seconds.
- Native worker peak allocated/reserved memory was 1,957.714/4,600 MiB for A
  and 1,961.667/5,000 MiB for B. The systemd unit ended `success`, is inactive,
  and no GPU compute process remains.
- Final validation passed script compilation, 3/3 lineage tests, all 307
  JSON-compatible configs, `git diff --check` and repository health with 687
  tests passed and 88 optional-runtime skips.
- Milestones at 3000/6000/9000/12000 coincide with opacity resets. Their
  immediate low PSNR is transient and is not treated as irreversible onset.

## Direct evidence

- Initial parameters, initial RNG, runtime flags, all 30,000 sampled cameras,
  loaded image tensors and camera tensor metadata are exact. Recorded RNG never
  diverges.
- XYZ/opacity state and XYZ/opacity/screen gradients first differ after update
  1; sampled rendered RGB first differs at step 2. Frozen state-0 and state-599
  repeats produce exact forward outputs but non-exact backward gradients with
  unchanged RNG. Maximum XYZ/screen differences are `2.24449e-7/1.28523e-7`
  at state 0 and remain non-zero at state 599. This is consistent with the
  frozen CUDA backward's unordered `atomicAdd` accumulation, without claiming
  that a single kernel instruction fully explains every historical endpoint.
- Discrete topology first differs at step 600. A/B clone 167/163 children, the
  selected parent sets have symmetric difference 90, and stable-lineage
  witnesses cross the fixed `0.0002` score threshold. Point count and prune
  outcomes also first differ at step 600. This directly establishes numerical
  perturbation followed by threshold amplification without RNG divergence.
- Screen-size pruning is ineffective in this source lifecycle. Both arms have
  119/119 active events with oversized points before append, totaling
  50,083,067/50,482,692 point-event incidences. All 288 append calls per arm
  clear `max_radii2D`; actual final screen-prune incidences are zero.
- Stable-lineage linkage finds 509/512 active-period top-32 contributor slots
  had previously exceeded 20 px. The link is causal to image influence but not
  uniformly harmful: removing A's top 32 at step 7000 improves `156.png` by
  2.897785 dB and `159.png` by 0.987100 dB, whereas at step 30000 it improves
  `156.png` by 0.673610 dB but harms `159.png` by 0.538490 dB. B shows the same
  view dependence. A pruning-order fix therefore requires a new matched
  experiment and cannot be declared beneficial from this diagnostic alone.

## Endpoints

- A train/holdout PSNR is 23.296920/19.818818 dB; B is
  23.289016/19.957586 dB. Holdout B-A is `+0.138768 dB`, SSIM `+0.002288`,
  LPIPS `-0.003915`.
- Region-1 PSNR is 18.599764/18.535791 dB. Both trajectories outperform the
  historical RGB152/156/158 holdout endpoints and do not reproduce the earlier
  0.85--0.99 dB low runs. Instrumentation synchronization can itself alter CUDA
  scheduling, so the historical worst endpoint remains only partially caused.
- Final point counts are 792,639/793,519. Independent replay verifies all
  artifact/trace hashes, camera roles, lineage, thresholds, counts, 144 probes
  per arm and all 383 evaluation images; maximum independent PSNR error is
  `7.10543e-15 dB`.

## Artifacts and decision

- `analysis-result.json`: SHA256
  `2df513ae1ed7b162d2330bf96cf85046fa3bb185dc3e51ba6998df1e599dc066`.
- `independent-audit.json`: SHA256
  `29ddddd96f415bdc9d5b3fc02e1081d8336bab7bfa21c9e0b63b6ba198dcc1ef`,
  status `passed`.
- `causal-summary.json`: SHA256
  `fbd590c7252799c2fcd6303b68e897f3ff5a2f6a30eec1a0aa859e229f575eac`.
- `screen-prune-lineage-audit.json`: SHA256
  `5399b4b6ea2fc049129266e7d677c821b32a4cca31c3ac2a24a690ca759c1ba7`,
  status `passed`.

The accepted conclusion is limited to backward non-repeatability, threshold
amplification and a disabled native screen-radius prune. It does not identify a
single complete cause for the historical worst quality or authorize a fix.
RGB140+SEM378 and the Standard efficiency fallback remain unchanged. Rollback
is to ignore these read-only diagnostic sidecars; no frozen training source or
production model was modified.
