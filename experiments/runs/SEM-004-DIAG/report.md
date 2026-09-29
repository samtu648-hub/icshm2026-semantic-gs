# SEM-004-DIAG — frozen SEM-001 stay_cable false-positive source diagnostic

- Status: **diagnostic-only; not a semantic candidate**.
- Parent baselines: SEM-001 (frozen S0), SEM-002/S1 (drop), SEM-003/S2 (drop).
- Scope: read-only error, LabelMe geometry, frozen contribution and training-side observation-direction statistics. No optimizer, new checkpoint, parameter update, GeometryPrior, pseudo-label, or RGB/Scaffold training is permitted.

## Frozen protocol

The only semantic model is the final SEM-001 per-anchor five-logit checkpoint on frozen RGB-030-FULL Scaffold-GS. Class ID 2 is `stay_cable`. The experiment uses the existing 219/74 spatial split and LabelMe-derived full-frame PINHOLE labels. Training-side statistics freeze all diagnostic definitions; validation is an error audit only and cannot select a threshold, mask, loss, or capacity.

The contribution audit decodes exact frozen rasterizer marginal contributions. It neither gates nor backpropagates through them. The SEM-003 all-contributor evidence artifact is read only to annotate dominant-anchor support and its already-frozen conflict eligibility; it does not reactivate the failed S1/S2 mechanisms.

## Pre-registered decision rule

The external, Git-ignored result will contain four training-only signals: dominant error geometry, LabelMe fine-geometry prevalence, frozen-contribution separation, and held-out-within-training directional-pattern reproducibility. At least three must pass the thresholds in `config.yaml` before the result may be labelled `hypothesis_ready_for_new_preregistration`. Otherwise the required conclusion is `no_go_for_new_semantic_mechanism`; neither conclusion starts a new model.

## Actual run and reproducibility

- Status: **completed diagnostic-only; not a semantic candidate**.
- Run HEAD: `4dc7ed989577ae453822c218c6f45180c191709e`; the run used the uncommitted SEM-004 diagnostic implementation plus pre-existing user changes. The captured tracked-diff SHA-256 was `e5bd5b0ba01bb030b09053c2fdc867083a917ae6f79eca9316307b312d0c3418`.
- Seed: `20260805`; GPU: RTX 5070 12 GB; peak PyTorch allocation: 387 MiB; elapsed: 181.23 s.
- Inputs: final SEM-001 checkpoint SHA-256 `8e3dbe30e17f67081a72380018fc281c155ec00eed83cd5a19b3285bf4fb8aa3`; read-only SEM-003 all-contributor evidence SHA-256 `84b60b867e07e2a1a1e17cd47d053e1b00c8c19a820efdb0dfd53bdd35b8d3b0`.
- Result SHA-256: `2bd77c9b5ad8528891c40a238aa9f5efd15ce77136c7b7011114894ea5730a8e`; output-manifest SHA-256: `c01476b8a0fe032d51335526a6cfb7d54b1672b331d5c0e60d911e7ba90ff064` (310 Git-ignored files, 74.8 MB).
- External output root: `$ICSHM_OUTPUT_ROOT/r2/sem_004/diagnostic`. It contains 219 training and 74 validation-read-only semantic GT/prediction/error triptychs, component/anchor CSVs, plots, progress log, hashes and manifest. No artifact is tracked by Git.

## Frozen error audit

The validation-read-only result exactly reproduces the frozen SEM-001 local metric: mIoU with/without background `0.61610 / 0.56334`, stay_cable IoU `0.06876`, and background→stay_cable `13,166,331` pixels (15.0076% of true background). It therefore audits the intended checkpoint rather than a drifted renderer.

| Frozen SEM-001 error structure | Training (219; used for readiness rules) | Validation read-only (74) |
|---|---:|---:|
| background→stay_cable pixels | 32,169,907 | 13,166,331 |
| near GT cable boundary (≤8 px) | 2,077,480 (6.46%) | 293,404 (2.23%) |
| spatially remote from all GT cable | 30,092,427 (93.54%) | 12,872,927 (97.77%) |
| false-positive large connected expansions (≥4096 px) | 29,664,858 px | 12,314,767 px |
| predicted cable components | 67,748 | 26,400 |
| predicted cable component median area | 4 px | 4 px |

The false positives are overwhelmingly remote rather than a simple 2D dilation of true cable boundaries. The validation-only audit agrees with, rather than selects, this training-side characterization. “Remote” here is an image-space correlation, not proof of a 3D causal mechanism.

## LabelMe geometry audit

The released JSON re-rasterized exactly to every existing SEM-001 PINHOLE label: 219/219 training and 74/74 validation labels. Thus this audit found no label-conversion mismatch to explain the false positives. The 788 training-side stay_cable polygons are exceptionally thin: mean thinness `0.8445`, median `0.9208`, and 72.21% meet the pre-registered ≥0.80 thinness rule. The validation read-only polygons have comparable mean thinness `0.8589`.

The mean stay_cable rasterization-area change after SIMPLE_RADIAL→PINHOLE conversion is only −0.55%, so thin, discontinuous annotation geometry—not a large systematic warp-area change—is the relevant observed label characteristic. Per-shape overlap, adjacency and deterministic representative cases are in the Git-ignored label-geometry CSV/summary.

## Frozen contribution and direction correlations

No contribution gate was used. For training-side true-background pixels that SEM-001 predicts as stay_cable versus correct background, respectively:

| Diagnostic | bg→stay_cable | correct background |
|---|---:|---:|
| mean contributors/pixel | 105.98 | 93.82 |
| max/foreground contribution concentration | 0.10805 | 0.09837 |
| dominant-anchor support views | 70.38 | 80.63 |
| dominant-anchor stay_cable evidence fraction | 0.21405 | 0.04480 |
| dominant anchor in pre-existing SEM-003 conflict mask | 6.401% | 0.0935% |

The last row is a 68.46× training-side enrichment. The validation read-only audit shows the same direction (7.895% versus 0.598%); this supports a conflict-associated observation, but does **not** show that the failed SEM-003 Dirichlet loss should be retried.

For the separate training-only direction check, 14,863 anchors met stable-support requirements. Of them 7,433 had a ≥0.10 stay_cable-evidence range across fixed global azimuth bins; 7,069 reproduced that pattern across deterministic, disjoint training halves (47.56%). The method uses neither camera/image IDs nor a learned view-conditioned module.

## Conclusion and strict boundary

**`hypothesis_ready_for_new_preregistration`**, not a quality claim and not permission to train. All four pre-registered diagnostic signals passed: remote error structure, fine LabelMe cable geometry, frozen contribution/conflict separation, and training-half directional reproducibility. The evidence is correlational; it does not establish that a residual improves mIoU or blind-test accuracy.

SEM-001 remains the only frozen semantic baseline. SEM-021/022 and SEM-003 remain dropped; their thresholds/lambda will not be retried. SEM-004/S3 remains blocked until a separate Goal authorizes exactly one variable: an anchor-local background↔stay_cable residual using fixed direction bins, with RGB frozen, no camera/image IDs, no validation-selected mask, and a matched SEM-001 10k control. The pre-registered prospective batch is two 10k jobs, about 1.8 GPU-hours, <1 GiB peak PyTorch allocation per job and about 8 GiB external disk; it has not been launched.

## Freeze and external-camera re-audit

The diagnostic created no semantic optimizer or checkpoint. Frozen RGB artifact hashes remained equal, frozen RGB has no trainable parameters, and the repeated RGB render has maximum absolute pixel difference `0`.

The externally supplied official `140.png` `CAMERA_FROM_WORLD` camera with identity Sim(3) produced 1320×989 RGB, `[5,989,1320]` logits and a single-channel uint8 0–4 PNG; no test RGB or labels were read. The RGB/semantic/logits SHA-256 values are respectively `563e041da15e713f2acc3139f2b922d129b7db7910b18d18b8ac6489e3de10d3`, `e516db5b8eed1dc3eef952e170ecba03cfe95a9359ceb45655f967cc2c240085`, and `42631a3ed6381bdb8205f7f982820dc20aa0cd2922b51ca1d34721369f9a03ff`.
