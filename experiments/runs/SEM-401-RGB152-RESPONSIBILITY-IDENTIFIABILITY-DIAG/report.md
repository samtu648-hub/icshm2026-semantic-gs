# SEM-401-RGB152-RESPONSIBILITY-IDENTIFIABILITY-DIAG

- Status: completed; two semantic-evidence gates failed; responsibility-initialization training not authorized.
- Parent: SEM-400-RGB152-SEMANTIC-DIRECTION-SPLIT.
- Variable: none; read-only projection diagnostic.
- Confirmation access: forbidden; training labels only.

## Execution and isolation

The fixed protocol is `docs/experiments/sem401_responsibility_identifiability_protocol.md`. Attempts `r1` and `r2` stopped before label access because the script entry path and then an unnecessary optional OpenCV import were unavailable. Both failures are preserved. Attempt `r3` changed only those engineering dependencies and completed on CPU. It read exactly the 201 registered training labels, zero confirmation labels, the sealed SEM400 parent list/directions/common logits, and the frozen RGB152 geometry.

## Results

There were 203,704 positive-depth, in-frame paired parent-view observations. Of these, 108,386 had at least 1-pixel positive/negative child-centre separation. The resolvable separation quantiles were `1.0000 / 2.0885 / 4.9832 / 14.1695 / 590.8585` pixels. A total of 2,406/4,096 parents had at least three resolvable views, or `58.740234%`, so the spatial-coverage gate passed.

The projected label evidence did not separate with the geometry. Among covered parents, the Jensen-Shannon divergence quantiles were `0 / 0 / 0 / 0.002467 / 0.320644`, and the `stay_cable` frequency-difference quantiles were `0 / 0 / 0 / 0.023256 / 0.65625`. Only 9/4,096 parents (`0.219727%`) had JS at least 0.02 and different side-majority classes, versus the preregistered 512-parent requirement. Only 3/4,096 (`0.073242%`) had complementary `stay_cable`/non-cable majority and at least 0.10 cable-frequency difference, versus the 256-parent requirement.

| Gate | Observed | Required | Pass |
|---|---:|---:|:---:|
| Parents with at least three resolvable views | 58.740234% (2,406) | 50% | yes |
| Differentiated parents among all selected | 0.219727% (9) | 12.5% (512) | no |
| `stay_cable`-separated parents among all selected | 0.073242% (3) | 6.25% (256) | no |

## Decision

`training_authorized=false`. The children are usually separated by multiple pixels, but the two world-space positions still land in the same semantic region across training views. Because this projection-only calculation is already an optimistic upper bound that ignores opacity and occlusion, adding actual rasterized responsibility is not expected to create the missing population of complementary supervision.

The current semantic-gradient displacement plus responsibility-initialization route is therefore closed. No SEM401 training arm, threshold reduction, larger offset, direction scan, or reuse of the opened SEM400 confirmation set is allowed. The complete result is `outputs/quality/SEM-401/diagnostic-r3/result.json`; evidence SHA256 is `9f9cdd5ce9740d45909e16e3616afd579fad2464bed667ea636c391658b9a4a5`.
