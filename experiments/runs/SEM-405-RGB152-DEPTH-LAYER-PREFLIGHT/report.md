# SEM-405-RGB152-DEPTH-LAYER-PREFLIGHT

- Status: completed; two all-required gates failed; current split closed.
- Parent: SEM-404.
- Variable: none; unlabeled representation-identifiability diagnostic.
- Semantic labels, confirmation cameras and buffer cameras: forbidden.

The sealed protocol is `docs/experiments/sem405_depth_layer_preflight_protocol.md`. It replaces the closed fixed-80%-prefix formula with one geometry-only split at the largest adjacent depth discontinuity for which each side owns at least 20% of accepted ray mass. Passing would have authorized only a separately preregistered label diagnostic, not training.

## Result

The fixed 24×18 grid on each of three training camera poses produced 1,296 samples, with zero label or protected-camera access. All traces were nonempty and 1,293/1,296 (99.7685%) admitted a 20%/20% mass split. The selected relative depth gap had median 9.5165%; 1,093 rays (84.3364%) passed the fixed 1% clear-layer condition. This strongly supports geometric two-layer availability.

Among clear rays, the layer JS-divergence median was 0.011842 bits and its 75th percentile was 0.040891 bits, passing the fixed 0.02 high-quantile gate. However, only 104/1,093 clear rays changed semantic argmax: 9.5151%, below the preregistered 10% minimum. This is a real gate failure and is not rounded up.

The diagnostic color replay had one sample above the strict `1e-5` maximum: `1.0395281e-5` at `001.png (918,930)`; all other samples were below the gate. Depth order had zero violations, and the frozen RGB PNG was exact before and after. Because the semantic gate already failed, no replay-only rerun was used to promote the experiment.

| Gate | Observed | Required | Pass |
|---|---:|---:|:---:|
| Diagnostic color replay maximum | 1.03953e-5 | at most 1e-5 | no |
| Nonempty traces | 100% | at least 90% | yes |
| Mass-eligible splits | 99.7685% | at least 80% | yes |
| Clear two-layer rays | 84.3364% | at least 20% | yes |
| Clear-layer argmax disagreement | 9.5151% | at least 10% | no |
| Clear-layer JS divergence p75 | 0.040891 bits | at least 0.02 | yes |
| Depth-order violations | 0 | at most 0 | yes |

## Decision

`label_diagnostic_authorized=false`; no labels or training are opened. The exact largest-adjacent-gap definition is closed, with no threshold or mass-fraction scan. The result nevertheless narrows the problem: geometry supplies abundant two-layer structure, but a local maximum gap is slightly too weak as a semantic ownership unit. A subsequent candidate must change the partition objective itself—for example, a whole-ray weighted two-mode partition—under a new ID and sealed gates.

Complete result: `outputs/quality/SEM-405/preflight-r1/result.json`; evidence SHA256: `ccecba272cd56b0bc93f9ff93869947e91f91057eb5ffc3597fdd0095d9959b6`.
