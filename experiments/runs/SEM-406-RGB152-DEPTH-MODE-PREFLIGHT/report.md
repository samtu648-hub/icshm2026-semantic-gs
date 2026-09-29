# SEM-406-RGB152-DEPTH-MODE-PREFLIGHT

- Status: completed; two semantic gates failed; current split closed.
- Parent: SEM-405.
- Single variable: local largest-gap split → whole-ray variance-optimal ordered two-mode split.
- Semantic labels, SEM405 samples, confirmation cameras and buffer cameras: forbidden.

The sealed protocol is `docs/experiments/sem406_depth_mode_preflight_protocol.md`. It retains the 20%/20% mass requirement and SEM405 semantic-difference gates, changes only the geometry partition objective, and uses three different unlabeled training camera poses. Passing would have authorized only a separately preregistered training-label diagnostic, not training.

## Result

The fixed grid on the three new camera poses produced 1,296 nonempty traces. Every trace admitted a 20%/20% mass split, and the variance-optimal partition explained a median 73.0165% of weighted depth variance. A total of 1,124 rays (86.7284%) passed the fixed 50% clear-mode condition. Diagnostic color replay passed with maximum absolute error `3.95214e-6`; depth order had zero violations and frozen RGB was exact before and after.

The geometry did not induce sufficient semantic ownership. Only 75/1,124 clear modes had different semantic argmax: 6.6726%, below the unchanged 10% gate. JS divergence had median 0.006684 bits and 75th percentile 0.019171 bits, also below the fixed 0.02 gate.

| Gate | Observed | Required | Pass |
|---|---:|---:|:---:|
| Diagnostic color replay maximum | 3.95214e-6 | at most 1e-5 | yes |
| Nonempty traces | 100% | at least 90% | yes |
| Mass-eligible splits | 100% | at least 80% | yes |
| Clear two-mode rays | 86.7284% | at least 20% | yes |
| Clear-mode argmax disagreement | 6.6726% | at least 10% | no |
| Clear-mode JS divergence p75 | 0.019171 bits | at least 0.02 | no |
| Depth-order violations | 0 | at most 0 | yes |

## Decision

`label_diagnostic_authorized=false`; no labels or training are opened. The whole-ray variance-optimal formula is closed without threshold scans. Together SEM405 and SEM406 show that abundant one-dimensional depth structure does not reliably identify semantic ownership, whether the boundary is a local discontinuity or a global two-mode optimum. The next representation must add cross-view surface identity/consistency rather than another per-ray cut.

Complete result: `outputs/quality/SEM-406/preflight-r1/result.json`; evidence SHA256: `bf314cd9ad6730e2ae10aa94a5f862f3913d22bd18e3016940dcc23c3263989b`.

Repository verification passed after recording SEM405/406: upload safety, secret scan, Markdown links, 271 JSON-compatible configurations, shell syntax, and 674 unit tests (87 optional skips). `git diff --check` also passed.
