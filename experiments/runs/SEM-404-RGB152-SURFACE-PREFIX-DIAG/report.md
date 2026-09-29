# SEM-404-RGB152-SURFACE-PREFIX-DIAG

- Status: completed; two all-required gates failed; training not authorized.
- Parent: SEM-403.
- Variable: none; read-only training-label diagnostic.
- Confirmation and buffer access: forbidden.

The fixed protocol is `docs/experiments/sem404_surface_prefix_diagnostic_protocol.md`. It tests one sealed 80%-mass front surface prefix against native semantic composition on deterministic cable-boundary/interior and non-cable-near/far training pixels. All gates are required.

## Execution and isolation

All attempts used the same 201 training labels, 25,728 deterministic samples, 80% surface fraction and quality gates; confirmation and buffer access remained zero. Attempt `r1` compared a replay buffer with a separate semantic CUDA call and exposed rare equal-depth ordering noise. Attempt `r2` corrected validation to use each three-channel block's own forward buffer without changing scientific evidence. Attempt `r3` additionally mirrored CUDA float32 arithmetic instead of CPU float64 in the replay. All three produced identical argmax quality counts. The strict semantic replay maximum improved from `0.002117` to `0.00002910`, with only 14/25,728 samples above `1e-5`, but the frozen maximum-error gate still failed and is not relaxed.

## Results

Each stratum contributed 6,432 samples. Boundary rays were substantially more semantically layered: front/tail argmax disagreement was 38.2307% on the cable boundary, versus 12.9042% for pooled cable-interior and far-noncable samples, an enrichment of 25.3265pp. Median effective contributors were 28.08 at the cable boundary, 28.13 inside cable, 25.85 just outside cable and 23.82 far from cable. The fixed 80% prefix still contained a median 63/59/57/49 contributors respectively, so it is a surface band rather than a hard winner.

The proxy was strongly positive on cable pixels. On the boundary, it changed 611 predictions, corrected 370 and harmed 45: net `+325`, correction precision 89.1566%, and accuracy `26.6947% → 31.7475%` (`+5.0529pp`). Cable interior accuracy also increased `+3.2960pp`. Far non-cable was nearly unchanged (`-0.0311pp`, net `-2`).

The benefit shifted errors across the boundary. In the 2px non-cable exterior, the prefix corrected 92 but harmed 176, net `-84`; accuracy changed `89.7233% → 88.4173%`, a `-1.3060pp` drop versus the sealed maximum 1pp. This protection gate fails even though the cable-side gates pass.

| Gate | Observed | Required | Pass |
|---|---:|---:|:---:|
| Semantic replay maximum error | 2.91018e-5 | at most 1e-5 | no |
| Sampled cable-boundary pixels | 6,432 | at least 4,096 | yes |
| Changed cable-boundary pixels | 611 | at least 512 | yes |
| Cable-boundary net corrections | +325 | at least +256 | yes |
| Cable-boundary correction precision | 89.1566% | at least 60% | yes |
| Cable-interior accuracy drop | -3.2960pp (gain) | at most 1pp | yes |
| Near-noncable accuracy drop | 1.3060pp | at most 1pp | no |
| Boundary front/tail disagreement enrichment | 25.3265pp | at least 5pp | yes |

## Decision

`training_authorized=false`. SEM404 provides the first positive evidence on the new mainline that ray-layer responsibility is specifically enriched at `stay_cable` boundaries and can expose cable-side corrections. It simultaneously shows that a global class-agnostic 80% front prefix is not safe: it transfers too much error to the immediately adjacent non-cable side. This exact formula is closed; no fraction, depth-band, threshold or stratum-conditioned scan is allowed on the same evidence.

The broader visibility/surface hypothesis remains open, but its next implementation must change representation rather than tune this prefix—e.g. an explicit surface/surfel ownership unit that can preserve both sides of a boundary. No frozen-geometry training or semantic-to-XYZ stage is launched. Complete result: `outputs/quality/SEM-404/diagnostic-r3/result.json`; evidence SHA256: `bd3d5dafab311aa130b22d91c3dcda6978f9e5a000bf7c54d9a2cdbe7d732474`.

Repository verification passed after recording SEM403/404: upload safety, secret scan, Markdown links, 269 JSON-compatible configurations, shell syntax, and 670 unit tests (87 optional skips). `git diff --check` also passed.
