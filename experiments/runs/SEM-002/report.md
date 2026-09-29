# SEM-002 / S1 — contribution-aware visibility supervision decision

- Status: completed main-fold screen; **drop / unproven**.
- Parent: `SEM-001`; code freeze `56695181985332e55f030567f3293758638d2ef9`.

## Scope and integrity

`SEM-020-EQ` proved disabled-gate equivalence with SEM-001: logits and class PNG are identical, frozen RGB difference is zero, and the custom all-contributor path is within its pre-registered numerical tolerance. The training-side-only diagnostic froze cumulative `0.95` and relative-max `0.05` thresholds from all and only 219 semantic training cameras. Both 8-iteration CUDA smokes passed backward, semantic checkpoint reload, external official-camera RGB/logits/PNG, and RGB=0.

Both main-fold candidates then used the same SEM-001 initialization, 219/74 spatial split, labels, seed, 10k budget, class weights, resolution, official `CAMERA_FROM_WORLD` camera convention and frozen RGB artifact. No unlabeled RGB, validation supervision, pseudo-label, GeometryPrior, RGB change, S2/S3 mechanism or other GS backbone entered either run.

## Main-fold summary

| Candidate | Gate | mIoU incl. bg Δ | stay_cable IoU Δ | background→stay_cable Δ | Protected-class condition | Result |
|---|---|---:|---:|---:|---|---|
| SEM-021 | Cumulative prefix 0.95 | +0.80 pp | +0.52 pp | −37,017 px | tower −0.54 pp | Drop: mIoU < +1 pp; tower exceeds limit |
| SEM-022 | Relative maximum 0.05 | +1.51 pp | +0.87 pp | −760,439 px | tower −0.89 pp | Drop: tower exceeds −0.5 pp limit |

Both preserve RGB exactly, reload semantic checkpoints, and accept the external official COLMAP camera. `SEM-022` gives the strongest evidence that reducing low-contribution supervision reduces the target background→stay_cable error, but the protected tower regression means this S1 mechanism has not passed the pre-registered reliability bar.

## Decision and next boundary

The S1 hypothesis is **partly supported but not confirmed**: contribution-aware gating improves the target false-positive metric and stay_cable recall/IoU, yet neither candidate satisfies every main-fold gate. SEM-001 remains the frozen S0 semantic baseline. Do not start SEM-023, S2 conflict statistics, S3 view residuals, or a three-fold/three-seed batch from these results. A future protocol revision requires a new pre-registered experiment rather than threshold tuning or rerunning these seeds.
