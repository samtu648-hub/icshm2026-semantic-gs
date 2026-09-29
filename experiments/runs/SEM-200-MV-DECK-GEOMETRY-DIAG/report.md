# SEM-200-MV-DECK-GEOMETRY-DIAG — multi-view deck geometry gate

## Status

Completed on 2026-09-06. The calibration gate failed; the 111-view check and
reused 74-view development set remained unopened.

## Question

Can deck labels from other spatially separated training cameras be lifted to a
stable 3D surface through the frozen RGB-030-FULL geometry and reprojected to
an unseen camera with enough precision to improve SEM-160?

## Frozen protocol

The exact dominant generated Gaussian supplies camera-space depth. Source deck
pixels are sampled every four pixels, must agree with at least one of three
source-training cameras under a 5% depth and deck-label check, and are then
projected from the eight strongest allowed source cameras. A target pixel needs
two source votes. Frozen-renderer depth supplies deployable target visibility;
no target RGB or label is an inference input.

Only SEM-160 background pixels that retain the original foundation-safe deck
eligibility may become deck. Existing deck, `stay_cable`, tower, foundation and
foundation rejection pixels remain exact. There is no parameter grid.

All 219 target vote maps and candidates must be materialised before target
evaluation. The deterministic 108-view calibration half is opened first. The
111-view check remains closed unless the one fixed recipe passes calibration;
the reused 74-view development set remains unavailable throughout.

## Result

All 219 frozen depth maps, vote maps and candidates were sealed before target
evaluation. Relative to SEM-160 on calibration:

- mIoU: `0.868790 -> 0.868716` (-0.00742 pp)
- deck IoU: `0.945301 -> 0.944942` (-0.03592 pp)
- deck recall: `0.988000 -> 0.988456` (+0.04563 pp)
- deck precision: `0.956281 -> 0.955486` (-0.07954 pp)

The rule added 10,035 pixels: 3,426 true deck, 5,691 background, 846
`stay_cable`, 71 tower and 1 foundation, for 34.14% addition precision. Sixty-two
targets had no allowed source camera and 71 had no two-vote pixel. A reporting
aggregation error occurred only after the sealed predictions were complete; it
was fixed and evaluation resumed from the hashed artifacts without changing or
recomputing the candidate.

Depth/evidence/prediction/result SHA-256 are respectively
`fac12844...57a9d`, `9795670f...6026`, `2def85fc...f632` and
`0d0ea220...a9d`. Drop this recipe and retain SEM-161.
