# SEM-395 frozen cross-view reliability protocol

Experiment ID: `SEM-395-RGB150-CROSSVIEW-RELIABILITY-DIAG`.

## Question and scope

This experiment asks whether a single conservative two-source filter can identify
pixels where frozen RGB150 geometry and frozen SEM392 source semantics form useful
cross-view evidence. It does not train a model, change a checkpoint, replace a
prediction in production, or evaluate the 30-view Dev30 holdout.

Only the 270 names in `primary_fold.train_rgb_300` may be diagnostic targets or
sources. The 30 Dev30 holdout images and labels remain unopened. The 100 unlabeled
images provide neither supervision nor source evidence.

## Mandatory stage order and seal chain

1. `prepare` validates all parent and protocol identities. It reads camera, split,
   trajectory, registry, and parent metadata only. It must not open a label file.
   It deterministically seals exactly 24 targets, two sources per target, six
   visualization targets, the only filter, the budget, the complete frozen RGB150
   renderer source snapshot hashes, and a future spatial-confirmation split candidate.
2. `render` verifies the prepare seal, acquires the shared GPU lock, reloads the
   frozen RGB150 point cloud, verifies selected RGB renders pixel-exactly against
   saved RGB150 PNGs, and saves native inverse-depth sum plus alpha. It reads no
   semantic label.
3. `evidence` verifies both earlier seals. It uses the frozen depth/alpha and sealed
   SEM392 probability NPZs to execute the one fixed filter. It saves sparse reliable
   projections and disjoint rejection counts, but reads no semantic label.
4. `score` opens only the already sealed target/source labels from the 270-view
   training partition. It reports descriptive counts, confusion, fixed groups,
   six visual panels, the future-split history audit, and one of four outcomes.

Every stage uses exclusive output creation and preserves failures. A downstream
stage verifies the exact SHA256 of every prerequisite result and its artifacts.

## Camera and depth convention

All diagnostics use the RGB150 full-resolution 1320x989 PINHOLE grid. Official
poses are CAMERA_FROM_WORLD: `x_c = R x_w + t`, with `C = -R^T t`. COLMAP
intrinsics use edge-origin coordinates; array pixel `(x,y)` has edge coordinate
`(x+0.5,y+0.5)`.

The native Standard renderer exposes `D_inv = sum(T alpha / z)`. A second render
with all primitive colours equal to one on black yields `A = sum(T alpha)`. This
experiment uses `Z_proxy = A / D_inv`. It is an opacity-normalized reciprocal
inverse-depth surface proxy, not expected depth, geometric truth, or an independent
surface reconstruction.

## Frozen selection

Target and source selection uses only official centres, optical axes, trajectory
segments, and COLMAP shared-track counts. Every eligible segment receives one
target before remaining slots are allocated. The two sources are selected from the
same 270-view training universe under the exact limits in `config.yaml`. No RGB,
probability, prediction, label, error, or visual-quality value participates.

The six visual targets are evenly spaced indices, including both endpoints, in the
sealed target list sorted by segment, allocation slot, and filename. Their identity
cannot change after scoring.

## Single reliability filter

For every target pixel, reject in this fixed order:

1. invalid target proxy depth;
2. target alpha below 0.85;
3. incomplete or discontinuous 3x3 target depth neighbourhood (>5% range);
4. source projection outside/behind the source camera;
5. invalid source proxy depth;
6. source alpha below 0.85;
7. incomplete or discontinuous 3x3 source depth neighbourhood (>5% range),
   sampled by the nearest source pixel only for this binary validity test;
8. forward proxy-depth disagreement above 4%;
9. backward projection outside/behind the target camera;
10. backward proxy-depth disagreement above 4%;
11. cycle reprojection above 1.0 native pixel;
12. bilinearly sampled and renormalized SEM392 source confidence below 0.80.

A target pixel is reliable only when both selected sources pass all checks and their
smallest-class-ID argmax labels agree. No second scheme, threshold sweep, class
exception, morphology, target-confidence gate, or post-score source replacement is
allowed under this ID.

## Scoring and interpretation

After the evidence seal, the training-side labels are used to report:

- reliable-pixel five-class confusion and correction/damage counts;
- sources-agree/target-GT-disagrees counts;
- source-view nearest-label agreement as a diagnostic separation of semantic source
  errors from geometry/annotation mismatch;
- target-confidence, alpha, covisibility, trajectory-segment, and class groups;
- first-rejection attribution, including low opacity, depth discontinuity,
  occlusion/depth conflict, cycle conflict, source confidence, and two-source
  semantic disagreement.

The counterfactual replacement is descriptive. It is neither a training gain nor a
deployed fusion result. Ground truth must never alter eligibility, source selection,
or the saved candidate prediction.

The final status must be exactly one of the four strings frozen in `config.yaml`.

## Future spatial confirmation candidate

`prepare` also selects three ten-view contiguous labeled trajectory regions using a
seeded camera-only ordering. Every camera within the strict 0.5220235426 centre
radius is a buffer. The candidate is accepted only if at least 340 RGB views and
250 labeled views remain and removing the regions does not increase the number of
connected components in the retained >=8-track camera graph relative to the full
released graph. The graph is already disconnected under this truncated metadata
criterion, so claiming global connectedness would be false.

This candidate is not independent today: released views have already participated
in historical models. The result therefore includes a history audit and calls the
partition a future once-only confirmation split. A valid future execution must
rebuild initialization, RGB, and semantics from scratch while excluding every
confirmation and buffer view from RGB, labels, observations, point colours,
features, pseudo-labels, source banks, tuning, and visualization selection.

## Budget and stop policy

The limits are 24 targets, 48 pairs, 30 GPU minutes, 120 CPU minutes, 11,200 MiB
peak GPU allocation/reservation, and 4 GiB new output. Any access violation, hash
change, frozen-RGB mismatch, malformed probability, coordinate/count mismatch, or
budget excess stops the run and preserves the failure record. No same-ID repair may
change the scientific filter or selection; an engineering-only retry must add, for
example, `--attempt r2` to every stage and retain the failed attempt. The default
attempt is `r1`.
