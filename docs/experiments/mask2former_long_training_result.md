# Mask2Former long-training development result

SEM434 (RGB152/v1) and SEM435 (RGB154/v2) are complete. Both use historically
opened development folds, not blind or hidden-test data.

Training fit rises monotonically in both folds, proving the original 6000 endpoint
does not exhaust optimization capacity. Spatial confirmation is not monotonic:
v1 peaks at 8000 and then partially overfits, while v2 dips at 8000/10000 and peaks
at 12000. The shared 12000 endpoint is above each run's own 6000 control in both
folds, but the regional response differs: v1 region-3 mIoU/boundary decline while
v2 improves all three regions.

Mean confirmation mIoU across the two folds is 89.907290% at 6000, 90.029113% at
8000, 89.973711% at 10000 and 90.225699% at 12000. The same means for foreground
are 87.539011%, 87.686204%, 87.617890% and 87.927989%. Thus 12000 is the best
aggregate development endpoint, but it is not a stable shared curve shape.

The conservative decision is `mixed_fold_trend_no_full300_transfer`: do not train
RGB140 all-300 or claim a new final model from these data. A justified follow-up is
one paired second semantic seed with the exact same 6k+fixed-low-LR recipe, or a
new preregistered tail that tests whether an 8k early-stop rule is robust. It must
not select a seed or tail separately per fold. No Scaffold experiment is started:
the present evidence is an optimization-duration × spatial-region interaction,
not a demonstrated reconstruction-backbone interaction.

The 97/94 available released-unlabeled predictions change by only about
0.18%--0.22% of pixels depending on endpoint, but v1 and v2 even disagree on the
direction of cable-area change. Without GT and without the all-300 RGB140 model,
this supports only the possibility that duration changes public-view predictions;
it does not establish that long training helps the full 100 images or the 144-frame
trajectory.

A qualitative-only review page covers identifiers 301--400 and displays each
fold's frozen RGB render, 6000 control and selected long endpoint side by side:
`outputs/quality/SEM-434/r2/review/unlabeled_long_training/index.html`. RGB152 has
97 registered released views (374--376 missing); RGB154 has 94 (310--311 and
377--380 missing). The page labels missing views and contains no GT-derived score.
