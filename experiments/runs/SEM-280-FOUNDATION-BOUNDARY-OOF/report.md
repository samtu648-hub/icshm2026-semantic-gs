# SEM-280-FOUNDATION-BOUNDARY-OOF

Status: completed calibration gate failed; fixed boundary-loss candidate dropped.

## Hypothesis

SEM-270's matched native960 five-class control improved foundation IoU by
4.779pp, precision by 6.146pp and total mIoU by 0.973pp relative to SEM-160,
but lost 1.761pp recall. After sealing that failed decision, 81.81% of the
23,658 old true-foundation pixels deleted by the control were within 16 pixels
of a true foundation boundary, while only 43.89% were within 4 pixels.

Test one fixed training-only intervention: retain all five semantic classes and
multiply weighted cross entropy by 2.0 only on true foundation pixels in the
fixed 16-pixel inner boundary band. The matched control uses multiplier 1.0 and
the original weighted CE reduction. No label dilation or boundary operation is
used at inference.

## Frozen protocol

Both arms start from the same original SegFormer-B2 initialization and share
the SEM-270 native960 three-fold split, guards, seed, class weights, exact
view/crop/flip plan, AdamW settings, 3000-step budget, final-only checkpoint,
960/480 inference and fixed foundation probability threshold 0.5. Only the
background/foundation lane may change; deck, `stay_cable`, tower and RGB remain
exact. Historical foundation rejection is released in this research copy.

Both 219-view prediction sets must be sealed before reused108 calibration
scoring. The candidate must pass the original foundation IoU, precision,
recall and mIoU gates versus SEM-160, improve recall over the matched control
by at least 1pp, retain control IoU within 0.5pp and mIoU within 0.1pp, and keep
each spatial fold within 1pp foundation IoU of its SEM-160 baseline. Failure
closes check/final/developer scoring and does not change frozen SEM-161.

## Result

The preregistered full run completed all six models, 18,000 optimization steps,
438 OOF probability maps and 438 fused PNGs. Both 219-view arm outputs were
sealed before the reused 108-view calibration labels were decoded. The
candidate failed the calibration gate, so the 111-view check, reused 74-view
developer quality audit and all-219 final fit remained closed.

Calibration metrics are:

| arm | foundation IoU | precision | recall | five-class mIoU |
|---|---:|---:|---:|---:|
| SEM-160 baseline | 65.5965% | 67.6357% | 95.6056% | 86.8790% |
| matched five-class control | 69.4430% | 72.8044% | 93.7658% | 87.6624% |
| boundary-loss candidate | 66.4589% | 69.0353% | 94.6830% | 87.0550% |

Relative to SEM-160, the candidate changes foundation IoU/precision/recall by
`+0.8624/+1.3996/-0.9226pp` and total mIoU by `+0.1760pp`. It therefore passes
only the precision-gain and mIoU-nondecrease quality checks; it misses the
`+1pp` IoU target, the maximum `-0.5pp` recall drop, and the absolute 95%
recall floor. Relative to the same-run control, it recovers `+0.9172pp` recall
but loses `-2.9841pp` IoU, `-3.7690pp` precision and `-0.6073pp` mIoU. It
misses all three control-relative gates.

Spatial safety also fails. Candidate foundation IoU relative to SEM-160 is
`+10.0248/-1.6708/-5.7374pp` on folds 0/1/2. Relative to the matched control,
candidate IoU is `-4.0755/-2.4362/-2.7898pp`, while recall changes by
`+0.7935/+0.1339/+1.8412pp`. Only fold 0 passes the preregistered baseline
safety floor. The candidate makes 142,463 correct and 120,046 harmful changes
relative to SEM-160, versus 186,422 correct and 99,675 harmful changes for the
control. It recovers 9,274 true-foundation pixels from the control but also
adds 73,604 background false positives. The visual panels agree with this
trade-off: some broken foundation edges become more complete, but the broad
positive emphasis also retains or expands incorrect foundation regions.

## Engineering and audit evidence

The engineering smoke completed in 4.70 seconds. The full run took
2,822.984 seconds (47.050 minutes, 0.7842 GPU wall-clock hours): 2,684.161
seconds training, 113.760 seconds OOF inference and 25.063 seconds for setup,
hashing, calibration and figures. Peak allocated CUDA memory was 5,588.151 MiB
for training and 1,958.110 MiB for OOF inference. All checkpoints strictly
reload, both arm initial states and sampling plans match within each fold, all
protected prediction masks for deck, `stay_cable` and tower are exact, RGB is
unchanged, and all 18 frozen artifacts remain hash-exact.

The first independent audit (`audit-20260907-r1`) deliberately stopped because
it required a separately trained SEM-270 control to be bit-exact. That was an
over-strong cross-run reproducibility assertion, not a preregistered quality
gate. The corrected audit (`audit-20260907-r2`) retains the discrepancy as a
warning and passes the actual run invariants: all six current checkpoints
reproduce their saved native predictions exactly, all 438 prediction files,
three split/guard/sampling plans, 18,000 steps, label/RGB hashes, class weights,
boundary counts and frozen artifacts verify. Across the two independent GPU
training runs, zero of 219 SEM-270 control probability arrays are bit-exact;
mean absolute probability difference is 0.00059460, maximum is 0.65875 and
94,300 fused pixels differ (0.03298%). This limits direct bitwise claims across
training runs, but does not invalidate the same-run paired candidate/control
comparison; the candidate also fails multiple baseline-relative gates.

Official-camera audit attempts r1/r2/r3 stopped before rendering on missing
legacy environment dependencies (`colorama`, `plyfile`, then `torch_scatter`).
They are retained as engineering failures. R4 used the already validated
SEM-270 `r1-cuda128` environment and passed four actual trained-model probes:
both fold-0 arms on official COLMAP camera 140 at 1320x989 and 320x240. Frozen
RGB/base semantics, disabled-branch equivalence, input RGB, protected masks,
`SIMPLE_RADIAL` and `COLMAP_CAMERA_FROM_WORLD` handling all pass. This is an
engineering probe without target labels, not hidden quality evidence.

Config SHA-256 is
`7887130cf3ccce8672f0cda123db52eac95be0158fc55cfe6d3e61bd5ba467dd`;
full result SHA-256 is
`fe14b301a6fd371556a79da7cfe048929521fcbea9dc17e3861409d02384c115`;
prediction-seal SHA-256 is
`eebe3b2d9b4cbbb701818f6dedb7ca4a2f1a1ef742ff70bbbda60c14759c49fc`.
The independent-audit and external-audit result hashes are recorded in
`artifact_hashes.json`.

## Disposition

Drop this fixed 2x inner-16px positive boundary-loss candidate and retain it as
a negative/positive trade-off ablation. Do not rescue it by scanning the loss
multiplier, radius, probability threshold, seed or checkpoint on the same
calibration labels. The formal `RGB-030-FULL + SEM-161` bundle and its rollback
chain remain unchanged.

The evidence narrows the next mechanism: one-sided positive boundary emphasis
can recover recall, but does not locate the boundary and therefore gives back
too much precision. If foundation research continues, the next independently
registered mechanism should model both sides of the boundary (for example a
fixed signed-distance/contour target with explicit outside-boundary penalty),
retain the five-class region objective, use a genuinely matched same-run
control, and avoid another scalar multiplier sweep. This is a recommendation,
not an unrun result.
