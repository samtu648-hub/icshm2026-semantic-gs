# SEM-300-FOUNDATION-SIGNED-DISTANCE-OOF

## Status

Completed on 2026-09-07 as `completed_calibration_gate_failed`. The engineering
smoke passed, then the matched three-fold OOF run completed all six 3,000-step
arms and 438 held-out predictions. It stopped at the preregistered
108-view calibration gate: check, developer, and final fitting remained
closed. `RGB-030-FULL + SEM-161` and all frozen artifacts remain read-only.

## Hypothesis

SEM-270's native five-class control improves foundation IoU and precision but
loses recall. SEM-280 recovers some recall by up-weighting true pixels inside a
16px boundary band, but gives back too much precision because it does not teach
the model which side of the boundary is outside foundation. This experiment
keeps five-class context and makes that distinction explicit through a signed
distance target: positive inside foundation, negative outside, clipped at 16px.

## Single-variable protocol

Both arms use the same new five-class SegFormer wrapper, original pretrained
initialization, spatial folds, guards, seed, crop/flip sequence, optimizer,
3,000-step budget, class weights, 960/480 inference, fixed class-4 threshold
0.5 and background/foundation-only fusion. Deck, `stay_cable`, tower and RGB
remain exact. The control auxiliary loss weight is zero; the candidate alone
uses the fixed signed-distance SmoothL1 weight 0.5. The distance prediction is
training-only, so inference does not introduce a second threshold or boundary
post-process.

The candidate must improve foundation IoU and precision by at least 1pp against
SEM-160, retain recall within 0.5pp and at least 95%, not reduce mIoU, recover
at least 0.5pp recall against its same-run control while retaining that
control's foundation IoU/mIoU, and keep every spatial fold within 1pp of its
SEM-160 foundation IoU. It must also preserve classes 1/2/3 and RGB exactly.
Failure closes check, developer and final fitting without a retry or parameter
scan.

## Boundary

All 219 OOF predictions and both configs are sealed before the already reused
108-view calibration score. The unopened 111-view check is decoded only if all
calibration gates pass; the 74-view developer labels remain closed until a new,
separate final registration. This is incremental OOF evidence, not hidden-view
generalization.

## Execution and integrity

The valid full run is `outputs/sem300/full-20260907-r1`, with seed 20260805,
config SHA-256 `47aa1585aab8371e5d4accb073d3e7f4a69ee253da0dbaaf334c7d757dd79b1b`,
result SHA-256 `303c6483e721aa46b8600337b5fa9b03b7784d6f6714df670975e5b1babcd071`,
and prediction seal `b5ec6c6adbd80deec922e98747a04f76c10497ffb2255acc929d33a277b48e56`.
All six checkpoints reloaded exactly; the 18-artifact freeze manifest was
identical before and after (`53ca7d469eed2666cd1a5d3bfd8c4374275daa6e173cddd62d88a4d410794815`).
RGB and the protected deck, `stay_cable`, and tower pixels were exact.

Two later duplicate launch attempts are retained only as engineering failures:
`r2` exited before loading the runner because of an incorrect working
directory, and `r3` OOMed while the valid `r1` already held the GPU. Neither
attempt produced a quality result, accessed target labels, or changed the
valid run.

## Calibration result and decision

All 219 OOF predictions were sealed before decoding exactly the already reused
108 calibration labels. Relative to the SEM-160 protected baseline, the
signed-distance candidate reached foundation IoU/precision of
0.675643/0.720129 (+1.968/+4.377pp) and mIoU 0.872820 (+0.403pp), but recall
fell to 0.916227 (−3.983pp, below both the 95% floor and the 0.5pp retention
limit). The corresponding same-run five-class control reached foundation
IoU/precision/recall 0.696864/0.729333/0.939952 and mIoU 0.877117. Thus the
candidate versus control changes were −2.122/−0.920/−2.372/−0.430pp.

The candidate also failed the per-fold safety gate: its foundation IoU versus
SEM-160 was +13.216pp, +0.052pp, and −7.153pp in folds 0, 1, and 2. Its
improvements in foundation IoU and precision versus SEM-160 do not compensate
for recall loss, matched-control regression, or the unsafe third fold.

The failed checks are candidate recall versus control, candidate IoU/mIoU
retention versus control, baseline recall floor/retention, and per-fold safety.
Foundation IoU/precision versus baseline, nondecreasing baseline mIoU, RGB,
and protected-class exactness passed. Therefore the fixed signed-distance
auxiliary loss is dropped; no loss-weight, truncation-radius, threshold, or
checkpoint scan is allowed under this experiment. Do not decode the 111-view
check set, the 74-view developer set, or train a final model from SEM-300.
