# SEM-313 — DINOv2 foundation review, matched OOF

## Registration — 2026-09-08

Status: completed_calibration_and_check_passed_pending_final. The following
registration preceded new candidate export and new quality scoring.
Parents SEM-312/311/160 and SUBMIT-020; commit
`49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus sealed sources/config;
seed20260805 inherited from the six frozen heads. Manifest
`data/manifests/contest_dataset_v1.json`; `sem_005_oof_spatial_v1`.

SEM-312 found that DINOv2 additions correct23,170/harm453,554 pixels, while
removals correct72,554/harm15,508 (6,011 remain wrong in another class).
Removal correct/harm pairs in folds0/1/2 are54,393/6,553;16,558/7,913;
1,603/1,042. Only the DINOv2 removal direction passes its prospective research
screen. This motivates a minimal inference-only auxiliary reviewer, not a learned
meta-head: there is no new fit on calibration and no extra teacher/OOF leakage.
The108 were used to choose the mechanism, so their next score is NOT independent.

Keep an existing baseline foundation prediction only when the given frozen head
has p(foundation)>=0.5; otherwise change it to background. All baseline background,
deck, stay_cable and tower pixels are unchanged. Both arms have identical output
authority and threshold; the only paired treatment is DINOv2 vs MiT evidence.
Unmodified SEM-160 is the strong baseline. This deliberately tests *review*, not
full0/4 replacement or whether suppressed foundation false negatives can be recovered.
No bin, radius, agreement rule, confidence threshold or checkpoint is selected.

Retain ADR-065's original new-branch composite gates: versus SEM160 mIoU>=+0.30pp,
foundation IoU>=+1.50pp, recall loss<=2pp and recall>=90%; versus matched MiT
mIoU>=+0.10pp and foundation IoU>=+0.50pp. At least two folds improve foundation
IoU, no fold loses>1pp foundation IoU or>0.30pp mIoU. Protect RGB and class1/2/3.
Seal all438 outputs, audit438 independently, replay six actual parent heads and
four actual reviewer official140 native/small probes before108 calibration.
Only a complete pass permits one reused111 check. No74, fitting or production
replacement even after success. No post-result rescue.

Risk: the reviewer can only trade false positives for recall; benefits may be
dominated by fold0. Development selection and inherited upstream exposure remain.
Rollback: do not load the independent reviewer; frozen SEM161 is unchanged.

## Execution and quality

Exactly one deterministic matched pair was built from the sealed six parent
heads, with zero training and no threshold/seed/checkpoint search. Build5.640s,
independent engineering26.517s, calibration3.178s, calibration recomputation4.696s,
once-only check3.292s and check recomputation4.922s. All runs completed without
failed execution or a rescue retry.

On the108 reused calibration views:

- SEM160 baseline mIoU86.8790%, foundation IoU65.5965%, precision67.6357%, recall95.6056%.
- DINO review mIoU87.4041%, foundation IoU68.1787%, precision71.2395%, recall94.0719%.
  Versus baseline: +0.5250/+2.5823/+3.6037/−1.5337pp respectively.
- Matched MiT review mIoU86.8472%, foundation IoU65.4294%, precision69.7943%, recall91.2756%.
  DINO beats it by mIoU+0.5569pp and foundation IoU+2.7494pp.
- DINO fold0/1/2 foundation IoU deltas versus baseline:+6.4709/+0.7840/+0.0842pp.
  All aggregate and spatial gates passed; independent recomputation passed
  before opening this run's check quality labels. This108 chose the mechanism.

On the111 reused check views:

- SEM160 baseline mIoU86.8315%, foundation IoU65.4920%, precision66.1045%, recall98.6049%.
- DINO review mIoU87.6428%, foundation IoU69.4725%, precision70.6791%, recall97.6017%.
  Versus baseline: +0.8112/+3.9805/+4.5745/−1.0032pp respectively.
- Matched MiT review mIoU86.8123%, foundation IoU65.3829%, precision68.2944%, recall93.8787%.
  DINO beats it by mIoU+0.8304pp, foundation IoU+4.0896pp, recall+3.7230pp.
- DINO fold0/1/2 foundation IoU deltas:+7.1125/+2.4619/+0.3801pp. Every fold
  improves, though gains are not uniform; fold0 still contributes most.
- Foundation FP falls594,462→476,058 (−118,404), TP falls1,159,348→1,147,553
  (−11,795), FN rises16,403→28,198. Of130,199 removed pixels,110,483 become
  correctly background,11,795 true foundations are lost, and7,921 remain wrong
  as background instead of foundation. Neutral changes lower foundation FP but
  do not constitute correct five-class recovery.
- Every pre-registered quality gate passes. Independent per-view/fold/aggregate
  confusion-matrix and three-axis transition recomputation exactly reproduces
  all111 counts, metrics and gate decisions. No74 developer labels or final fit.

## Engineering and interpretation

All438 review PNGs independently reconstruct from the original sealed parent
PNGs (separate from the probability-based builder). The inherited original
audit freshly checks438 parent probability/PNG records and replays all six actual
heads at native resolution. Official COLMAP140 RGB and frozen SEM161 replay
exactly at989×1320 and320×240. Four additional actual-head reviewer probes
execute the NEW rule: native MiT/DINO change338/454 pixels; both small probes
change0 pixels, so small probes prove execution/equivalence, not nonzero benefit.
Reviewer disabled mode is exact, and class1/2/3 plus RGB remain exact.
All18 frozen artifacts remain unchanged. All outputs are local and Git-ignored.

Twenty-one targeted tests passed in the actual model runtime:11 new diagnostic/
review/access-guard tests plus10 parent model/quality regressions. Failed
calibration and missing engineering tests explicitly verify zero label decoding.
Final `bash scripts/repo_health.sh` passed:125 configs,219 tests with45 optional
runtime skips; secret/upload/link/shell checks passed. The targeted21 model-runtime
tests above passed without skips. The default dataset directory is absent, so
repo_health skipped that optional check; actual experiment inputs were checked
against the explicit artifact root. `git diff --check` passed. Original SEM311
config/result hashes remain exact; no pre-existing user edits were discarded.

Keep the fixed DINOv2 auxiliary review as a passed OOF research candidate, NOT
as a silently installed production upgrade. SEM311's full replacement remains
failed. We chose this mechanism using108 and111 is historically reused; inherited
SEM001/RGB exposure and one parent seed prevent full-chain blind/NVS claims.
The reviewer cannot repair false negatives or recognize new foundation pixels.
No further108/111 tuning is allowed. A single final-head fit and a separately
registered frozen74 development audit would be needed to compare with SEM161;
all400/all300 final fitting is still a later, separately registered workflow.

## Evidence fingerprints

- Config: `54a1ab16be99d41ee28e5d5fbebce324594d978efee82f3d0ce3f4367e8f98f2`.
- Build `outputs/sem313/build-20260908-r1/result.json`:
  `8edb653a837f5d5bfc0d58a91d43d826557b0d81ea46aec4c508bac96c237a22`.
- Engineering `outputs/sem313/engineering-20260908-r1/result.json`:
  `8432ec128c7e1f9c9e757907a663f6d90840db6995ee88ebd812a03151f445b3`.
- Calibration `outputs/sem313/calibration-20260908-r1/result.json`:
  `feae3364fb299ef50f8ec74fdb5f931b3e18f485eef269b963bb639cce423360`.
- Calibration audit `outputs/sem313/calibration-audit-20260908-r1/result.json`:
  `afe3a07cb878e663810cf4395a998a2e346ca75f3856f11c72ae094964e5441c`.
- Check `outputs/sem313/check-20260908-r1/result.json`:
  `c213cc12419da9d609df811af4c0cad1ee6eb32d9525990c00e1ffbd637fbc91`.
- Check audit `outputs/sem313/check-audit-20260908-r1/result.json`:
  `07e156a036a8000017b07871f7131850020109d4bc5daf81ab257adadf6cb74e`.
