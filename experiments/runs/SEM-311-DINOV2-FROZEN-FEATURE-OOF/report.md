# SEM-311 — Frozen DINOv2 feature-source experiment

## Registration, 2026-09-08

Status: completed_calibration_gate_failed. Official sources, full paired OOF,
independent engineering and quality recomputation passed; candidate not promoted.
The user chose DINOv2 after reporting DINOv3 access rejection. SEM-310 retains its
engineering-only evidence and blocked status; no DINOv3 quality experiment ran.

- Parents: SEM-160 OOF baseline, SEM-310 engineering framework, SUBMIT-020 freeze.
- Base Git commit: `49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus per-run sealed
  worktree source/config snapshots; existing user changes remain untouched.
- Seed20260805; manifest `data/manifests/contest_dataset_v1.json`; inherited
  `sem_005_oof_spatial_v1`,219 OOF, reused108 calibration/111 check, no74 scoring.
- Candidate: public official `facebook/dinov2-base`, revision
  `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, Apache-2.0 model card. Safetensors
  expected SHA256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`.
  Only official public downloads with token=false; no DINOv3 request or mirror.
- Treatment: frozen pretrained DINOv2 ViT-B/14 versus frozen pretrained MiT-B2.
  Input projections differ by512/768 channels; shared stride16 width128 five-class
  decoder, same decoder initialization, crop/flip sampling, train-fold inverse
  frequency weights, AdamW0.0003/weight_decay0.01,3000 steps per arm/fold.
- Both arms use native896 crops and448 inference stride:896 is divisible by
  patch14/common stride16/MiT stride32, preserving the crop grid without asymmetric
  resizing/cropping. This differs from SEM-310's960 protocol in BOTH arms. No
  cross-run comparison can isolate DINOv2 versus DINOv3 (which has no quality result).
- Frozen final-layer features resampled to common stride16; source architecture,
  pretraining, native stride and input projection count form one feature-source
  treatment. No pure-pretraining or equal-total-parameter claim.
- Output: five-class CE, fixed class4 p>=0.5 fusion on background/foundation only.
  RGB, deck, stay_cable and tower remain exact. All18 frozen artifacts are read-only.

## Prospective gates

Same composite gates as ADR-065, fixed before training: candidate versus SEM-160
mIoU >=+0.30pp, foundation IoU >=+1.50pp, foundation recall drop <=2pp and recall
>=90%; versus same-run MiT control mIoU >=+0.10pp and foundation IoU >=+0.50pp.
At least two spatial folds improve foundation IoU; no fold loses >1pp foundation
IoU or >0.30pp mIoU against SEM-160. Exact RGB/protected classes, encoder hashes,
fresh checkpoint replay and independent official-camera engineering gates required.

Both arms'219 predictions must be sealed before scoring reused108 calibration.
Only calibration pass permits once-only reused111 check. No74, final fitting,
threshold/seed/loss/checkpoint rescue or production replacement in this ID.
This is incremental development OOF with inherited SEM-001/RGB exposure, not
full-chain unseen-camera or fresh blind evidence. Full budget4 hours; disk floor100GiB.

## Execution and results

The runner supports `prepare`, pretrained `smoke` (2 steps per arm),
and `full` (6 heads/18000 steps). The independent audit checks all prediction hashes,
fusion, splits, sampling and pretrained encoder/head replays, with optional real
official140 rendering at native and320×240. No random candidate mode exists.

One read-only source-inspection execution approval timed out before running;
the single permitted retry succeeded, confirmed gated=false and the fixed revision.
No accuracy metrics are asserted until a full registered run finishes.

- Prepare `outputs/sem311/prepare-20260908-r1`:32.406s; all4 official files and
 18 frozen artifacts/219 rendered RGB/219 baseline PNG verified. No token used.
- Pretrained smoke `outputs/sem311/smoke-20260908-r1`:4.815s,2 steps per arm,
  one native OOF prediction each; frozen encoders and fresh reload exact. DINOv2
  has86,580,480 frozen parameters, training CUDA allocated peak495.97MiB; MiT
  has24,196,288 frozen, peak544.21MiB. These are short branch-only allocated
  figures, not whole-process/full-training resource guarantees.
- Independent smoke audit `outputs/sem311/smoke-audit-20260908-r1`:15.228s,
 2 native exact replays and4 actual trained-head official140 probes at native
 989×1320/320×240. RGB, frozen SEM-161, protected classes and disabled branch exact;
 target label decoding0. All18 production hashes remain exact.
- Three new model tests passed in the model runtime. Repository health passed:
 123 configs,204 tests with38 optional-runtime skips. The ten original SEM-310
 runtime source files still match its sealed execution hashes.
- Full fixed invocation: `outputs/sem311/full-20260908-r1`; completed in804.320s
  (13.405min),6 heads/18000 training steps,438 probability maps plus438 fused PNGs.
  Peak per-head training CUDA allocated memory545.39MiB. No calibration-driven retry.

## Full quality result: reused108 calibration only

- SEM-160 baseline: mIoU86.8790%; foundation IoU65.5965%, precision67.6357%,
  recall95.6056% (TP966735,FP462591,FN44435).
- Frozen MiT control: mIoU82.1884%; foundation IoU42.7715%, precision44.1892%,
  recall93.0226% (TP940617,FP1187996,FN70553).
- Frozen DINOv2 candidate: mIoU84.2265%; foundation IoU52.6492%, precision53.7165%,
  recall96.3633% (TP974397,FP839565,FN36773).
- Candidate versus matched MiT: mIoU+2.0381pp, foundation IoU+9.8777pp,
  precision+9.5273pp, recall+3.3407pp. Foundation IoU improves versus this control
  in each of the three spatial folds. This is positive single-seed feature-source
  strategy evidence, not proof of pure-pretraining causality or deployment readiness.
- Candidate versus SEM-160: mIoU−2.6525pp, foundation IoU−12.9473pp,
  precision−13.9192pp, recall+0.7577pp. Foundation FP rises by376974 while TP gains
  only7662. The replacement corrects95724 pixels but harms469062 previously correct
  pixels; it adds478709 foundation pixels and removes94073. Other structural-class
  masks remain unchanged; this failure is not due to prohibiting target-class edits.
- Foundation IoU deltas against SEM-160, folds0/1/2:−2.0799/−13.7710/−22.7009pp.
  Every fold's foundation and mIoU safety gate fails. Recall and matched-control
  superiority pass, but overall baseline improvement and two-improving-fold gates fail.
- Exactly108 quality labels were decoded after both arms'219 predictions were sealed.
  No111 check/74 developer scoring, threshold/loss/seed/checkpoint scan, final fit
  or production change. The108 and111 partitions are reused development sets, not
  fresh blind validation; all comparisons above are within the same108 scope.

## Independent audit and evidence files

- `outputs/sem311/full-audit-20260908-r1`:22.388s; all438 prediction/probability
  records, split guards, sampling/label-source/weight metadata and source hashes
  verified. All6 trained heads independently replay their native held-out probe
  exactly. Four official140 probes exercise actual full-trained fold0 heads at
 989×1320 and320×240; frozen RGB/SEM-161 replay, protected masks, unchanged input
  RGB and disabled equivalence all pass. This engineering audit decodes0 target GT.
- `outputs/sem311/quality-audit-20260908-r1`:3.914s; separate pixel-confusion
  construction reproduces all108 per-view,3-fold and aggregate metrics, corrected/
  harmed counts and prospective gate decisions. It reopens only the already-scored
 108 calibration labels; no additional quality partition is opened.
- Final tests:5 SEM-311 model/grid/gate tests +2 independent metric tests +3 original
  SEM-310 regression tests passed in the actual model runtime. Final lightweight
  repo_health passed:123 configs,208 tests with42 optional-runtime skips; actual
  experiment data was verified via the explicit artifact root, not the absent
  default dataset path. The older whole-runtime dependency/float assertion issues
  recorded by SEM-310 were not changed or claimed resolved.
- Six preregistered visual names produce5 local crop panels;282.png has no foundation
  extent in GT/SEM-160/candidate and the fixed renderer skips that empty extent.
 295.png visibly improves pier separation, whereas162.png still shows widened
  foundation regions. These examples do not override the full108 quantitative failure.
- Config SHA256: `dcf810326c7ec9afc11402b75a8139faa7839f0d05253ba97181e82d7a77513e`.
- Full result SHA256: `568e476a36def2397182ca281eb913851780ca1e089b84257b25957fe718632a`.
- Prediction seal SHA256: `657a334eb3d2412f7ef2ca73c3c25ffd0d756312d23a3dc42c8de01db7b7c306`.
- Engineering audit SHA256: `ae6802e7e6a43c100cc2561d629fd15412f1c9bce920a050149fde4f248e9832`.
- Quality audit SHA256: `696b7ddb9f45fe280a442b1b42987899884e13f35ae62df3250dd2bec7b2574a`.

## Decision

Drop this fixed full-lane replacement recipe; retain the paired feature-source
evidence and all failure outputs. The result does not establish that all DINOv2
uses fail, nor that a particular decoder/weight/threshold is the sole cause.
If continued, a newly registered auxiliary-evidence mechanism should first target
false-positive control and beat the existing baseline with matched controls;
this is a proposal, not an implemented improvement. No automatic continuation or
rescue training is started. Frozen RGB-030-FULL + SEM-161 remains the production
candidate; rollback is simply to omit the independent heads.

Rollback: do not load SEM-311 heads. Existing RGB-030-FULL + SEM-161 and every
fallback are unchanged. Risk: frozen generic features may not resolve narrow
foundation boundaries or cross the rendered/real domain gap; actual folds decide.
