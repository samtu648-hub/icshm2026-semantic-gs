# SEM-326 — Dense foundation review final consolidation

Status: `completed_developer_gate_failed`, 2026-09-08. Positive developer candidate retained, not promoted. Parents SEM325/SEM161/SUBMIT020; seed20260805; commit49cdcdb1c9915a9544d568754af95ea20a85ec9c plus sealed source snapshots. The immutable config retains its preregistered status; this report and registry record completion.

Test unchanged SEM3254→0 rule with full219 training reference bank against frozen SEM161 on74 reused developer views. All data/label/baseline/split manifests, main gates, spatial groups, external probes and resource budgets are fixed in config before inference. No training or source/threshold sweep. Target GT follows prediction sealing and engineering audit. Formal SEM161 and18 artifacts remain unchanged.

Development PNGs are PINHOLE; external official outputs are distorted. The new external adapter uses internal PINHOLE rendered RGB for matching and maps output pixel centers to PINHOLE before applying the same support rule, preserving original RGB and all other class masks. This coordinate adapter is an interface requirement, not a retuning of the matcher.

Prepare r1 failed before any RGB/label decoding or matching: the inherited OOF access assertion incorrectly forbade74 target roles as well as source roles. r2 corrects the assertion to permit the explicitly registered74 targets while still forbidding every74/guard source. Failure retained; no quality/config/selection rule changed.

External r1 failed before rendering: normalizing the supplied venv Python with `resolve()` dereferenced its symlink to the system interpreter, which lacks Torch. r2 preserves the explicitly supplied venv executable via `absolute()`; no installation or environment/model/quality change. Both parent and child failure records retained;74 native prediction and independent reconstruction already completed separately and are not rerun for this wrapper-only fix.

## Result and decision

The fixed dense reviewer transfers a real positive signal to the final SEM161 pipeline, but does not pass the complete preregistered final gate. On the same74 developer views:

- Five-class mIoU: **83.387017% → 83.682675%, +0.295658pp**; required +0.30pp, short by0.004342pp.
- Foundation IoU: **72.770345% → 74.242876%, +1.472531pp**; required +1.50pp, short by0.027469pp.
- Foundation precision: **78.275843% → 81.475684%, +3.199842pp**.
- Foundation recall: **91.186564% → 89.319965%, −1.866600pp**. Relative loss satisfies the2pp allowance, but absolute recall fails the90% floor by0.680035pp.
- Only11,484 pixels change4→0:8,262 corrected,3,122 formerly correct foundation pixels harmed,100 still wrong (34 deck/10 stay_cable/56 tower GT). Net correct pixels **+5,140**. Foundation FP drops42,328→33,966, while TP drops152,515→149,393 and FN rises14,741→17,863. FP reduction8,362 must not be confused with8,262 fully corrected pixels.

Eleven of14 gates pass. Only `miou_gain`, `foundation_iou_gain`, `recall_floor` fail. The two gain shortfalls are numerically small; **lost true foundation remains the substantive issue**, not evidence that dense correspondence is useless. Keep this positive developer candidate and its unchanged outputs, retain official RGB-030-FULL+SEM161, and do not adjust thresholds or retrospectively relax gates for this ID.

All four preexisting trajectory groups improve (mIoU / foundation IoU delta in pp): seg-007 n29 **+0.038335/+0.190593**; seg-008 n30 **+0.400823/+1.997898**; seg-009 n7 **+0.872011/+4.354364**; seg-010 n8 **+1.132891/+5.643744**. These are the existing segment intersections, not newly invented OOF folds. Fixed140–146 mean per-view mIoU **83.863630%→84.814621%, +0.950991pp**; baseline-ranked worst10% (8 views:061,060,059,062,065,063,057,058) **58.733667%→58.751901%, +0.018234pp**. All group and difficult-subset safety gates pass; worst-view benefit remains very small.

This final result is descriptively stronger than the previously recorded SEM314 DINOv2 final reviewer (+0.087pp mIoU/+0.437pp foundation IoU, recall87.926%), on the same historical developer set. It is not a fresh randomized head-to-head or proof that one backbone is globally superior. Full219 reference bank and SEM161 final baseline migration are jointly tested as a deployment recipe. The74 have been reused historically, so this is development evidence, not blind/hidden-set generalization.

## Execution, coordinates and independent verification

The full219 training bank was hash-sealed before access; source roles exclude all74 developer,7 buffer and100 unlabeled images. Matching uses frozen rendered PINHOLE target RGB, never target real RGB or GT. Source real RGB is distorted; transferred source labels are the existing PINHOLE LabelMe-raster outputs, after source coordinate undistortion. No new training or label generation.

- All222 fixed pairs/444 directions evaluated.170 pairs pass;72/74 targets have a first passing source (rank1=64/74), using8 unique selected sources.080 and157 abstain and remain exact SEM161. Median512-grid reliable coverage including abstentions5.150795%.
- RoMa remains fixed512×512 with original fallback kernel and scoped Pillow overlay, pinned upstream/weight hashes inherited from SEM320; no backbone download, installation or environment replacement. Peak CUDA allocated2508.446MiB/reserved2718MiB. Match stage88.825124s; a separate-process first-pair replay5.115308s is array-exact.
- Geometry stage27.887538s and independently recomputed222-pair audit35.234241s pass; pose-only ranking exact. Native1320×989 prediction33.285895s, independently reconstructed74 maps20.980643s. Reliable native support5,438,826 pixels, source-label out-of-grid count0. Interpolate512 fields at native pixel centers then apply fixed tests; this is not native-resolution matcher inference or a resized binary gate.
- All74 prediction maps were sealed before target GT. Deck/class1, stay_cable/class2 and tower/class3 masks are exact; target RGB bytes and18 frozen artifact hashes are unchanged. Baseline confusion matrices exactly reproduce recorded SEM161 metrics.
- Four actual fresh renderer+reviewer subprocess probes complete in54.055190s, with zero target GT decodes. Native140 twice: each989×1320, source137, reliable215,269/changed238, exact repeat (13.531620s,13.465820s). Small140:240×320, source137, reliable16,779/changed0,11.730857s. Novel official-frame pose:989×1320, source137, reliable229,745/changed203,12.940886s. No out-of-source-label-grid accesses. The small probe verifies exact abstention/no-op, not positive quality at that size.
- Novel camera is named `novel140.png`, with official140 world-to-camera translation+[0.05,0,0]; no target tracks or GT. It is one local pose perturbation, not arbitrary-view coverage. Three known probes have official distorted RGB and baseline semantics exactly equal to frozen saved references. All probes independently reconstruct the new raw-output reviewer, preserve protected classes, and have exact disabled-mode bypass. The matching-only Pillow overlay is removed from renderer subprocesses.
- Only after native/external engineering gates passed, the74 GT were decoded for a single paired quality scoring3.383023s. An independent auditor rereads only those same74, counts via `np.add.at` versus primary histograms and verifies every view, aggregate, group, subset and final decision. Quality-audit engineering status passes while `quality_passed=false`; these are distinct conclusions.

Six fixed panels054/081/140/159/160/183 were retained and visually inspected.140 shows predominantly corrected foundation fringe.183 shows corrected outer fringe together with thin harmed true-foundation parts. Other fixed samples have small scattered changes. These examples motivate a boundary/visibility hypothesis but do not establish where all3,122 harmful deletions originate. Original140/183 headings were cramped; a separate display-only layout adds padding and wrapped headings, verifies exact native image tiles, and reads only sealed composite panels (zero GT decodes/inference/resampling). Original audit code, panels, PNG predictions and scores remain unchanged.

## Provenance and retained artifacts

All paths below are relative to the project unless configured under `ICSHM_OUTPUT_ROOT`/`ICSHM_DATA_ROOT`. Data, outputs, logs and source snapshots stay gitignored. The main config fixes dataset/split/label/trajectory manifests and SHA256; source snapshots record dirty-worktree source rather than relying on base commit alone.

- Config `experiments/runs/SEM-326-DENSE-FOUNDATION-REVIEW-FINAL/config.yaml`: `b3f4f3094bf08bcd59c67fbb610286c9a4a0e757df6d2b8db7a480469f395c18`.
- Dataset manifest `data/manifests/contest_dataset_v1.json`: `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e`.
- Split `experiments/splits/sem_001_scaffold_static_v1.json`: `1764d3281fedc94e03c85f2ffc272e324bb12c0d8dba1dc1955078f71d147854` (219 train/74 developer/7 buffer/100 unlabeled).
- `outputs/sem326/matching-20260908-r2/prepare/result.json`: `b44d12a5723f67fb430e3e7d55a57351d4ac14c159aad8759f016e42ebd15acf`; includes source hashes and full reference bank. Runtime result: `4081713022df648595bf9cdea2d3ce46cc7f52149bb15decc8d534a57014c002`.
- `outputs/sem326/predict-20260908-r1/result.json`: `a6f5b216cb902ca9e8393ffddbedf7ec4c26c7c7f5b1724060d10268c5bfe0f4`; linked `access_seal.json` and `source_snapshot/` freeze the actual reviewer.
- `outputs/sem326/native-audit-20260908-r1/result.json`: `f0b26297ab38b8f31963f7502e83f6bf0a05b885fa0f3b3e8ce92f28c1ac6aad`.
- `outputs/sem326/external-20260908-r2/result.json`: `de61cf23a54d7dca540aa58995df4ce9e1f87975f8ed73954176760dc31e0df4`; includes each probe result/config/renderer/code/output hash.
- `outputs/sem326/developer-score-20260908-r1/result.json`: `f75ff3aaf5dec6faa6960826c5722776d1d053f72ec9b4c3a39a43be5dfa8ee4`; linked score-access seal and scoring source snapshot.
- `outputs/sem326/quality-audit-20260908-r1/result.json`: `5451463bc6053fe9d4486c1f32300f077d6d0794531ba44283beedfa54ff6a70`; same directory `panels/` contains all6 originals.
- `outputs/sem326/panel-layout-20260908-r1/result.json`: `68551371b42d5126e07038f1d7ae54841343265dd2b4bb39f6230c643e1c6ff7`; separate display-only native-pixel copies.
- Retained failures: `outputs/sem326/matching-20260908-r1/prepare/failure.json`, `outputs/sem326/external-20260908-r1/failure.json`, and its `native1/failure.json`. Both implementation causes and narrow fixes are described above; no failed artifact was deleted.

## Validation, remaining risk and rollback

Five new SEM326 unit tests cover known radial coordinate inversion, equivalence with unmodified native review, protected-class/disabled bypass, exact frozen-renderer-only-two-changes contract, and preservation of the venv executable symlink. Combined SEM314 and SEM320–326 tests pass39/39 in the existing runtime. `bash scripts/repo_health.sh` passes133 JSON-compatible YAML configs and258 unit tests (50 optional-runtime skips in system Python); `git diff --check` passes. Original known matcher read-only NumPy warning remains; exact array audits pass. No inference code was altered after its successful seal; panel layout is separate.

Rollback needs no restoration: production RGB-030-FULL+SEM161, SUBMIT020's18 artifacts and SUBMIT030 bundle remain untouched. Do not load the separate SEM326 reviewer or its output directory to use the existing baseline; disabled reviewer also returns the baseline exactly. No Git commit/push or official upload was performed. All-data fit and paper/presentation work remain deferred.

Next: retain this candidate and separately preregister a **training-side harmful-deletion diagnosis**, reusing sealed SEM325 calibration correspondence/outputs. Test whether harmful deletions are associated with source foundation-boundary proximity, match uncertainty or visibility mismatch; do not claim any mechanism proven by the six developer panels. Use that evidence to choose at most one new inference-available protection variable, then freeze it before check/final evaluation. No74 threshold/source/ensemble sweep, no immediate new backbone, and no need to repeat the completed environment setup. This next experiment has not run in SEM326.
