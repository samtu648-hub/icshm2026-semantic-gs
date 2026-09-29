# SEM-270-FOUNDATION-BINARY-OOF

Status: completed_calibration_gate_failed; binary candidate dropped, five-class
control retained as promising development evidence, not promoted.
Date: 2026-09-07. No111 check scoring, no74 developer audit and no final fitting.

The dedicated binary candidate loses4.060pp foundation IoU to SEM-160 OOF and
8.839pp to the matched five-class control. The control improves foundation IoU
by4.779pp and overall mIoU by0.973pp, but loses1.761pp foundation recall and
therefore does not satisfy the registered retention gate either. Neither
replaces frozen SEM-161. This is not a marginal threshold miss for the binary
candidate: its precision, IoU and mIoU also deteriorate.

User authorized the independent foundation/background specialist recommended
after SEM-263. Formal RGB-030-FULL + SEM-161 stays read-only; 18/18 source
artifacts were present and SHA-256 exact at start. All pre-existing uncommitted
changes are preserved; no reset, commit, push or upload.

## Preregistered hypothesis and comparison

The existing foundation verifier uses a five-class model and one-way rejection.
SEM-260 reused calibration indicates foundation IoU 65.596%, precision 67.636%,
recall 95.606%, with 96.753% of foundation false positives actually background.
Test whether a dedicated binary task yields more discriminative foundation
evidence on the same frozen rendered RGB.

Two newly trained arms share SegFormer-B2 pretrained initialization, original
three spatial folds and guards, seed20260805, native960 crops, identical view/
crop/flip sequence, AdamW lr6e-5/wd0.01, bf16, 3000 steps and final-only saves.
Control learns all five classes; candidate learns foundation versus all other
classes. Both use the same train-only inverse-frequency CE formula clipped to
0.05-20. Label grouping, resulting class frequencies and final output width
are intrinsic to the task intervention, not independent hyperparameter sweeps.
Shared initial tensors must match; neither starts from contest-fitted weights.
This is a matched new native960 control, not a replay of historical SEM-101's
768 crop or SEM-150's one-way threshold0.10 rule.

Both arms use native960/480 sliding inference and fixed p(foundation)>=0.5.
In the SEM-160 OOF baseline's 0/4 lane, old foundation may be deleted and any
old background may become foundation, including historical foundation rejects.
Predicted deck/stay_cable/tower masks remain exact. No morphology, hand ROI,
RGB/geometry/opacity/appearance/densification/camera changes or threshold search.
This is replacement classification, not a learned residual; strength0 tests
exact bypass and strength1 is the only deployed setting.

## Gates and access

On both reused108 calibration and111 check: foundation IoU >= baseline+1pp,
precision >= baseline+1pp, recall >= baseline-0.5pp AND >=95%, total mIoU
nondecreasing, candidate foundation IoU > matched control, protected masks and
RGB exact. Both219 prediction sets and configuration precede calibration
label scoring. Check is scored only after calibration passes; no rescue sweep.
Check pass permits separately registered all219 final fitting and a frozen74
developer audit: same relative gates, absolute recall floor90%, exact other
classes/RGB. That remains reused development evidence, not hidden validation.

Training labels for each model are strictly its own train fold, not its OOF or
guard set. The inherited SEM-001 baseline used all219 labels; therefore the
whole pipeline remains incremental OOF, not full-chain blind OOF. Historical
metrics already present in project documents are not new audit results.

Engineering: label/hash and camera-guard isolation, shared training sequence,
class counts/weights, checkpoint strict reload, disabled equivalence, actual
trained branch official COLMAP native/small camera, RGB equality and final18
frozen hashes. Independent verification must recompute prediction protection,
provenance and training weights. Save five-class metrics, per-view failures,
same-view panels, native foundation closeups, time/memory, source snapshots,
config and artifact hashes. Six fixed visual views show both corrected and
harmed crops. Resource floor100GiB; maximum2 GPU wall-clock hours. Smoke is
engineering only, never quality-based selection.

## Execution

Smoke r1 passed (two steps per arm, one outer target, no quality scoring):
shared initialization, strict reload, disabled equivalence, protected classes,
RGB and18 frozen hashes. Runtime4.988s. Seven CPU unit contracts passed.
Full three-fold run completed: six models,18,000 training steps,438 float32
probability maps and438 fused class PNGs, all sealed before108 calibration
scoring. Independent provenance/native replay and trained-branch external
camera audits passed. The smoke remains saved; no training rerun was needed.

## Calibration results

The comparator is SEM-160 OOF on the same reused108 calibration views, not a
direct final/hidden comparison with SEM-161. Values are pixel-aggregated.
Percentage-point deltas refer to the same partition and baseline.

- Baseline foundation IoU65.596457%, precision67.635725%, recall95.605586%;
  total mIoU86.879012%.
- Five-class control: foundation IoU70.375814%, precision73.781392%,
  recall93.844952%; total mIoU87.851967%. Deltas: IoU+4.779358pp,
  precision+6.145668pp, recall-1.760634pp, mIoU+0.972956pp.
- Binary candidate: foundation IoU61.536383%, precision63.568548%,
  recall95.061562%; total mIoU86.054228%. Deltas: IoU-4.060074pp,
  precision-4.067177pp, recall-0.544023pp, mIoU-0.824784pp.
- Binary versus five-class control: IoU-8.839431pp, precision-10.212844pp,
  recall+1.216610pp. Higher recall than the control does not compensate for
  substantial background overprediction.

Binary candidate five-class IoU / precision / recall (percent):

- background:98.240138 /99.559715 /98.668804.
- deck:94.530132 /95.628069 /98.800008.
- stay_cable:89.636149 /94.142319 /94.930720.
- tower:86.328338 /88.865847 /96.798254.
- foundation:61.536383 /63.568548 /95.061562.

The three protected-class pixel masks and metrics are identical in baseline,
control and candidate. Foundation TP/FP/FN are966,735/462,591/44,435 for the
baseline,948,932/337,208/62,238 for control, and961,234/550,888/49,936 for
binary. Full five-class metrics/confusions for each arm and every view are
saved in `calibration_sealed.json` and the final result.

Control adds74,417 and removes217,603 foundation predictions:196,074 changed
pixels become correct,91,313 previously correct pixels become wrong,4,633 are
wrong-to-wrong. Binary adds314,003 and removes231,207:221,950 become correct,
307,041 become wrong and16,219 are wrong-to-wrong. Neither arm is prevented
from making genuine background/foundation changes.

Candidate passes only the absolute95% recall floor and engineering RGB/
protection gates; it fails IoU gain, precision gain, relative recall retention,
mIoU and beating-control gates. Control is not silently substituted for the
failed candidate: it fails recall retention/floor itself and was not authorized
by this protocol for a post-failure check-selection step.

## Spatial consistency

Original folds0/1/2 contribute46/42/20 calibration views. Foundation IoU /
recall (percent), aggregated within each fold:

- Fold0: baseline59.845968 /99.001101; control73.266750 /97.810236;
  binary76.741622 /98.443388.
- Fold1: baseline63.650917 /94.209878; control66.517041 /92.344208;
  binary67.786463 /89.601720.
- Fold2: baseline74.360957 /94.177385; control72.194804 /92.042054;
  binary48.928937 /97.894167.

Binary improves IoU in folds0/1, but its fold2 background-to-foundation count
is335,930 versus baseline90,813 and control94,291. Its overall failure is
strongly spatially concentrated, not evidence that every view deteriorates.
Among104 views with a defined baseline/candidate foundation IoU comparison,
binary improves82 and worsens22; control improves87 and worsens17. Four other
views are excluded from that defined-IoU comparison, not from aggregate metrics.

This new matched control changes both resolution and fusion relative to the
historical SEM-150 setup. Its positive baseline gain cannot be attributed to
resolution alone. The between-arm comparison tests the complete registered
specialization recipe; derived class weights/output width also differ.

## Post-seal diagnosis and actual images

`outputs/sem270/diagnostic-20260907-r1/result.json` reopens only the SAME108
already scored calibration labels, with access hashes, after the failed
decision is fixed. It creates no new model, inference, threshold or candidate.

- Control loses23,658 old true-foundation pixels and recovers5,855, yielding
  net17,803 lost TP. Of the losses,43.89% are within4 pixels of a GT foundation
  boundary,62.01% within8,81.81% within16. Its remaining FN are only39.32%
  within4 pixels; this is not merely a one-pixel edge problem.
- Control removes190,219 background false positives but adds67,655.
- Binary loses24,343 old TP and recovers18,842. It removes203,108 background
  false positives but adds282,698;64.11% of the new background false positives
  lie more than16 pixels from a GT foundation boundary. The far expansion
  is more serious than a small boundary-threshold discrepancy.

The six preregistered views each have a full same-view panel, foundation extent
crop and two native224x128 windows enlarged2x, for24 visual files under
`outputs/sem270/full-20260907-r1/panels/`. Columns: frozen rendered RGB, GT,
SEM-160, five-class control, binary, newly-correct green/newly-harmed red.
Window selection is deterministic maximum corrected/harmed count over a fixed
grid within each fixed view; both kinds are retained, with exact crop boxes
and hashes. GT is display-only after the prediction seal.

Viewed295_corrected: both new models remove false purple foundation fragments
between labeled structures, retaining real foundation portions. Viewed
162_harmed: binary adds false purple regions alongside structures and broadens
another region; the control is less expansive but also removes some real
foundation. These examples support retaining successes and failures, not using
selected panels as proof of universal improvement.

## Engineering evidence and resources

Independent audit `outputs/sem270/audit-20260907-r1/result.json` passed:
six checkpoints reconstructed from original initialization, strict load and
exact native probability replay on first-held views017/001/031; all438
probability maps and438 class PNGs;438 parent RGB/baseline image hashes;
three paired3000-step sampling plans; actual train/OOF/guard separation and
official-camera guard distances; training label hashes, class counts and
derived loss weights; current sources, snapshots and prediction/calibration
seals. Every model has380 changed state tensors. Shared initial tensors match
across arms; this does not mean their learned weights or class-loss values are
identical. Training counts114/128/177, outer prediction counts95/84/40.

Binary negative/positive weights by fold are[0.5047705,20],
[0.5042435,20],[0.5032113,20]. The positive weight hits the preregistered cap;
five-class weights retain distinct deck/stay_cable/tower negatives. Exact
weights/counts are in checkpoint metadata and independent audit results.

`outputs/sem270/external-20260907-r1/result.json` passed all four actual
trained-model probes: two fold0 arms on official COLMAP camera140 at1320x989
and320x240, using `COLMAP_CAMERA_FROM_WORLD` and `SIMPLE_RADIAL`. Frozen RGB
and base semantic replay, disabled-branch equivalence, unchanged input RGB,
and exact protected masks pass. Native control/binary change281/284 pixels;
small control/binary change70/27. No GT/reference RGB is an inference input.
This is one camera at two sizes, not140 cameras or a final/hidden quality audit.

Full run2,775.004s=46.250min=0.770835 GPU wall-clock hours: training2,638.388s,
OOF inference111.708s, remaining24.908s setup/hashing/calibration/figures.
Independent audit9.705s, external audit13.770s and post-seal diagnosis4.513s
are separate from full time.
Peak CUDA allocated training5,584.041MiB; OOF inference1,958.247MiB; external
1,948.536MiB. These are PyTorch allocated peaks, not reserved or total device
memory. Binary/five models have27,348,162 /27,350,469 trainable parameters;
the2,307-parameter difference is the required final output projection.
Hardware RTX5070 12GB; Python3.12.3, PyTorch2.11.0+cu128, CUDA12.8.

Seven new contracts pass in both the repository and actual model Python.
Repository health passes190 tests with28 optional-runtime skips. All18 frozen
artifacts remain SHA-256 exact. No check/developer quality scoring or final
model was produced. Existing uncommitted edits and all failed/smoke outputs
remain; no Git reset, commit, push or upload.

## Provenance, disposition and next experiment

Git HEAD49cdcdb1c9915a9544d568754af95ea20a85ec9c plus the saved dirty source
snapshot; seed20260805; dataset `data/manifests/contest_dataset_v1.json`;
original `sem_005_oof_spatial_v1` with SHA-256
`ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`.
Final result SHA-256:
`fa6809236801834f59b7b255ecea35763fc8ea2e2c3226dce7258167249f5b84`.
Config SHA-256:
`a5a10e615086111b87232377befc6a7d253e00bda579691fbd83220c7ee73ff2`.
Prediction-seal SHA-256:
`036d164d6df533c6965b9d803e00bd0babcb47d907375d8f626badccd0647add`.
Configuration stays byte-identical to preregistration, including its historical
status field. `artifact_hashes.json` indexes configs, six checkpoints,
training sequences, sources, seals, audits and panels; individual OOF PNG and
probability hashes are transitively indexed by the sealed prediction manifest.

Do not promote binary or quietly promote the control. Do not run check/final
or rescan0.5 after this failure. Formal RGB-030-FULL + SEM-161 remains in use;
rollback simply means not loading either research classifier, without changing
any frozen files.

The next worthwhile independent experiment starts from the successful aspects
of the native five-class recipe: retain its multiclass supervision and same
backbone/RGB, but test a single training-side foundation false-negative /
boundary-completeness objective against an otherwise identical CE control.
The control already has4.779pp IoU and0.973pp mIoU headroom, with identifiable
foundation losses around a broader boundary region. Fix the objective and
its scale before any new OOF run; do not sweep thresholds, globally lock all
old foundation, or assume a tiny4px edge-only correction suffices. This is a
proposal supported by calibration diagnosis, not a run or guaranteed gain.
Neither pure resolution causality nor a universal failure of binary models
follows from SEM-270; full-chain and hidden generalization remain untested.
