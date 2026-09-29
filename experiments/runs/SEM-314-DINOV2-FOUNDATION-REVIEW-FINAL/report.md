# SEM-314 — Final frozen DINOv2 foundation reviewer

## Outcome — 2026-09-08

Status: `completed_developer_gate_failed`. Engineering and independent quality
recomputation passed; the candidate failed the registered quality gates and is
NOT promoted. `RGB-030-FULL + SEM-161` and the existing submission bundle remain
unchanged. The config's `preregistered` field is intentionally retained as part of
the immutable pre-run contract; this report and the registry carry the outcome.

One fresh all219 head completed exactly3,000 steps, followed by sealed74 inference,
the engineering gate, one frozen74 decision and an independent recomputation of
the same74. There was no threshold/loss/seed/checkpoint sweep, rescue training,
extra final MiT head, production replacement, official upload or Git push.

### Same74 quality result

All values below belong to this experiment, seed20260805, the base commit and
source snapshot identified below, and `sem_001_scaffold_static_v1`. These74 are
historically reused developer views, not a new blind or official hidden test.
Main mIoU is calculated from pooled confusion matrices including background;
difficult-subset mIoU is the registered mean of per-view scores.

- mIoU: SEM161 **83.387017%** → SEM314 **83.474097%**, **+0.087080pp**;
  fails the minimum **+0.30pp** gain.
- Foundation IoU: **72.770345% → 73.207788%**, **+0.437443pp**;
  fails the minimum **+1.50pp** gain.
- Foundation precision: **78.275843% → 81.389570%**, **+3.113727pp**.
- Foundation recall: **91.186564% → 87.926293%**, **−3.260272pp**;
  fails both maximum2pp loss and absolute90% floor.
- Foundation TP/FP/FN: baseline152515/42328/14741, candidate147062/33627/20194.
  FP falls8701, but5453 previously correct foundation pixels are lost.
- Of14154 removals,6114 correct an error,5453 harm a correct pixel, and2587
  remain wrong (true deck/`stay_cable`/tower: incorrect foundation→incorrect
  background). These2587 reduce foundation FP but are not five-class corrections.
- Per-view mIoU:32 improve,42 worsen,0 tie. Class1/2/3 predictions and their
  metrics are exact; the only authorized transition is4→0.

Spatial groups were fixed from the preexisting trajectory metadata before scoring,
not chosen after inspecting errors. These are four intersections within one
historical holdout, not four OOF folds. Each line gives mIoU/foundation-IoU delta:

- seg-007,29 views054–082: **+0.083589/+0.419562pp**, both safety gates pass.
- seg-008,30 views083–090 and140–161: **+0.308335/+1.543990pp**, both pass.
- seg-009,7 views169–175: **−2.275009/−11.369345pp**, both fail.
  Foundation IoU62.252252%→50.882908%, recall90.150033%→64.535769%.
- seg-010,8 views176–183: **+0.013528/+0.067108pp**, both pass.

Fixed140–146 mean mIoU83.863630%→82.971687%, **−0.891943pp**: fails
the maximum0.30pp loss. The baseline-ranked worst10% (8 views061/060/059/062/
065/063/057/058) improves58.733667%→62.334353%, **+3.600686pp** and passes.
Its improvement does not override the other quality failures.

### Engineering, access and resources

-219 training labels and RGB files independently verified; the class counts,
  clipped inverse-frequency weights and sampling plan exactly match the contract.
  All74 developer views excluded from training;7 guard views and100 unlabeled
  images not used for supervision. No new108/111 scoring or selection.
- Frozen official DINOv2 encoder unchanged; fresh final-head reload exact.
  All74 float32 probability maps and74 review PNGs sealed before developer labels.
- Independent74-output reconstruction matches the fixed old-foundation/p4<0.5
  rule. Native actual-final-head replays054 and159 are bit-exact.
- Actual final head executed in three fresh official-COLMAP140 processes:
  native1320×989 twice and320×240 once. Both native probability/prediction hashes
  match exactly, with142 changed pixels each. Small probe changes0 pixels: this
  proves execution/equivalence, not small-resolution quality improvement.
- Official frozen RGB and SEM161 baseline replay match the saved reference;
  reviewer bypass, class1/2/3 protection and all18 frozen production artifacts pass.
- Independent74 per-pixel confusion matrices, transition tensor, pooled/group/
  difficult-subset metrics and every gate agree with the first quality decision.
- Training loop142.454s; whole training stage149.915s; prediction19.104s;
  engineering35.655s; first quality1.404s; independent quality plus panels2.630s.
  Peak head-training CUDA **allocated** memory495.719MiB is not total process or
  device memory. The registered1 GPU-hour budget and100GiB free-disk floor held.

### Fixed visual evidence and presentation-only correction

The registered054/081/140/159/160/183 panels were saved during independent quality
recomputation. Crops are the union of GT/old predicted foundation plus the fixed
16px margin; GT is diagnostic only and never influences predictions or gates.
Each panel contains frozen RGB, GT, SEM161, SEM314 and a change map (green correct,
red harmed, amber still wrong). The original narrow140 crop exposed overlapping
captions. A separate `scripts/layout_sem314_panels.py` reflows the same five
native pixel arrays, without resizing, into readable rows. It opens only sealed
panel images, not labels, predictions or model inputs. All30 body arrays survive
PNG save/reload exactly. Original audit images/code/config/snapshots are untouched.
Reflowed140 and081 were visually inspected; both captions and the change legend
are readable. This presentation-only step is not a new quality experiment.

Artifacts, all Git-ignored under `outputs/sem314/`:

- `train-20260908-r1/`: final head, contract, sampling plan, immutable source snapshot.
- `predictions-20260908-r1/`: sealed74 probabilities, PNGs and manifest.
- `engineering-20260908-r1/`: independent audit and native1/native2/small probes.
- `developer-20260908-r1/`: failed frozen decision,74 scores, groups and access ledger.
- `quality-audit-20260908-r1/`: independent recomputation and6 original panels.
- `panel-layout-20260908-r1/`:6 readable panels and native-pixel roundtrip manifest.

### Provenance

Base Git commit: `49cdcdb1c9915a9544d568754af95ea20a85ec9c`, with user-owned
dirty worktree preserved. Exact executed source hashes and copies are sealed in
`train-20260908-r1/result.json` and `source_snapshot/`; the uncommitted base commit
alone is not claimed to reproduce the new code. Decoder width128,896crop/448stride,
AdamW0.0003/0.01,3k steps,five-class CE and p4<0.5 remain the parent recipe.
Official model provenance and license/revision are inherited from SEM311's config.

SHA256:

```text
config             53573ce126035020582d893112139e812803e79e3ba5595d1c320ab97c233344
final head         4a88a741aecf9e2961991e66d0a4b5d868159636ae52e141fe3950d0b06106ef
head contract      7b999ec318feefa5135004d8dd5001d27c9412a4650feadd888ca85b254588f6
train result       1386af22d850acbe96d19a96376c7aca17776768d056a68288629d6637a8751e
prediction result  a3a1972479b5505426d2057e14175e33d03daf9dc661dc08f67986c3f10f3782
engineering result 02430e8a1664b91ae0ec47b3c4c832f961423f7ed6aa628cfc66878f68bd6dc3
developer result   b96c6e7885843e05e49422ceed8927f7506c85ec201d5c3cabc1ff3247169941
independent result 370313059840f94426f9e3f9f81863326a527ae42dba80d7fdf2cc7574c0a805
panel layout       6030e7fc275c30a0405b867b335251825b517b704cdb9796656e3531a88275e8
```

### Validation and decision

New final protocol unit tests6/6 pass. The model-runtime targeted regression suite
passes27/27 without skips: final protocol, complementarity, reviewer, SEM313
protocol, SEM311 quality, frozen DINOv2 and frozen-feature modules. Repository
health passes126 configs and225 tests with46 optional-runtime skips; optional
default-path dataset check is skipped, while this experiment explicitly verifies
the actual artifact-root inputs. `git diff --check` passes. This is not a claim
that the older model-environment whole-suite dependency issues disappeared.

Drop this fixed final reviewer recipe; preserve the head and negative evidence.
SEM313 remains positive incremental OOF evidence but did not establish stable
transfer to this final composition. Training coverage, held-out view distribution
and inherited baselines differ, so this result does not isolate a single causal
failure in the backbone, head or threshold, nor disprove every DINOv2 application.
Do not tune on74 to rescue it. Next research returns to the separately planned
dense-correspondence source/license/dependency/12GB preflight, not another run of
this final recipe. All400 RGB/all300-label fitting and paper/presentation work
remain deferred until method selection.

Rollback is already in effect: continue the unchanged SEM161 interface and omit
the optional reviewer wrapper. No file removal, bundle replacement or restoration
operation is necessary; all failed candidate outputs remain recoverable locally.

## Registration — 2026-09-08

Original pre-run status: preregistered; no final training or developer scoring yet.
User authorized continuation after SEM313. Parents SEM313/311/161 and SUBMIT020.
Base commit `49cdcdb1c9915a9544d568754af95ea20a85ec9c`, seed20260805,
manifest `data/manifests/contest_dataset_v1.json`, split
`sem_001_scaffold_static_v1`:219 train,74 reused developer,7 guard excluded.

One fresh head on the official frozen DINOv2-B/14; keep SEM311's width128 decoder,
native896/448, five-class inverse-frequency CE, AdamW0.0003/0.01,3,000 steps and
SEM313's fixed review p4<0.5. Recompute weights on all219 as required by final
training; no OOF checkpoint continuation or full-backbone fine-tuning. The only
new fit is DINOv2; the strong baseline is unchanged SEM161. Do not claim a new
matched MiT comparison on74. No architecture/loss/seed/threshold sweep.

Seal head and source/config contracts before74 inference; seal all predictions
before labels. Engineering must verify native probability replays, actual final
head at official COLMAP140 native twice and320x240 once, reviewer bypass, protected
class1/2/3 and RGB exactness, and18 immutable production artifact hashes.

Pre-fixed developer gates relative to SEM161: mIoU>=+0.30pp, foundation IoU>=+1.50pp,
recall loss<=2pp and absolute recall>=90%. Every existing trajectory-segment
intersection with74 must retain foundation IoU within1pp and mIoU within0.30pp.
Fixed140–146 and baseline-ranked worst10% mean mIoU may lose at most0.30pp.
These are segment checks within a single historical holdout, not new OOF folds.
Fixed panels054/081/140/159/160/183 record both improvements and damage; no visual
selection changes predictions or gates.

Failure stops without retry/tuning or promotion. Success may register an additive
local development candidate only: no overwrite of SEM161, no bundle replacement,
official upload or Git push. The74 are historically reused, not a fresh blind test.
All400 RGB/all300 supervised final fitting remains out of this experiment.

Budget:1 GPU-hour and100GiB disk floor. Rollback: omit this optional reviewer.
