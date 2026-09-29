# SEM-310 — Frozen DINOv3 feature-source experiment

2026-09-08 subsequent access update: the user supplied a successful local read-token
login screenshot, then reported that the model access request was rejected. The
earlier anonymous401 result remains unchanged historical evidence; no new gated
download is attempted. The user chose a separate SEM-311 DINOv2 experiment.
SEM-310 stays blocked, with original config, ten runtime source hashes and sealed
smoke/audit outputs unchanged. Access rejection is not a model-quality failure.

## Registration (2026-09-08)

Status: engineering smoke and independent camera audit passed; formal pretrained
DINOv3 experiment blocked on official model access. No quality result, candidate
promotion or final fit exists.
User authorized execution after choosing score plus mechanism evidence, the
current RTX 5070 12GB, and prospective small recall tradeoffs for stable mIoU gains.

- Parents: SEM-160, SEM-270 and SUBMIT-020; seed 20260805.
- Git base: `49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus each invocation's
  content-addressed worktree source/config snapshot (existing dirty work preserved).
- Data: `data/manifests/contest_dataset_v1.json`; spatial split
  `sem_005_oof_spatial_v1`, all219 incremental OOF, reused108 calibration/111 check.
- Candidate: official `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`; frozen MiT-B2 matched control.
- Sole primary treatment: frozen pretrained feature source. Channel counts,
  native stride and input projection parameter counts differ intrinsically;
  this is not a pure pretraining causal test or an equal-total-parameter claim.
- Common stride16, width128 five-class decoder; matched decoder initialization,
  train-fold labels/class weights, native960 crops/flips, AdamW and 3000 steps.
  Online no-grad extraction preserves crop context. No backbone updates.
- Inference: native960/480, fixed class4 probability >=0.5, mutable background
  and foundation only; deck, stay_cable, tower and RGB remain exact.

## Prospective gates and stopping

Against SEM-160: mIoU >=+0.30pp, foundation IoU >=+1.50pp; foundation recall
loss <=2.00pp and absolute recall >=90%. Against the matched control: mIoU
>=+0.10pp and foundation IoU >=+0.50pp. At least two spatial folds must improve
foundation IoU; no fold may lose >1.00pp foundation IoU or >0.30pp mIoU against
SEM-160. RGB/structural protection, immutable encoders, strict reload and
independent engineering audit must pass. These gates do not rehabilitate old
failed experiments.

Seal both arms' all219 predictions before calibration scoring. Calibration pass
permits one reused111 check; failure stops without threshold/loss/seed rescue.
No74 developer scoring or final fitting in this ID. This is inherited/incremental
OOF, not fresh blind or full-chain unseen-camera validation. Budget <=4 GPU wall
hours per full invocation; disk floor100GiB. Failure outputs are retained.

## Execution modes and source access

`scripts/run_sem310_frozen_features.py` accepts CLI artifact/model/output roots:

- `prepare`: verify frozen18, RGB219/baseline219, folds and official source hashes;
  download only from the pinned official repository using local account access.
- `architecture-smoke`: two steps per arm and one held-out native inference each;
  real pretrained MiT-B2 plus explicitly RANDOM DINOv3 architecture. This tests
  memory, tensor interfaces, freezing and reload only; it cannot test DINOv3 quality.
- `smoke`: the same engineering checks with approved official pretrained DINOv3.
- `full`: six trained heads,18000 steps,438 OOF outputs and gated quality scoring.

Anonymous official access returned HTTP401 and no local Hugging Face credential
was configured at initial inspection. User was asked to obtain access/login or
provide an approved local weight directory; no token is requested in chat, no
mirror/access-control workaround is used, and no dependency environment is changed.

## Results, rollback and remaining risk

Actual results on2026-09-08 (all outputs Git-ignored):

- `outputs/sem310/prepare-20260908-r1`: frozen18, rendered RGB219, SEM-160 PNG219,
  split and camera guards verified. Official pinned download stopped at config.json,
  HTTP401, token_configured=false. No quality training. Result SHA256
  `8e1e545baee5a519883f5a59aa05500db5b46bcd2e1f2da216f18f77a78021c1`.
- `outputs/sem310/architecture-smoke-20260908-r1`:5.367s, fold0 two training steps
  per arm, exactly202.png and247.png training labels, one held-out native prediction
  per arm. Random DINOv3 has85,660,416 frozen parameters/395,013 trainable head
  parameters; MiT has24,196,288 frozen/362,245 trainable. Training CUDA allocated
  peaks468.28MiB/672.37MiB respectively; these are short isolated branch allocated
  memory figures, not whole-process or full-training GPU capacity guarantees.
  Backbone state hashes remained unchanged, common decoder initialization/labels/
  sampling/weights matched, and fresh head reloads were exact. Result SHA256
  `b678332e26d507ee6b37e369b2d8f97b27fe5b3e97da25e24c30b0351ff200d1`.
- `outputs/sem310/architecture-audit-20260908-r1`:15.725s, independent process
  verified two sealed prediction/probability pairs and exactly replayed both native
  predictions. Four trained-head probes (two arms ×989×1320 and320×240) used
  freshly rendered official140 COLMAP camera metadata. Frozen RGB/SEM-161 replay,
  class1/2/3 protection, unchanged RGB input and disabled-branch equivalence passed.
  Reference images used only for equality assertions are frozen rendered outputs,
  never real target photographs. Zero target labels decoded. All18 frozen hashes
  remain exact. Result SHA256
  `e2b835e6dec9a8478f99a5b68159d6b29c9825c8e246e546aa116d4c759954a0`.
- Configuration SHA256:
  `f9255cd79cd1176344bf821b818850880a2589808eb90a433063d9519445aec0`.
  The MiT loading-info probe separately confirmed no missing/mismatched encoder
  weights; only unused classifier.weight/bias were unexpected.
- New model unit tests:3/3 passed in the training Python runtime. Repository health:
  passed,122 configs,201 tests with35 optional-runtime skips; default dataset path
  absent, while actual experiment data was verified via the explicit artifact root.
- Additional whole-suite run in the model runtime:204 tests,1 failure,4 errors,
  9 skips. Unrelated existing failures were OpenCV without ximgproc, three test
  modules importing absent pytest, and an exact float32-vs-Python0.4 assertion in
  test_semantic_postprocess. No frozen dependency or unrelated test was changed;
  this run is not represented as an all-green full-runtime suite.

No calibration,111 check or74 developer quality metrics were computed. This
architecture smoke is not evidence of pretrained DINOv3 accuracy. Formal smoke
and the six-head18000-step comparison remain unstarted until official access.
Frozen production remains RGB-030-FULL + SEM-161. Rollback is to omit the independent
SEM-310 head; no production file is overwritten. DINOv3 may lack required boundary
localization, and spatial fold failure remains a stopping condition despite gains.

For continuation, obtain official access and log in locally (never paste a token
into chat), then rerun `prepare` in a NEW output directory; use `smoke`, `full`
and `scripts/audit_sem310_frozen_features.py` with explicit CLI roots after source
verification. Preserve the blocked prepare and random-weight smoke records.
