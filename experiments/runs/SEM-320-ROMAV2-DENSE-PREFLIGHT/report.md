# SEM-320 — RoMa v2 dense-correspondence preflight

## Outcome — 2026-09-08

`completed_engineering_passed_with_access_wording_deviation`. Official sources,
complete checkpoint, fixed512 runtime and fresh-process replay pass. An overly
broad access-policy sentence is documented below; this is NOT reported as a
deviation-free protocol pass. No semantic candidate or production change.

- Author source commit `95c9968145c8906b7b59383258e9f73b02853d89` and DINOv3
  source `adc254450203739c8149213a7a69d8d905b4fcfa` verified as clean local
  checkouts;49 and163 tracked source files respectively hashed before execution.
- Official complete `romav2.0.1.pt`:1,095,883,548 bytes, SHA256
  `1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7` matches
  the author release metadata and downloaded bytes. No HF token/request, gated
  backbone weight download or backbone extraction. Local strict state loading
  and model execution deny unexpected network retrieval.
- Runtime uses the upstream native PyTorch correlation fallback, not a newly
  compiled fused kernel. Pillow12.0.0 is a Git-ignored import overlay; the original
  environment retains Pillow10.2.0. Torch2.11.0+cu128, torchvision0.26.0+cu128,
  numpy1.26.4, einops0.8.2, rich14.3.4 and tqdm4.67.3 are read-only existing packages.
  This source-path/overlay setup intentionally omits Linux fused-local-corr; do
  not describe it as a standard dependency-complete pip installation.
- Source plan: fold0 target017/source015 (393 shared released tracks), fold1
  target001/source046 (12), fold2 target032/source029 (51). The target is always
  the lexicographically first calibration target in its fold; sources maximize
  shared tracks within that fold's train, with distance/name ties. No visual or
  confidence-based reselection occurred.
- Fixed official fast512/512, no high-resolution pass, no compile, no learned
  parameter changes. Three pairs executed forward and reverse plus one rendered
  identity probe; a fresh process executed the first forward again. All8 outputs
  have finite512×512 dense warp/overlap, and replay arrays are exactly identical.
- RTX5070 runtime peak CUDA **allocated2497.295MiB (2.439GiB)**,
  **reserved2716MiB**. These are process allocator measurements, not total board
  usage; the pre-launch device query showed1555MiB already occupied by other work.
  The primary stage took6.359s, including2.464s model loading; first forward0.399s,
  remaining warm directions/self0.099–0.103s. Independent replay stage5.208s.
  These are warm small-probe timings, not end-to-end native-resolution throughput.
- Self-map error at the512 grid: median0.073891px/P90=0.201491px. It is a sanity
  measurement, not a calibrated cross-view accuracy guarantee. Original native
  RGB bytes/arrays and all18 production artifacts remain unchanged.

### Access wording deviation and remaining limitations

The immutable config correctly prescribes same-fold training sources, but its
access_policy also says no111 check images at all. Two sources,046/029, belong
to that historical identifier partition. They were decoded as permitted same-fold
training-source observations, never as held-out targets or labels. This conflicts
with the broad sentence and is explicitly retained as a wording/scope deviation;
the original config/source snapshot is not rewritten to hide it. Actual access:
rendered017/001/032 and real015/046/029; no labels of any kind, real target RGB,
74 developer, guard or unlabeled images decoded. SEM321 registers its source
interpretation explicitly instead of retroactively declaring all111 images unread.

Source selection uses **released target COLMAP track metadata**. That is available
for these known training-side cameras but not an arbitrary new camera; this is a
privileged known-camera feasibility selector, not a deployable novel-pose source
retrieval method. Inherited reconstruction/metadata are not a full-chain blind OOF.
No external camera was newly rendered; source poses stay in official COLMAP w2c.

Upstream emitted a non-writable NumPy warning on converting Pillow arrays. It
does not write the CPU array in this path; both array and file hashes were checked
unchanged. Preserve the warning and the original execution source; a future new
runner can explicitly copy its read-only inputs instead of silently patching this
sealed run. There were no model/runtime exceptions or failed-run directories.

### Provenance and validation

All metrics above use this ID, base commit
`49cdcdb1c9915a9544d568754af95ea20a85ec9c`, seed20260805, the registered released
dataset manifest and `sem_005_oof_spatial_v1`. Dirty user worktree preserved; exact
new executed source/config copies and hashes are in the preparation snapshot.
No commit, push or upload. Config still says preregistered because its hash is sealed.

Git-ignored outputs:

```text
outputs/sem320/prepare-20260908-r1/result.json
  f9ca2db679d8c2982ff0c9ee786c33e407dfc7fc203b147629db0042cdc53755
outputs/sem320/runtime-20260908-r1/result.json
  4ee4df293e06bd2faa252cbb3794c6bdb88789bd59306e20cec6c1a75a3000bc
outputs/sem320/replay-20260908-r1/result.json
  201d6abb0875e76f14c4b7f06683a190d30fc723d0e7d7f547e072dadc07d624
config
  3dcefab61d0c4d1aa30bbe0df1ab6e21f1bb7a60a9e86eecab4191de90307c3f
```

Four new preflight contract tests passed, as did four companion geometry tests.
Repository health:128 configs,233 tests,46 optional-runtime skips; optional
default-path dataset scan skipped. Actual selected RGB/camera and frozen-artifact
hashes were checked explicitly. No claim that all original datasets or every
optional model-environment dependency were revalidated.

This proves the fixed fast512 matcher is runnable, not that dense semantic
transfer works. The separately registered SEM321 geometry diagnostic subsequently
failed its screen; do not turn this engineering pass into a semantic candidate.
Rollback: keep using the original SEM161 interface and omit all optional RoMa
scripts/import overlays. Upstream copies, complete weight and all outputs remain
Git-ignored and retained; no destructive cleanup is necessary.

Official references: [RoMa release](https://github.com/Parskatt/RoMaV2/releases/tag/v2.0.1),
[pinned loader](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/src/romav2/romav2.py),
[native fallback](https://github.com/Parskatt/RoMaV2/blob/95c9968145c8906b7b59383258e9f73b02853d89/src/romav2/local_correlation.py),
[DINOv3 component license](https://github.com/facebookresearch/dinov3/blob/adc254450203739c8149213a7a69d8d905b4fcfa/LICENSE.md).

## Registration — 2026-09-08

Original pre-run state: preregistered, no matcher inference or semantic scoring yet. The user authorized
the next planned dense preflight after SEM314 failed. Parents SEM314/SEM250 and
SUBMIT020, base49cdcdb1c9915a9544d568754af95ea20a85ec9c, seed20260805,
manifest `data/manifests/contest_dataset_v1.json`, existing `sem_005_oof_spatial_v1`.

First establish source/license/hash and isolated-runtime feasibility, then at most
3 fixed training-side pairs under the fixed official512 fast setting. Sources
must be in the target's spatial-fold train set; no target real image or labels.
All warp/overlap outputs are diagnostics only. Six forward/reverse matches, one
identity probe and one new-process first-forward replay; no training or semantic
candidate. Engineering tests are specified in the immutable config before execution.

Read-only source investigation confirms the full official author release uses
DINOv3-L but does not need a separate gated HF backbone download. Both MIT and
custom DINOv3 notices are retained; no extraction as substitute SEM310 weights.
Upstream Linux packaging requires fused-local-corr while source explicitly offers
a native PyTorch fallback. This preflight intentionally uses that existing fallback
and a Pillow12 overlay without modifying the working training/submission runtime;
it is not advertised as a complete standard pip installation.

The SIMPLE_RADIAL official source camera has nonzero radial distortion, while
the frozen rendered target is pinhole. A later geometry diagnostic must explicitly
undistort source coordinates, preserve the half-pixel convention, and not mistake
epipolar consistency for true semantic correctness. No novel camera is rendered
by this preflight; official COLMAP metadata is used only to select input sources.

Stop on source, isolation, runtime, memory or replay failure; preserve all outputs.
Success permits separate diagnostic registration, not SEM161 promotion, final
training, bundle modification, upload or Git push.
