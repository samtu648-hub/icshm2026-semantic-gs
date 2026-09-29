# Full released-data backbone comparison

Registered 2026-09-15 under RGB-130-STANDARD-ALL400, RGB-131-MIP-ALL400,
SEM-371-STANDARD-ALL300, SEM-372-MIP-ALL300 and QUAL-007-BACKBONE-COMPARISON.
Authorization: complete both 30k reconstructions and both matched 6k Mask2Former
fits; reuse RGB120 and SEM370; no automatic quality rejection or replacement.

## Common inputs and differences

- Dataset manifest: `data/manifests/contest_dataset_v1.json`; exact released
  400 RGB / 300 semantic names: RGB120-r2 `split.json`.
- Same RGB120 prepared PINHOLE scene, full 1320×989, released official
  CAMERA_FROM_WORLD poses, full-frame undistortion with released K, no crop.
  Same 303-derived DLT initialization PLY, no new point-cloud preparation.
- RGB seed 0, 30,000 iterations, CPU image caching, black background,
  no Standard/Mip exposure correction or depth supervision enabled, native densification.
  Standard pinned upstream defaults include opacity LR .025; Mip .05 and
  native 3D filtering / 2D kernel .1. Scaffold uses its learned anchor/offset
  representation and RGB120 settings. Same iteration count is not equal compute.
- Existing source revisions and installed source/extension hashes are captured.
  Native sources and frozen models are never patched. The new worker adapts only
  reader signature, explicit all400 camera access, common PLY, seed and GUI.
- Thirty-two-step RGB smokes are engineering checks at full image resolution,
  with all400 cameras and the same initialization. They are separate discarded
  fits, retained on disk; full runs start independently at step zero.

## Semantic recipe and evidence

SEM370's unchanged `mask2former` function is reused: official Mask2Former
Swin-L IN21k, 215,451,514 parameters, `model_final_6b4a3a.pkl`, five classes,
seed 20260805, 768 native crop and 512 inference stride, batch one, author AdamW
(.0001 base LR, .1 backbone multiplier, .05 decay), polynomial power .9,
global gradient clip .01, float16 AMP with the same finite-update handling,
6000 successful updates and checkpoints at 3000/6000. Both new fits start from
the same pretrained weights, never the existing trained SEM370 model.
An eight-update engineering probe precedes each fit and is separately recorded.
All 6000 crop/name/flip traces and the resolved model configuration must match
SEM370. Frozen rendered RGB hashes are checked before/after fitting.

Only raw five-class argmax is evaluated: background, deck, stay_cable, tower,
foundation. No extra experts, sidecars, rule gates or fusion are instantiated.
The final inference stage reloads the checkpoint in a fresh process and checks
all nine stored probabilities exactly, including reused SEM370 outputs.

## Evaluation and visual protocol

- All400 same-grid uint8 PINHOLE reference and render images; per-image PSNR,
  project Gaussian-window SSIM, LPIPS AlexNet v0.1; report arithmetic image means.
  These are training-camera fitting results, not hidden-view scores.
- All300 same prepared GT PNGs; independent 5×5 confusion counts, aggregate
  five-class mIoU and each class IoU/precision/recall. Per-image metrics retained.
  Same labels were used for fitting; no held-out claim.
- Fixed nine: 159,160,140,146,081,057,064,054,174. All arms use identical
  locations and crop coordinates. Photos are shown on the same undistorted grid
  as GT and renders. Existing diagnostic ROIs 159/160/146/174 are retained;
  any additional displayed crop is labeled as a visual inspection, not a test.
- Use all144 existing fixed novel flythrough cameras from
  `results/frozen_scaffold_sem001/flythrough_v1/cameras` under the external
  artifact root, with identity official-to-model transform and native K.
  Display the same full-resolution PINHOLE output grid across all arms.
  Camera metadata hashes are bound to each export. No GT is available: videos
  only inspect continuity, streaks, ghosts, flicker and disappearing structure.
  Unaligned adjacent-frame differences cannot be called temporal quality scores.
- Predict the100 unlabeled released images for visual checks only, no GT metrics.
- Capture total wall time (including preparation/IO), training-loop time where
  available, peak allocated/reserved VRAM and sampled device memory, renderer
  and semantic median timings (excluding first warmup frame), deployable size
  and total artifacts. Timing definitions and shared display usage are disclosed.

## Resources and rollback

Initial available disk approximately93GB. Estimate3–5hours,15–30GiB new output;
monitor rather than imposing an arbitrary reserve. GPU jobs share the project
lock and check the process table before admission. Full resolution/step counts
must not be reduced to recover OOM. Failure records remain. Existing18 frozen
artifacts and RGB120/SEM370 endpoints are hashed for final verification.
Rollback is selecting the existing RGB120+SEM370 paths; no system replacement,
Git push, upload, deletion or old-checkpoint rewrite is authorized or performed.

## Scaffold appearance clarification

RGB120 actually trains a32-dimensional per-camera appearance embedding. SEM370
exports all its training RGB with deterministic appearance index0. The main
three-system RGB/semantic comparison therefore uses that exact existing deployment
input, including exact300-image replay. A separate RGB-only supplemental all400
evaluation uses each camera’s native training UID, reconstructed from the saved
cameras.json and verified author Scene(shuffle=False) call. It is explicitly not fed into
SEM370, not mixed into the main system ranking, and does not involve training.
The distinction must remain visible when judging reconstruction fitting capacity.

完整原生model/optimization/pipeline参数已按各自实际命令展开到运行目录resolved_parameters.json，并核对原始arguments源码哈希。三者均L1/SSIM光度配比0.8/0.2；Scaffold保留原生0.01体积正则、anchor/offset/MLP及外观学习率，Standard/Mip保留各自优化及增密机制。因此本研究比较完整原生骨干配方，不声称只隔离一个CUDA过滤算子的因果效果。
