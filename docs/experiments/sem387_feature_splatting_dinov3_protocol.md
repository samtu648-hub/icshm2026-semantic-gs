# SEM387 RGB150 Feature Splatting + DINOv3 protocol

## Scope

`SEM-387-RGB150-FEATURE-SPLATTING-DINOV3` is a project adaptation of Feature
Splatting, not a strict reproduction. It asks one fixed question on
`dev30_local_composition_v1`: does frozen DINOv3 feature distillation improve a
supervised per-Gaussian semantic field when RGB150 and all 3D Gaussian parameters
remain immutable?

The experiment has two freshly trained arms. `label_only` and
`dinov3_distilled` instantiate the same 32D per-Gaussian latent, shared 256D
decoder trunk, five-class head, 768D teacher head, initialization, optimizer,
and exact SEM384 6000-entry labeled sampling plan. The candidate alone adds one
crop from the 100 unlabeled RGB-training views and a frozen DINOv3 cosine loss at
each successful update. Candidate-only access and runtime are reported
separately.

## Frozen decisions

- Parent RGB: `RGB-150-STANDARD-ABSGRAD-DEV30`, 741,957 Gaussians.
- Semantic split: 270 labeled training views, 100 unlabeled training views, and
  30 sealed development holdout views.
- Main endpoint: 6000 successful optimizer updates; step 3000 is descriptive
  only and may not be selected by quality.
- Feature field: zero-initialized 32D latent, matching the corrected full
  Feature Splatting implementation at commit
  `e8647b3fdb19dfebd8bab61a8e495fde03ead119`.
- Decoder: weight-normalized `32->256` shared 1x1 trunk with ReLU, then matched
  `256->5` class and `256->768` DINOv3 heads.
- Optimizer: Adam, latent LR `0.0025`, decoder LR `0.001`, no weight decay.
- Supervision: unweighted pixel CE in both arms. The candidate adds
  `0.01 * mean(1-cosine)` on final DINOv3 patch tokens. `0.01` is the effective
  DINO coefficient in the official full Feature Splatting `clip_dino` objective;
  it is not selected on this dataset.
- Teacher: the already verified local DINOv3 ViT-B/16, final normalized
  `last_hidden_state` patch tokens only. CLS and four register tokens are
  removed. The teacher is in eval mode, has no optimizer membership, and is
  hashed before and after candidate training.
- Provenance: all four DINOv3 model files and the SEM381
  `download_manifest.json` are hash-bound. No network fallback or alternate
  DINO checkpoint is allowed. The Standard renderer tree is independently
  checked against the source manifest sealed by RGB150.
- Teacher input: the exact candidate unlabeled RGB150 crop and horizontal flip,
  with ImageNet normalization and no resize or center crop.

Standard 3DGS exposes only three override-color channels and clamps rendered
values to `[0,1]`. Each signed latent channel is mapped with detached bounds
`lower=min(x)-0.5` and `scale=max(x)-min(x)+1`, rasterized in native three-channel
passes, and restored as `scale * R((x-lower)/scale) + lower * R(1)`. This is
algebraically the same front-to-back alpha composite. It does not introduce a
new projection, sorting pass, opacity, or substitute splat model.

## Access gate

Preparation hashes only the 370 RGB-training renders and 270 training labels.
The 30 holdout RGB bytes and labels remain unopened during preparation,
preflight, and both full fits. Inference may begin only after both arm result
records attest sealed step-6000 checkpoints and their hashes. Holdout metrics
can never select architecture, weight, layer, resource rescue, or checkpoint.

## Engineering gate

Before the full run, each arm performs exactly eight successful updates from the
same initial state. The gate checks finite losses and gradients, nonzero latent
and class-head gradients, no control-arm teacher-head gradient, candidate DINO
gradient, DINO and RGB parameter hashes, exact frozen RGB pixels, chunk order,
probability conservation, checkpoint reload, the exact labeled trace, and peak
memory below the physical 12GB device limit. Failure closes the attempt and is
preserved; recipe changes require another experiment ID.

## Evaluation

After both endpoints are sealed, both arms render the 270 training views, 30
Dev30 holdout views, 100 unlabeled training views, and the fixed no-GT camera
trajectory. Report five-class and foreground mIoU, per-class IoU/precision/recall,
`stay_cable` FP/FN and 2px boundary precision/recall/F1, all three registered
holdout groups, paired per-image changes, fixed local views, trajectory label
change, model size, training and inference time, peak memory, external-camera
latency, checkpoint replay, and RGB invariance. Compare against SEM384 and
SEM386 under the same Dev30 metric definitions, while retaining the existing
SEM386 visual-review limitation.

No second seed, all-data fit, geometry optimization, DINO layer search,
loss-weight search, pseudo-labeling, fusion, or post-processing is authorized.

## Execution status

The exact registered DINOv3 directory was restored and passed all fixed file and
manifest hashes. Preparation read only the 370 training RGB and 270 training
labels, both eight-update arms passed the engineering gate, and both fresh arms
completed 6000 successful updates before holdout access. The candidate was
`0.019109 pp` below the matched control on Dev30 five-class mIoU and
`0.022977 pp` below it on foreground mIoU; it improved 10 and degraded 20 of the
30 holdout images. Both Feature Splatting arms were more than `6.17 pp` below
SEM384. The run is complete, preserved as a negative result, and not promoted;
no forbidden follow-up is authorized by this outcome.
