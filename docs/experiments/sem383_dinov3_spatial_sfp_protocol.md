# SEM383 DINOv3 spatial-prior SFP protocol

## Question

Does a lightweight RGB spatial-prior branch improve the spatial continuity and
local boundaries of frozen-DINOv3 semantic predictions relative to SEM382 while
RGB140, labels, sampling, decoder, losses, learning rates and budget stay fixed?

## Input-domain diagnostic first

Before training, run views `311`, `350` and `356` through frozen SEM378-6000 and
SEM382-6000 using both the prepared real photo and same-camera RGB140 rendering.
Use each model's existing native `768/512` sliding protocol without sharpening,
new images, optimization or labels. Save probabilities, hard predictions,
domain transition counts, class-area changes, entropy/confidence changes and
same-location crops. These no-GT results are localization clues, not accuracy or
a unique causal diagnosis. RGB140 predictions must replay the saved outputs
byte-for-byte.

## Official source and implementation scope

- Reference: Meta `facebookresearch/dinov3`, commit
  `6876159a11b4df116f30f667f8c9888617df0751`, source
  `dinov3/eval/segmentation/models/backbone/dinov3_adapter.py`, governed by the
  DINOv3 License. Publications must acknowledge DINOv3 and redistribution must
  include and follow that agreement.
- The official adapter uses a stride-4/8/16/32 convolutional Spatial Prior
  Module followed by deformable Extractors; its ViT-B outputs are all 768
  channels and require the custom `MultiScaleDeformableAttention` CUDA op.
- SEM383 is deliberately a custom deformable-free RGB spatial-prior residual on
  the existing SFP. It is not the Meta DINOv3 Adapter and not ViT-Adapter. It
  omits deformable Extractors, ConvFFN, level embeddings and custom CUDA ops to
  retain the SEM382 decoder feature contract and fit native 768 crops on 12 GB.

## Single treatment

- Direct control: SEM382-6000. Performance reference: SEM378-6000.
- Keep the complete SEM382 SFP. Add one RGB branch with stride-4 channels 64,
  then depthwise-separable downsampling stages with channels 96/128/192.
- Project the four levels to 192/384/768/1536 channels and add them to the
  corresponding SFP levels through per-channel gates initialized to `1e-3`.
- Use GroupNorm with eight groups and GELU. The branch consumes the same
  ImageNet-normalized RGB tensor as frozen DINOv3.
- The new branch has exactly 514,048 trainable parameters. Existing SFP plus new
  branch has 4,356,544 parameters; decoder has 20,250,310 parameters; frozen
  DINOv3 has 85,660,416 parameters.
- Reconstruct from the official ADE checkpoint and the same hash-verified local
  DINOv3 files. Do not load SEM382's trained SFP or decoder. Verify that every
  shared initial trainable tensor equals SEM382's common initialization.

## Frozen training protocol

- RGB140, 300 labels, manifest/split, seed `20260805`, the exact 6000-entry
  sampling plan, native 768 crops, flips, batch one and 768/512 inference stay
  fixed.
- DINOv3 remains `eval`, `requires_grad=False`, absent from the optimizer and
  state-exact. Hidden states remain 3/6/9/12.
- Existing SFP, new spatial branch and decoder all use initial LR `1e-4` under
  the existing AdamW, weight decay, global norm clip `0.01` and poly `0.9`.
- Run eight successful native-768 updates as shape/gradient/numerical/resource
  preflight. Every spatial stage, projection and gate must receive finite nonzero
  gradients. Reconstruct the common initial state before formal training.
- Run exactly 6000 successful updates. Save 3000 for analysis and 6000 as the
  preregistered primary endpoint; never stop or select by observed quality.
- OOM or numerical failure is preserved and explained. Do not lower resolution,
  change channels, disable the branch or alter the method to rescue this ID.

## Evaluation and interpretation

Infer all fixed 544 RGB140 inputs for both checkpoints. Compare SEM383-6000,
SEM383-3000, SEM382 and SEM378 on all 300 labeled training views using overall
and class IoU, per-view changes, corrected/harmed pixels, `stay_cable` FP/FN and
2 px boundary F1. Review 159/160/146/057/064/311/350/356 and trajectory 96--98,
plus all 100 unlabeled and 144 trajectory views. Repeat the photo/render input
diagnostic for SEM383-6000.

All 300-view scores are training fit, not held-out generalization. Unlabeled and
trajectory outputs are qualitative. A broad `stay_cable` envelope is not proof
of individual cable geometry. Report training time, memory, inference cost,
parameters, frozen-state checks and local gains/harms; then stop without starting
another experiment.
