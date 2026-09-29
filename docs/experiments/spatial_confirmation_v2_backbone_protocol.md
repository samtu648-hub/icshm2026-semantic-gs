# Disjoint spatial-fold backbone replication protocol

Experiment IDs: `RGB-154-ABSGRAD-SPATIAL30-V2`,
`RGB-155-STANDARD-SIGNED-SPATIAL30-V2`,
`SEM-412-RGB154-SPATIAL255`, and `SEM-413-RGB155-SPATIAL255`.

## Question and roles

Does the AbsGS reconstruction-backbone advantage observed on
`spatial_confirmation_v1` persist on the camera-only selected, non-overlapping
`spatial_confirmation_v2` fold?

The fixed fold contains 349 RGB-training views, 255 semantic-training labels,
30 confirmation views, and 21 excluded buffer views. Its three contiguous
regions are 054--063, 096--105, and 166--175; overlap with the v1 confirmation
set is zero. Selection used only official camera centres, trajectory segments,
and the released camera graph with seed 402. Image pixels, annotations, model
outputs, and v1 quality results did not participate in fold selection.

Both RGB arms share the same 81,516-point training-only DLT initialization,
official cameras, seed 0, 30,000 updates, losses, optimizer, and schedule. The
only RGB variable is AbsGS absolute-gradient split versus original Standard
signed-gradient densification. Both endpoints must be sealed before either
confirmation RGB is scored.

Both semantic arms use the unchanged SEM396/SEM407 Swin-L Mask2Former recipe,
seed 20260805, 768 crops, and 6,000 paired samples. Initial full-state hashes and
sample lists must match exactly. Both checkpoints must be sealed before any
confirmation label is accessed.

## Decision

Report train/confirmation and all three regions for RGB PSNR/SSIM/LPIPS and
semantic five-class/foreground mIoU, `stay_cable` IoU/precision/recall, and 2px
boundary F1, together with model/runtime costs. The disjoint-fold replication
passes only if AbsGS exceeds Standard in overall PSNR, five-class mIoU,
`stay_cable` IoU, and boundary F1. Regional results remain descriptive and
must be reported even when mixed.

This fold is disjoint from v1 but historically exposed through earlier project
models, so it remains development replication rather than blind evaluation.

## Inference numerical recovery

The first SEM412 inference attempt stopped after 175 rendered views because one
FP16 sliding-window output contained exact-zero project-class mass. The safety
check correctly refused to fabricate background. That failed attempt is
preserved separately. The fixed recovery policy is to re-evaluate only a tile
that raises this exact empty-mass error in FP32, then apply the unchanged
five-class normalization. Other invalid scores still fail. The same policy is
applied to SEM412 and SEM413; checkpoints, images, labels, sliding windows,
thresholds, metrics, and decision gates are unchanged.
