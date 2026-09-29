# Fresh spatial-confirmation baseline protocol

Experiment IDs: `RGB-152-STANDARD-ABSGRAD-SPATIAL30` and `SEM-396-RGB152-SPATIAL256`.

## Purpose

This chain measures the existing RGB140/SEM378 recipe on a camera-only spatial split. It is a validation baseline, not a new method. All decisions, identities, data roles, seeds, endpoints, and evaluation rules are fixed before training.

The split is `experiments/splits/spatial_confirmation_v1.json`, promoted byte-for-byte from the SEM395 camera-only candidate membership. It contains 353 RGB-training views, 256 semantic-training labels, three contiguous ten-view confirmation regions, and 17 buffer views. Confirmation and buffer views are excluded from every initialization observation, point colour, RGB update, semantic update, feature/pseudo-label source, training diagnostic, and model-selection path.

The released views have participated in earlier research models. The new chain is fresh and spatially guarded, but the confirmation images are not historically unseen; results must be described as a once-only spatial development confirmation, not blind testing.

## RGB152

RGB152 reuses the RGB140 Standard 3DGS + AbsGS recipe without modification: seed 0, 30,000 iterations, signed clone threshold 0.0002, absolute split threshold 0.0004, densification from 500 through 15,000 every 100 steps, opacity reset every 3,000 steps, and official cameras.

The scene is rebuilt from the released SIMPLE_RADIAL data using the same full-frame OpenCV undistortion to PINHOLE. Only the 353 training views contribute tracks and point colours. The deterministic preparation produced 78,045 accepted points; the 30 confirmation cameras are present only as explicit evaluation cameras, while all 17 buffer cameras are absent from the prepared scene.

RGB evaluation renders all 383 prepared cameras once from the sealed 30k checkpoint and reports PSNR, SSIM, and LPIPS separately for 353 training views, 30 confirmation views, and each of the three ten-view regions. No checkpoint selection or recipe change follows the confirmation score.

## SEM396

SEM396 reuses the SEM378 Mask2Former recipe: ADE-pretrained Swin-L initialization, seed 20260805, 6,000 successful 768x768 native-crop updates, the original loss/optimizer/poly schedule/AMP retry policy, and native 768/512 inference. It trains only on the 256 semantic-training labels and RGB152 renders. The sampling plan is sealed before training.

The 30 confirmation RGB images and labels may be opened only after the 6,000-step checkpoint and all training artifacts are sealed. Inference reports five-class and foreground mIoU, per-class precision/recall/IoU, `stay_cable` FP/FN, two-pixel boundary metrics, per-view metrics, and the three spatial regions. No threshold, endpoint, seed, loss, crop, TTA, or post-processing search is allowed.

## Interpretation and next decision

RGB152 and SEM396 establish a spatial baseline for the current method. Their scores cannot be compared directly with all-data training-fit scores as if the roles were identical. The chain does not replace RGB140+SEM378 or RGB130+SEM371 automatically.

Only after SEM396 closes may a boundary-loss candidate be registered. Such a candidate must use the identical RGB152 checkpoint, seed, sampling plan, initialization, 6,000-step endpoint, and inference; its sole scientific variable may be one pre-specified boundary auxiliary loss. SEM396 itself authorizes no candidate training.

Any access leak, identity/hash drift, non-finite state, GPU conflict, output collision, checkpoint mismatch, or confirmation/buffer use before the permitted stage preserves the failure and stops the chain.
