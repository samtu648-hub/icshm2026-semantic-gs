# SEM-426 direct five-logit spatial baseline

## Objective

Measure the simplest native 3D semantic field against the matched 2D
Mask2Former baseline without changing reconstruction, data roles, random seed,
crop locations, flips or training-update count.

## Fixed comparison

- Shared reconstruction: sealed `RGB-152` Standard+AbsGS spatial model.
- Shared split: `spatial_confirmation_v1`.
- Shared roles: 256 labeled training cameras, 30 spatial confirmation cameras,
  17 excluded buffer cameras.
- Shared sampling: the exact 6,000 entries sealed by `SEM-396`, SHA-256
  `70a354e3e4a733942f347ba673926fb717d170287f12d0aca741bb047e29b2ef`.
- Shared seed: `20260805`.
- Fixed primary endpoint: step 6,000. Step 3,000 is descriptive only.

The representation differs necessarily: SEM396 uses ADE-pretrained
Mask2Former and this experiment uses five zero-initialized logits per frozen
Gaussian. The result therefore compares complete semantic representations; it
is not a parameter-count-matched architecture ablation.

## Direct semantic field

Only an `N x 5` logit tensor is trainable. Per-Gaussian softmax probabilities
are composited through two native Gaussian passes. Residual transmittance is
background. RGB position, scale, rotation, opacity, SH and exposure are frozen.

Loss is median-frequency weighted NLL, with counts and weights derived only
from the 256 training labels and clipped to `[0.05, 20]`. Adam learning rate is
`0.05`, matching the earlier SEM-001 static baseline; no search is allowed.

## Access and evaluation

Preparation and training may read only the 256 training labels and camera
metadata. The 30 confirmation labels may be opened only after the step-6,000
checkpoint hash is sealed. Buffer RGB/labels and released unlabeled semantics
remain unused.

Report training and confirmation five-class/foreground mIoU, all class metrics,
`stay_cable` 2px boundary F1, three spatial regions and worst-20% per-view
foreground mIoU. Compare to SEM396, but do not claim blindness, statistical
significance or generality beyond this split.

Passing or failing this experiment does not modify any production/default
checkpoint. Any spatial regularizer, geometry update, second fold, second seed,
Scaffold port or all-data fit requires a new registered ID.
