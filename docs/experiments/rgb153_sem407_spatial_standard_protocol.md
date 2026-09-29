# Matched Standard spatial-confirmation protocol

Experiment IDs: `RGB-153-STANDARD-SIGNED-SPATIAL30` and
`SEM-407-RGB153-SPATIAL256`.

## Question

Does the AbsGS absolute-gradient split used by RGB152/SEM396 improve the fixed
spatial confirmation result enough to justify its larger model and training
cost, or is the original signed-gradient Standard 3DGS recipe preferable for
the final system?

## Frozen comparison

The control reuses `spatial_confirmation_v1` byte-for-byte: 353 RGB-training
views, 256 semantic-training labels, 30 confirmation views, and 17 excluded
buffer views.  It reuses the exact RGB152 prepared scene and its 78,045
training-only initialization points.  The RGB seed is 0 and the endpoint is
30,000 updates.

The only reconstruction-method variable is densification:

- RGB152: the registered AbsGS recipe, signed clone threshold 0.0002 and
  absolute-gradient split threshold 0.0004;
- RGB153: the original Standard 3DGS signed-gradient clone/split rule with its
  default threshold 0.0002.

All data roles, official cameras, initialization, loss, optimizer, schedule,
resolution, checkpoint, and evaluation code are otherwise fixed.  The source
is the previously frozen official Standard 3DGS checkout used by RGB130.

SEM407 then reuses the SEM396/SEM378 ADE-pretrained Swin-L Mask2Former recipe,
seed 20260805, the same precomputed 6,000-sample crop/flip plan, 768 crop,
optimizer, schedule, endpoint, and native inference. Its only input change is
RGB153 rather than RGB152 renders. Confirmation images and labels remain
inaccessible until the step-6000 checkpoint is sealed.

## Fixed decision evidence

Report paired differences for confirmation and each spatial region in RGB
PSNR/SSIM/LPIPS, five-class and foreground mIoU, `stay_cable` IoU/precision/
recall, two-pixel boundary F1, training time, and Gaussian/model size.

AbsGS may be preferred only if the matched spatial evidence shows a meaningful
and spatially consistent benefit that is worth the already observed all-data
cost. A mixed or negligible result defaults to Standard for efficiency and
simplicity. Because these released views were used in earlier project history,
the result remains a historically exposed spatial development confirmation,
not blind testing.

