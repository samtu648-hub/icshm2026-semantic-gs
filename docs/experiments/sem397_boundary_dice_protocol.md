# SEM397 fixed boundary-Dice protocol

SEM397 tests whether a lightweight thin-structure boundary objective improves spatial semantic generalization while the reconstruction remains frozen. It is authorized by the pre-registered QUAL008 outcome `semantic_residual_supported` and uses SEM396 as its matched control.

The RGB152 checkpoint, `spatial_confirmation_v1` roles, 256 training labels, 30 confirmation labels, 17 excluded buffer views, ADE-pretrained Swin-L Mask2Former, seed 20260805, 6,000-update budget, native 768 crop, optimizer, polynomial schedule, AMP retry contract, sample names, crop coordinates, flips, and 768/512 inference are fixed to SEM396. The only method change is a weight-0.05 Dice loss on a differentiable one-pixel-radius morphological boundary of the normalized dense Class ID 2 (`stay_cable`) probability. The target uses the same morphology on the binary Class ID 2 label. The weight is fixed before execution and will not be searched.

The 6,000-step checkpoint must be sealed before any confirmation RGB or label access. Training must prove the full sampling trace matches the frozen plan and that its fresh initialized state matches the same-seed SEM396 initialization contract. Evaluation uses the same confusion and 2-pixel boundary definitions as SEM396.

Promotion requires every frozen gate: confirmation five-class mIoU above 89.204606%; boundary F1 gain of at least 0.5 percentage points; `stay_cable` recall above 91.651491%; `stay_cable` precision drop no larger than 0.2 percentage points; no confirmation class IoU drop larger than 0.3 percentage points; and no spatial region five-class mIoU drop larger than 0.5 percentage points. Failure of any gate closes the candidate. Passing all gates authorizes only a second-seed confirmation, not automatic all-data migration or replacement of the current default.

This is a historically exposed development split and a single seed. The result cannot be called blind performance. Poor RGB152 reconstruction, especially region 1, remains a competing bottleneck.
