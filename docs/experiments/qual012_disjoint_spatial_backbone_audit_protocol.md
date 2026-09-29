# Cross-fold reconstruction-backbone audit

Experiment ID: `QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT`.

This read-only audit joins the sealed `spatial_confirmation_v1` and
`spatial_confirmation_v2` reconstruction-backbone comparisons. It performs no
training, inference, threshold search, checkpoint selection, or metric
recomputation from labels.

The carried-forward v2 decision gate passes only if AbsGS-minus-Standard is
positive for overall confirmation RGB PSNR, semantic five-class mIoU,
`stay_cable` IoU, and two-pixel boundary F1. The audit also reports SSIM,
LPIPS, foreground mIoU, precision/recall, all three v2 regions, and the
training/model/rendering cost trade-off. Mixed regional or auxiliary metrics
must be retained.

The two folds have zero confirmation-image overlap and were selected from
camera geometry rather than pixels or labels, but both are historically
exposed development evidence. The audit supports a project-method choice; it
does not claim blind-test performance or universal dominance.
