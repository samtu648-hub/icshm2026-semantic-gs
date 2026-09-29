# QUAL008 spatial bottleneck diagnostic

QUAL008 is a read-only diagnostic over the already opened 30-view `spatial_confirmation_v1` result. It creates no predictions, does not train, and does not change RGB152 or SEM396.

Before execution, the RGB holdout CSV, RGB evaluation result, SEM396 inference result, split, prediction hashes, and label manifest are fixed. The analysis joins the same 30 image names, recomputes each view's five-class confusion and two-pixel `stay_cable` boundary counts from the sealed PNGs, and reports Spearman correlations between RGB PSNR/SSIM/LPIPS and semantic five-class mIoU, foreground mIoU, `stay_cable` IoU/recall, and boundary F1.

The 15 views with highest PSNR form a pre-specified better-RGB half. An actionable semantic residual exists if its aggregate boundary F1 is below 0.85 and at least three member views have `stay_cable` recall below 0.90, or if its aggregate five-class mIoU is below 0.92. Absolute Spearman correlation of at least 0.70 is described as strong RGB coupling. If both occur, the result is `mixed_rgb_and_semantic_bottleneck`; if only the semantic residual occurs, it is `semantic_residual_supported`; otherwise a boundary-loss candidate is not supported.

This classification authorizes only the design decision for a new, separately registered single-variable experiment. It is not evidence that boundary loss improves quality and does not authorize weight search, threshold search, checkpoint selection, or changes to RGB152.
