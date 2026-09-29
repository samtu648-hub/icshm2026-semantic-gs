# QUAL-008-SPATIAL-BOTTLENECK-DIAG

- Status: completed on 2026-09-20.
- Scope: read-only join of the fixed RGB152 and SEM396 results on the 30 historically exposed spatial-confirmation views.
- Training and prediction changes: none.
- Outcome: `semantic_residual_supported`; the pre-registered boundary candidate is supported for a separate experiment ID.

## Result

The rank association between PSNR and five-class mIoU was only 0.2988, and the association between PSNR and 2-pixel `stay_cable` boundary F1 was 0.2209. The 15 views with higher PSNR still reached only 90.7948% five-class mIoU, 88.5928% foreground mIoU, and 77.9753% boundary F1. Their `stay_cable` IoU/recall were 92.7868%/94.5778%. This satisfies the frozen semantic-residual rule because better-RGB five-class mIoU remains below 92% and boundary F1 remains below 85%.

RGB quality is still relevant. SSIM versus boundary F1 had Spearman 0.7891, while LPIPS versus boundary F1 and `stay_cable` recall were -0.6997 and -0.7789. The lower-PSNR half reached only 87.4634% five-class mIoU and 86.6406% `stay_cable` recall. Therefore the diagnostic supports a semantic boundary-loss test but does not claim that RGB is irrelevant or that a semantic loss can repair the poor RGB152 region-1 reconstruction.

The pre-registered decision fields are `semantic_residual_present=true`, `strong_rgb_coupling_present=false`, `boundary_candidate_supported=true`. The latter means only that one fixed, single-variable boundary candidate may be tested against SEM396. It is not evidence that the candidate improves quality.

The result and per-view CSV SHA256 values are `51fd3e52a1af9e9ddcf40181a7fa4843d983533fe929fe5e85360831010a2752` and `596f28e008bb6b4ff27ce7fb4b45d12f494b1f9ef4f7d7e3fcd8a3e1ac8e43c0`.
