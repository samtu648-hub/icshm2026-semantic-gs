# Frozen downstream diagnostic after the DropGaussian RGB trade-off

2026-09-21, authorized under the user's instruction to investigate worthwhile
extensions directly. RGB156/157 completed; dropout loses0.107260dB PSNR but
improves SSIM0.003063 and LPIPS0.001686. The original strict RGB gate failed;
SEM419/420 stay not run. This protocol does not rewrite that gate.

To avoid rejecting a potentially useful downstream trade-off by PSNR alone,
run a cheap fixed diagnostic: SEM396 and SEM410, each trained only on the v1
training labels, segment the frozen RGB156 and RGB157 outputs at all30 v1
confirmation cameras. Four combinations,120 predictions, same native768/512
inference, no loss/training/postprocessing/fusion. Seal all predictions before
reading labels, then independently count all per-class confusion matrices.
Report foreground/five-class/cable metrics for both seeds and all regions.

Positive foreground and nonnegative cable IoU under each frozen semantic model
would motivate separately registered matched semantic training, not promotion.
Otherwise close this dropout recipe without changing its rate. This is an
input counterfactual using historical development data; it does not isolate
geometry from appearance or replace matched-render-domain training.

Also retain the fresh-control discrepancy relative to historical RGB152:
the new r=0 control shows severe occlusion at156 despite similar training fit.
Check initialization, source, camera/config and endpoint integrity before
interpreting that discrepancy as optimization variability; do not silently
replace the new matched control with the more favorable historical run.
