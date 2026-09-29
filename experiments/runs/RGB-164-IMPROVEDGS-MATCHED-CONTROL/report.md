# RGB-164 ImprovedGS matched control

Status: completed matched control.

This arm uses the official ImprovedGS implementation, its budgeted absolute-gradient densification path and LAS, but disables EAS, RAP and MU. It shares the all-400 scene, official cameras, 72,292-point initialization, seed, private camera sampling trace, 30k endpoint, one-million Gaussian budget and base hyperparameters with RGB-165. Keeping LAS enabled is necessary because the official no-LAS fallback does not strictly limit ordinary splitting to the computed warm-up budget. The pair therefore asks whether edge scoring, reset-and-prune and late updates make the budgeted LAS reconstruction useful; historical RGB-140 provides the absolute strong-baseline comparison.

The first preflight attempt stopped before optimization because the official `fused_ssim` extension was not yet built. That dependency failure is preserved under `outputs/quality/RGB-164/preflight-r1`. After compiling the pinned submodule, `preflight-r2` completed 601 updates. The paired camera trace matched RGB-165 exactly.

The full all-400 run completed 30,000 updates in 851.701 seconds with 984,817 Gaussians and 4,920 MiB peak reserved memory. On the fixed all-400 fit evaluation it reaches PSNR 23.400472 dB, SSIM 0.902320 and LPIPS 0.226292, all better than RGB-140 (23.287337 / 0.896074 / 0.245853). The manual 385--392 bridge crops reach 26.263334 dB and 0.893479 SSIM, versus RGB-140's 24.836829 dB and 0.863438.

This is a strong control and shows that the shared budgeted LAS recipe is responsible for most of the gain. It does not solve the broad horizontal bridge/background smear, and the fixed novel-view trajectory still shows water haze and floaters. No semantic model has been trained on it.
