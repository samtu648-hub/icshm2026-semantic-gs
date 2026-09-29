# RGB-157-ABSGS-DROP010

Completed 2026-09-21; fixed30k candidate evaluated and not promoted.
Matched RGB156; only training-time opacity dropout r=0.1*t/30000, compensation
and dropped-point visibility exclusion. Fixed private RNG, ordinary full-model
evaluation, same spatial v1 initialization/roles/seed0. Output:
`outputs/quality/RGB-157/full-r1`. No quality result yet.
[Protocol](../../../docs/experiments/component_sampling_dropgaussian_protocol.md).

Fixed-mask replay, private RNG, finite backward gradients and restoration of
ordinary RGB passed. The701-step probe exercised two densification events,
temporarily dropped64479 Gaussian incidences and exited0 in63.880 native seconds
(70.168 runner seconds); it was not quality-scored or reused for formal training.

## Fixed endpoint result

- Training fit: PSNR23.251872dB, SSIM0.895686, LPIPS-Alex0.247757.
- Confirmation: PSNR18.284731dB, SSIM0.821008, LPIPS-Alex0.381212.
- Versus fresh RGB156: PSNR-0.107260dB, SSIM+0.003063,
  LPIPS-0.001686. Region PSNR differences:-0.306968/-0.015108/+0.000296dB.
- 790907 Gaussians versus793893 control; capacities are close but not fixed.
- Runner2655.074 seconds, peak allocated/reserved1977.271/5966 MiB.
- Both30000 camera traces match exactly. Normal full-model renders and RGB CSV
  summaries verified after both endpoints were sealed. No training failure.

The preregistered RGB gate fails because overall PSNR decreases and region1
loses more than0.3dB. SSIM/LPIPS improve slightly, so this is a trade-off rather
than all-metric regression. Severe region1 occlusion remains; a faint bridge
outline in156 is not reliable reconstruction recovery.

SEM419/420 were not run under the original gate. The separately registered
QUAL016 frozen two-semantic-model probe subsequently found foreground mIoU
changes-2.814603/-1.697606pp, despite better cable IoU. This provides no basis
for matched semantic retraining of the fixed dropout recipe. No dropout-rate
search or combined component-quota model was started.
