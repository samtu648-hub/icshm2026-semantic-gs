# RGB-158-NATIVE-ABSGS-REPLAY

Completed2026-09-21. Exact original unwrapped AbsGS30k path, no scientific
variable change, to diagnose the discrepancy between RGB152 and RGB156.
[Protocol](../../../docs/experiments/native_absgs_replay_protocol.md).

## Result

- Train PSNR23.296436dB / SSIM0.897336 / LPIPS-Alex0.242734.
- Confirmation18.536542dB /0.823748 /0.372253.
- Region1:14.395260dB /0.743248 /0.534687.
- 795399 Gaussians; training runner2685.132 seconds; peak allocated/reserved
  1950.815/5978MiB. All stages exited0; fixed30k checkpoint reloaded for evaluation.

Historical RGB152 confirmation was19.383911dB, fresh disabled-dropout RGB156
18.391991dB. The original unwrapped path also diverges substantially from the
historical result: about-0.847369dB overall and-3.009741dB in region1. View156
preserves more bridge structure than RGB156, while157--161 degrade sharply
(159 PSNR8.524964dB). Failure location changes; it is not simply a uniform
color offset or a dropout-wrapper-only phenomenon.

This demonstrates a repeatability concern at the full training-trajectory level,
not a proven cause. The same-seed records do not identify CUDA atomic ordering,
initialization details or another unlogged runtime factor as the cause. Preserve
all results rather than picking the best run or rewriting QUAL010--013. Fixed
checkpoint rendering remains reproducible; training reproducibility and
cross-run quality claims require stronger qualification.
