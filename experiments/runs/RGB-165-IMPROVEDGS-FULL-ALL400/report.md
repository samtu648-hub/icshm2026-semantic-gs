# RGB-165 full ImprovedGS all-400 candidate

Status: completed; bridge-fit leader, research candidate only.

This candidate enables the complete coordinated ImprovedGS package: edge-aware score (EAS), long-axis split (LAS), reset-and-prune (RAP), late update cadence (MU), and the same one-million Gaussian budget used by RGB-164. Everything else is matched to RGB-164, including all 400 RGB observations, official cameras, initialization, seed, camera trace and 30k endpoint.

The first RGB-164 preflight attempt exposed a missing official `fused_ssim` build before any optimization. That failure is preserved. After compiling the pinned submodule, both 601-step preflights completed with exact paired camera traces. RGB-165 then completed 30,000 updates in 881.710 seconds with 933,689 Gaussians and 7,298 MiB peak reserved memory.

## Result

On the released all-400 fit evaluation, RGB-165 reaches PSNR 23.427625 dB, SSIM 0.902810 and LPIPS 0.227736. Relative to RGB-140 this is +0.140288 dB, +0.006736 SSIM and -0.018117 LPIPS. On the fixed 385--392 bridge crops it reaches 26.679824 dB / 0.892641 SSIM, or +1.842995 dB / +0.029203 over RGB-140. The two-scale bridge gradient error falls from 0.011191 to 0.009748. Every one of the eight bridge views gains PSNR over RGB-140.

Relative to matched RGB-164, the complete EAS+RAP+MU addition is small and mixed: global +0.027154 dB / +0.000490 SSIM but LPIPS worsens by 0.001444; bridge crops gain 0.416490 dB while SSIM falls 0.000838. Gradient error improves by about 0.93% on average, but is worse on 389, 390 and 392. RGB-165 uses 5.19% fewer Gaussians than RGB-164.

The fixed 144-pose trajectory is qualitative because it has no RGB ground truth. It shows stable bridge towers and decks, but the water haze/floaters and the broad bridge/background smear remain. RGB-165 is therefore the current bridge-fit leader and a worthwhile research candidate, but the reconstruction defect is not solved and the incremental benefit of EAS/RAP/MU over the shared budgeted LAS control is not decisive. Do not train matching semantics or replace the production RGB-140 chain until a spatial holdout test confirms transfer beyond training views.

Evidence: `outputs/quality/RGB-165/full-r2/evaluation-r2/result.json`, `summary-r1.json`, `trajectory-r1/result.json`, and `evaluation-r2/montages/bridge-final-comparison.jpg`.
