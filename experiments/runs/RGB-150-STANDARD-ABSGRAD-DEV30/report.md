# RGB-150-STANDARD-ABSGRAD-DEV30

- Status: completed on 2026-09-17; fixed local-composition development baseline.
- Parent recipe: `RGB-140-STANDARD-ABSGRAD-ALL400`.
- Split: `dev30_local_composition_v1`, 370 train RGB and 30 development holdout RGB.
- Seed: `0`; endpoint: `30000`; official cameras fixed.
- Initialization: 79,630 points re-triangulated and colored from training observations only; PLY SHA256 `30916591f71c5fb2af8b548c996bc1c036dec597cd81caba21cab2376ad8574e`.
- Claim boundary: development holdout excluded from this run's optimization and initialization, but the official poses and split-design GT composition statistics are provided conditions. This is not a blind test or a strictly guarded spatial split.

## Results

- Training role, 370 views: PSNR `23.270499 dB`, SSIM `0.895716`, LPIPS-Alex `0.247654`.
- Holdout, 30 views: PSNR `21.726055 dB`, SSIM `0.878247`, LPIPS-Alex `0.252694`.
- Holdout gap relative to the training role: PSNR `-1.544445 dB`, SSIM `-0.017469`, LPIPS-Alex `+0.005040`.
- Regular-local: PSNR `22.464792 dB`, SSIM `0.873265`, LPIPS-Alex `0.209631`.
- Oblique-local: PSNR `21.750625 dB`, SSIM `0.868828`, LPIPS-Alex `0.297373`.
- Distant-local: PSNR `20.962747 dB`, SSIM `0.892648`, LPIPS-Alex `0.251077`.

The distant block has the lowest PSNR; the oblique block has the highest LPIPS. These are post-seal observations and did not alter SEM-384 membership, recipe, seed, or endpoint.

## Execution Evidence

- The native worker observed exactly 370 train and 30 evaluation cameras and loaded the registered 79,630-point training-only initialization.
- Final checkpoint: 30,000 iterations and 741,957 Gaussians.
- Training wall time: `2604.796 s` (`43.413 min`). Worker peak allocated/reserved memory: `1998.196/4952.000 MiB`; sampled total GPU memory peaked at `5995 MiB`.
- All 400 fixed cameras were freshly rendered before PSNR/SSIM/LPIPS evaluation.
- Training result SHA256: `4351e9989c554b1d033db2ae9fa12c34ebd30208e67804e6adb2b707af771903`.
- Evaluation result SHA256: `cd9deb55ec8bb0b1317e7c0baae61f3101609e18df2486067bebde615ee2468d`.

Conclusion: this run establishes the fixed development RGB baseline. It does not replace the all-data RGB140 endpoint and must not be compared to all-data training-fit scores as if the roles were identical.

Final verification: 11 task-specific tests passed, 240 JSON-compatible configs validated, and `repo_health` passed 568 tests with 76 optional skips.
