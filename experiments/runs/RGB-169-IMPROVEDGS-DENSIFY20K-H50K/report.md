# RGB-169 ImprovedGS densify-until-20k comparison

Status: completed; retain as a local 385--392 improvement, do not select as the unified reconstruction baseline.

RGB169 kept the RGB166 all400 ImprovedGS recipe, official cameras, 72,292-point initialization, seed 0, one-million Gaussian budget, EAS/LAS/RAP/MU and 50k total training fixed. The only primary variable was `densify_until_iter`, changed from 15,000 to 20,000. The full training, five-endpoint evaluation and fixed 144-pose trajectory completed successfully.

At 50k it contained 944,622 Gaussians and used 8,214 MiB peak reserved memory. Relative to RGB166 50k, all400 PSNR/SSIM/LPIPS changed by +0.004115 dB/+0.000100/−0.000609, with 185/400 views improving in PSNR. The manually fixed 385--392 bridge crop improved +0.476675 dB/+0.005376 SSIM, and the 385--392 difficult group improved +0.161205 dB. However 159--160 PSNR fell 0.005842 dB, and the all300 bridge-region PSNR changes for deck, `stay_cable`, tower and foundation were −0.026947/−0.053571/+0.020182/−0.017217 dB. Cable visible support also fell by 0.036558 percentage points.

The result shows that extending generic growth can help the selected 385--392 region, but it does not improve bridge structure consistently across the scene. RGB169 does not replace RGB166 or RGB170 and will not receive matched semantic training. It closes further scans of 25k/30k growth cutoff under this rule.

Evidence: `outputs/quality/RGB-169/full-r1/result.json`, `evaluation-r1/result.json` and `trajectory-r1/result.json`. Their SHA256 values are `4f138736e2ddd5dc43390b28f3208ea9055a1b5ab7383964d28e96ecf82a16cf`, `b20e7125acd90324fe60260cdf774784e9441c6da92993730c89dda21f21549a` and `b2d8ed0c58597673520da1aa2d8e1e38a00e7f2d0109e22f713cdf3038d0c652`.
