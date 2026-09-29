# RGB-170 ImprovedGS 1.5M Gaussian budget comparison

Status: completed; current strongest released-all400 bridge-fit candidate, spatial generalization unconfirmed.

RGB170 kept the RGB166 all400 ImprovedGS recipe, official cameras, 72,292-point initialization, seed 0, EAS/LAS/RAP/MU, 15k densification cutoff and 50k total iterations fixed. The only primary variable was the Gaussian budget, changed from 1,000,000 to 1,500,000. The full training, five-endpoint evaluation and fixed 144-pose trajectory completed successfully.

At 50k it contained 985,530 Gaussians, 53,561 more than RGB166 (+5.747%), despite ending below one million after subsequent pruning. It trained in 1,367.464 seconds with 6,328 MiB peak reserved memory. Relative to RGB166 50k, all400 PSNR/SSIM/LPIPS changed by +0.014167 dB/+0.000336/−0.000657, and 252/400 views improved in PSNR. The 385--392 fixed crop improved +0.243198 dB/+0.003589 SSIM. Deck, `stay_cable`, tower and foundation region PSNR improved +0.034553/+0.014981/+0.042327/+0.106197 dB, and cable visible support increased 0.098614 percentage points.

The result is positive but modest and mixed by route. Difficult-group PSNR changed by +0.016457 for the legacy set, +0.029809 for 159--160, −0.038278 for 235--245 and +0.076900 dB for 385--392. LPIPS also regressed slightly outside 385--392, and the no-GT trajectory temporal-change proxies rose slightly. Therefore RGB170 is selected only as the strongest full-data bridge-fit development candidate; it is not yet evidence of better unknown-view reconstruction.

The one-million setting affected topology during densification, but extra capacity did not solve broken or missing thin cables. Stop generic budget and growth-cutoff scans. The next reconstruction work is a read-only Gaussian support diagnosis on RGB170, separating absent/misplaced support from excessive scale, orientation or suppressed opacity. Only one evidence-matched repair should follow, then a spatial holdout confirmation. Matched full300 semantics start after the reconstruction recipe is frozen.

Evidence: `outputs/quality/RGB-170/full-r1/result.json`, `evaluation-r1/result.json` and `trajectory-r1/result.json`. Their SHA256 values are `90a58cfeab6aefadc656ce1918e8d0817a1af8caee9c010a82718a684921bf1b`, `39ea38376855190410970f2556a02cdea511c2607534dc36b16cca620da22aa0` and `a5ee78684eba7d2643641d052f8a1e93f516e73ec79782bf397f32dd733551b2`.
