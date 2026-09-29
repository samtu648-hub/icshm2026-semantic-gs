# RGB-166 ImprovedGS 50k horizon curve

Status: completed; 50k selected as the strongest released-all400 bridge-fit endpoint, research candidate only.

RGB-166 inherits RGB-165 and changes the configured training horizon from 30,000 to 50,000 iterations. It saves 30k, 35k, 40k, 45k and 50k endpoints. All 400 released photographs, official cameras, 72,292-point initialization, seed, one-million budget and EAS+LAS+RAP+MU components remain fixed. Training completed in 1,382.719 seconds with 931,969 Gaussians and 6,570 MiB peak reserved memory.

The same-run 30k endpoint is the authoritative control for the horizon curve. It is not byte-identical to the historical RGB-165 30k endpoint because the configured total horizon also controls the exposure schedule and dense optimizer allocation. The two 30k endpoints nevertheless have nearly identical all-400 metrics. Historical-parent comparisons are descriptive; 30k-to-50k conclusions use the single RGB-166 trajectory.

## Result

All-400 metrics improve monotonically from 30k to 50k: PSNR 23.428528 to 23.490679 dB, SSIM 0.902798 to 0.904821, and LPIPS 0.227520 to 0.221597. The fixed 385--392 manual bridge crops improve by 0.369768 dB and 0.008051 SSIM. The wider difficult groups also improve: 159--160 gains 0.096891 dB / 0.002592 SSIM / -0.011698 LPIPS; 235--245 gains 0.397036 dB / 0.011151 / -0.015196; and full-frame 385--392 gains 0.255844 dB / 0.007686 / -0.014316.

The 300 labeled views provide a bridge-specific check. From 30k to 50k, region PSNR rises by 0.220576 dB for deck, 0.403523 for `stay_cable`, 0.567452 for tower and 0.533013 for foundation. Every class reduces region MAE, gradient error and boundary-gradient error. `stay_cable` local-contrast retention rises from 0.789204 to 0.799780 and visible support from 0.691303 to 0.702513. These improvements are consistent but modest: about 29.75% of reference-visible cable pixels still fail the fixed half-contrast support test at 50k.

The fixed 144-pose trajectory has no RGB ground truth and is therefore qualitative. Visual review finds no new gross instability at 50k. Mean image change shrinks from 0.833694/255 for 30k-to-40k to 0.675071/255 for 40k-to-50k, consistent with diminishing refinement. Persistent water haze, bridge/background mixing and missing or broken thin cables remain visible.

## Decision

Select 50k when producing the current full-data ImprovedGS bridge-fit candidate. Do not extend this step sweep to 60k: the measured gains are incremental and additional optimization cannot create missing three-dimensional cable support after densification ended at 15k. RGB-166 does not replace the production RGB140+SEM378 chain and does not authorize matching semantic training, because the 400 scored photographs participated in reconstruction and the 144-pose trajectory lacks RGB truth.

The next reconstruction task is a bridge-structure support diagnosis across all labeled cable regions and the known difficult groups. It must determine whether missing cables have no Gaussian support, have displaced support, or have valid support suppressed by scale/opacity. The next candidate should then choose line-aware multi-view initialization/densification for absent geometry, or structure-weighted optimization and scale/opacity treatment when support already exists.

Evidence: `outputs/quality/RGB-166/full-r1/summary-r1.json`, `evaluation-r1/result.json`, `evaluation-r1/montages/`, `trajectory-r1/result.json`, and `trajectory-r1/review.jpg`.
