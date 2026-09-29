# RGB-154-ABSGRAD-SPATIAL30-V2

Completed on 2026-09-20 as the AbsGS arm of the zero-overlap spatial-fold
replication. The run used 349 training views, 30 confirmation views, seed 0,
30,000 updates, and the shared 81,516-point training-only initialization.

## Results

| Role | PSNR (dB) | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| Train | 23.302391 | 0.896511 | 0.243230 |
| Confirmation | 20.327519 | 0.882148 | 0.285409 |
| Region 1 | 20.028750 | 0.899524 | 0.320450 |
| Region 2 | 21.658623 | 0.876955 | 0.265967 |
| Region 3 | 19.295185 | 0.869964 | 0.269809 |

Training took 2,775.318 seconds and produced 801,066 Gaussians. Median
full-resolution render time was 0.015170 seconds. The final checkpoint and PLY
SHA256 values are `217654d6c7169b8b709b19ef321a94ae0c4534e009fc8505b0f0133ed1edd0ac`
and `90efeaa37f578a647003040ab1bf9f27628969c3bb51538947751a466c770c9d`.

Relative to RGB155, confirmation PSNR/SSIM/LPIPS changed by
+0.220856 dB/+0.000656/+0.003657. Thus the pre-registered RGB PSNR direction
passes, while LPIPS is slightly worse and regional evidence is mixed.
