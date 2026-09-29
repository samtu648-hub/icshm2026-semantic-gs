# RGB-155-STANDARD-SIGNED-SPATIAL30-V2

Completed on 2026-09-20 as the original Standard signed-gradient control for
RGB154. All data roles, cameras, initialization, seed, losses, optimizer,
schedule, and 30,000-update budget were matched.

## Results

| Role | PSNR (dB) | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| Train | 23.202952 | 0.893083 | 0.251809 |
| Confirmation | 20.106664 | 0.881492 | 0.281751 |
| Region 1 | 19.715653 | 0.899427 | 0.305594 |
| Region 2 | 21.801421 | 0.877109 | 0.266785 |
| Region 3 | 18.802917 | 0.867940 | 0.272875 |

Training took 1,943.792 seconds and produced 661,020 Gaussians. Median
full-resolution render time was 0.010364 seconds. The final checkpoint and PLY
SHA256 values are `41d73d9ffa5f98f8dc95af24308010a4f2ae134577a04da77ce543b3f295a175`
and `4aec1b04a7d7ff8c44c5c2a6f11f618b292b27fccdd76dabc001bdbf9ae66389`.

Standard is the efficiency fallback: AbsGS used 42.78% more reconstruction
time, 21.19% more Gaussians/PLY bytes, and 46.36% more median render time.
