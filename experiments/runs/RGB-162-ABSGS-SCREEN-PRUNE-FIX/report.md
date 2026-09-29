# RGB-162-ABSGS-SCREEN-PRUNE-FIX

Status: `stopped_after_smoke_large_negative` on 2026-09-22.

The only scientific change from RGB161 preserved existing `max_radii2D` values
across clone/split append and initialized only new children to zero. The fixed
3200-step smoke completed in 220.398 seconds. All four eligible append
observations retained oversized histories, so the intervention changed exactly
the intended lifecycle.

The effect was destructive. At the first active event, screen pruning removed
291,067 points after opacity pruning; point count fell from 354,740 before the
event to 68,599 afterwards. The smoke endpoint had 68,599 Gaussians versus the
control's 306,405. Across 33 fixed bridge views mean PSNR fell from 19.098015 to
16.366904 dB and mean SSIM from 0.762676 to 0.725491. View 390 fell 9.424304 dB;
052/051 fell 6.278049/5.814827 dB. Visual review shows large holes, dark fields,
lost bridge detail and smeared tower/deck structure. Views 155/156 gained PSNR
but lost SSIM, so isolated gains do not rescue the candidate.

Training result SHA256:
`e2c309b624fa9c8f6fdd04c5d26bb4f7cb77f64101168e188e130c7a36660952`.
The 30k run and matched semantic training were not started. The failed candidate
is retained as mechanism evidence and must not be promoted.
