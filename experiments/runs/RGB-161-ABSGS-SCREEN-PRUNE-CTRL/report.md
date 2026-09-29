# RGB-161-ABSGS-SCREEN-PRUNE-CTRL

Status: `completed_smoke_control_full_not_run` on 2026-09-22.

The native control completed the fixed 3200-step causal smoke from the same
78,045-point training-only initialization, 353/30/17 spatial-v1 roles and seed
as RGB162. The first sandboxed launch stopped before optimization because
`nvidia-smi` could not access the driver; it is preserved as `smoke-r1`.
`smoke-r2` completed in 230.576 seconds.

At the two events after the 3000-step screen threshold became active, native
append reset every `max_radii2D` value. Both observed append sites had oversized
parents before append and zero oversized values afterwards. The final smoke
model contained 306,405 Gaussians. Across the 33 fixed bridge views its mean
PSNR/SSIM was 19.098015 dB / 0.762676. This arm is an early control only; a 30k
endpoint was intentionally not trained after RGB162 failed the recovery review.

Training result SHA256:
`52fda37ea67b8a98920537ff899b9ba9aefcb6a3fd2936166e35c09802301aa6`.
It does not replace RGB140, RGB152, or their semantic models.
