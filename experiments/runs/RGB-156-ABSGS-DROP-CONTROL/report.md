# RGB-156-ABSGS-DROP-CONTROL

Completed 2026-09-21;701-step engineering and fresh30k control both passed.
Parent RGB152, identical 78045-point initialization, spatial v1 353/30/17,
seed0; dropout disabled. Output: `outputs/quality/RGB-156/full-r1`.
[Protocol](../../../docs/experiments/component_sampling_dropgaussian_protocol.md).

The701-step unscored probe exercised two native densification events and was
discarded. The disabled-dropout path reproduced native RGB exactly, gradients
within5e-5 absolute/0.002 relative tolerance, and global RNG unchanged. Formal
training starts from the original78045 points, not the probe endpoint.

## Fixed endpoint result

- 353 training /30 confirmation /17 buffer, seed0,30k; no holdout optimization.
- Training fit: PSNR23.300438dB, SSIM0.897253, LPIPS-Alex0.242631.
- Confirmation: PSNR18.391991dB, SSIM0.817945, LPIPS-Alex0.382897.
- Region1: PSNR13.842456dB, SSIM0.722848, LPIPS0.571758.
- 793893 Gaussians; runner2655.127 seconds; peak allocated/reserved
  1955.781/6234 MiB. All worker/stage exit statuses0.
- The complete30000 camera trace matches RGB157 exactly; ordinary evaluation
  runs without the dropout hook. Saved RGB CSV aggregates were independently
  recomputed and endpoint hashes checked.

This fresh disabled-dropout control is0.991920dB below historical RGB152 on the
same confirmation set despite nearly identical training fit. Full source
manifest, initial PLY, upstream arguments except output, and data roles match.
View156 is almost fully occluded. Do not substitute the better historical
baseline in the paired comparison or attribute the run divergence to a known
cause. RGB158 separately checks the completely unwrapped native path.
