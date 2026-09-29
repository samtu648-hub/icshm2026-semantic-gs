# RGB-160-ABSGS-TRACE-B

Registered 2026-09-21; completed. Identical native AbsGS 30k diagnostic repeat
B, with the same seed 0, data, source and instrumentation as RGB159. No quality
selection or method change. Output: `outputs/quality/RGB-160/full-r1`.
[Protocol](../../../docs/experiments/absgs_divergence_diagnostic_protocol.md).

Native training completed in 2,748.910 seconds (2,769.222 seconds including
wrapper/audit), with peak allocated/reserved 1,961.667/5,000 MiB. Final count is
793,519; all tensors/PLY fields are finite. Train PSNR/SSIM/LPIPS is
23.289016/0.897259/0.242705; holdout is
19.957586/0.836824/0.336606; region-1 PSNR is 18.535791 dB.

The holdout difference relative to A is `+0.138768 dB` PSNR,
`+0.002288` SSIM and `-0.003915` LPIPS. The paired trajectory first differs in
state/gradients at step 1 and topology at step 600 without RNG divergence. The
full causal boundary is in the QUAL018 report; this arm is not a selected
production replacement.
