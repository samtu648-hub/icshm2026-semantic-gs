# RGB-159-ABSGS-TRACE-A

Registered 2026-09-21; completed. Identical native AbsGS 30k diagnostic repeat
A, seed 0, spatial v1, training-only 78,045-point initialization. Read-only
trajectory and lineage instrumentation. Output:
`outputs/quality/RGB-159/full-r1`.
[Protocol](../../../docs/experiments/absgs_divergence_diagnostic_protocol.md).

Native training completed in 2,795.221 seconds (2,809.314 seconds including
wrapper/audit), with peak allocated/reserved 1,957.714/4,600 MiB. Final count is
792,639; all tensors/PLY fields are finite. Train PSNR/SSIM/LPIPS is
23.296920/0.897344/0.242325; holdout is
19.818818/0.834536/0.340521; region-1 PSNR is 18.599764 dB.

Relative to the identical B arm, state and gradient hashes first differ at step
1, sampled RGB at step 2 and topology/count at step 600 while RNG remains exact.
The full paired interpretation, contribution counterfactuals and independent
audit are recorded in the QUAL018 report. This arm is diagnostic only and does
not replace any frozen reconstruction model.
