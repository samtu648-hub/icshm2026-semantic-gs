# Native AbsGS replay after the new control discrepancy

Registered2026-09-21 under the user's explicit continuing-research authorization.
Historical RGB152 holdout PSNR is19.383911dB, while fresh disabled-dropout RGB156
is18.391991dB, with a large region-1 difference and near-total156 occlusion.
Source manifests, original initialization, image roles and upstream CLI arguments
(except output path) match. The r=0 engineering forward/gradient/RNG tests pass.
These checks do not identify the cause of long-trajectory divergence.

RGB158 repeats the original `run_rgb152_spatial` path with its original worker,
without importing or installing GaussianDropout. Keep source, initialization,
seed0, v1 roles,30k schedule, optimizer, losses and fixed endpoint unchanged.
Do not choose between completed endpoints for reporting. Evaluate all383 saved
cameras after the endpoint is sealed and compare with both RGB152 and RGB156.
Report all three RGB metrics, regions, point counts, state/source hashes and
time. A third outcome cannot alone prove the cause or statistical stability;
its purpose is to test whether severe failure also appears with the unwrapped
path. No new semantic fit, parameter change or automatic seed sweep follows.

Run after the already queued component pair and frozen-semantic probe to keep
GPU stages serial. Estimate45--60 minutes plus evaluation and2GiB output.
