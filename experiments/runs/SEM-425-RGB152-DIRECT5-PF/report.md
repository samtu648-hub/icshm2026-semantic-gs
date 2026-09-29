# SEM-425-RGB152-DIRECT5-PF

Status: completed engineering preflight; 22/22 independent checks passed.

This preflight established the minimum native direct-semantic path on the
frozen RGB152 model. It allocated five zero-initialized logits for each of
789,281 Gaussians, composited per-Gaussian probabilities with residual
transmittance assigned to background, and completed eight training-only crop
updates.

The first gradient norm was 0.012494135, probability sums differed from one by
at most `2.3841858e-7`, and peak reserved GPU memory was 1,894 MiB. RGB state
hashes and a fresh RGB render were exactly unchanged. Checkpoint state and
semantic render replay were exact. Confirmation and buffer RGB/label accesses
were all zero.

The independent audit passed all 22 checks and authorized only a separately
registered full spatial comparison, SEM426. This is engineering evidence, not
a quality result.

Evidence:

- `outputs/quality/SEM-425/preflight-r1/result.json`
- `outputs/quality/SEM-425/preflight-r1/audit.json`
- `docs/experiments/sem425_direct5_preflight_protocol.md`
