# SEM-022-PF — relative-maximum contribution-gate smoke

- Status: completed preflight, not a semantic quality baseline.
- Output: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/sem022_preflight/`; `result.json` SHA-256 `388e117b5fabccda0f281bf1677df03f187527f102ca589d601ffd4470185a5a`.

The eight-iteration smoke used the exact SEM-001 initialization, 219-view training sampler and relative-maximum threshold 0.05.  Loss was finite (1.60861 → 1.35646), peak PyTorch allocation was 784 MiB, semantic checkpoint reload passed, and RGB isolation remained exactly zero.  An external official `CAMERA_FROM_WORLD` camera yielded RGB, `[5,989,1320]` logits and a single-channel uint8 PNG whose IDs were restricted to 0–4.

The preflight's three validation views serve only CUDA/alignment/isolation checks; they are not a 74-view quality claim.
