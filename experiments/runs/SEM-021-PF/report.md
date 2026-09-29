# SEM-021-PF — cumulative contribution-gate smoke

- Status: completed preflight, not a semantic quality baseline.
- Output: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/sem021_preflight/`; `result.json` SHA-256 `99cae7b5f31279c485ab179ff21a8dcf359f3d8cdcc1c4d108a9f60444f55faa`.

The eight-iteration smoke used the exact SEM-001 initialization, 219-view training sampler and cumulative threshold 0.95.  Loss was finite (1.60848 → 1.35889), peak PyTorch allocation was 784 MiB, only semantic logits received gradients, checkpoint reload passed, and all three RGB isolation cameras had identical hashes with maximum absolute pixel difference zero.  An external `CAMERA_FROM_WORLD` official camera produced RGB, `[5,989,1320]` logits and a uint8 0–4 PNG.

The three validation views in this smoke are engineering evidence only and must not be compared with the 74-view SEM-001 result.
