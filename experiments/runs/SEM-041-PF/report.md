# SEM-041-PF — restricted direction-residual engineering smoke

- Status: **completed engineering preflight; not a quality baseline**.
- Parent: `SEM-001`, `SEM-004-DIAG`, `SEM-040-EQ`.
- Run commit/worktree evidence: `39cd07de5b60646e0c37db5f02562a10ccad0602` / `b731ee6567007bb92923bba4754967a48dbd0d8cfd4fb13186bd9b1d08806ca6`.
- External result/output-manifest SHA-256: `282583abd7cc1add2e0fe99ab4872f8b3a082c40eb9268ce6a11d60baf6fa19a` / `c1a855f1279a7ccb0e0dad4cabb0e93f79fa456e27de1c7ef17638c3f960a0c7`.

The runner completed 16 training-side iterations. Only `direction_coefficients[7069,3]` had gradients; all Scaffold parameters and frozen SEM-001 static logits remained immutable. There was no NaN, Inf or OOM, and peak PyTorch allocation was 792 MiB.

The residual checkpoint is 144,829 bytes, SHA-256 `8f2adeeb8a15fa2ed65fac4d2511d2bf4baae609bb95c09a51347cf6d55b3b12`, and reloaded bit-identically. Frozen RGB artifact hashes and three camera RGB hashes remained equal with maximum pixel difference `0`. The external official `140.png` `CAMERA_FROM_WORLD` camera produced 1320×989 RGB, `[5,989,1320]` logits and a single-channel uint8 0–4 PNG without reading reference RGB or labels. All gates required to start the registered 10k screen passed.
