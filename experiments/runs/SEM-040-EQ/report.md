# SEM-040-EQ — zero-residual equivalence gate

- Status: **completed**.
- Parent: frozen `SEM-001` and diagnostic-only `SEM-004-DIAG`.
- Run commit: `39cd07de5b60646e0c37db5f02562a10ccad0602`; tracked worktree diff SHA-256 `b731ee6567007bb92923bba4754967a48dbd0d8cfd4fb13186bd9b1d08806ca6` (pre-existing user changes preserved).
- External result SHA-256: `0e2adb7a4082bad8b68b964578a046a888dc57bef7a4665f62924fb8e50053f7`; output-manifest SHA-256 `dbdc0c330c0d5fb3c636b99c7e50ddf49e113d5b6c9a0b66a17eb2c88cfa7b08`.

The authenticated mask contains exactly 7,069 unique, sorted training-side anchors and produces exactly `7,069×3 = 21,207` zero-initialized trainable coefficients. The SEM-004-DIAG result, output manifest, direction CSV, final SEM-001 checkpoint/result and 74-view CSV all passed their pre-registered SHA-256 gates.

On official cameras 001, 054 and 140, the zero-residual path had maximum probability difference `0.0` versus SEM-001, bit-identical class IDs and RGB maximum absolute pixel difference `0`. No validation label was read. Therefore the later candidate cannot attribute renderer drift to the direction residual.

The first attempt is retained under `$ICSHM_OUTPUT_ROOT/r2/sem_004_s3/equivalence`: it stopped before rendering because the runner compared the registry-style completed status to the artifact's exact diagnostic status. Commit `39cd07d` corrected only that provenance string; the successful retry used the separate `equivalence-r1` directory.
