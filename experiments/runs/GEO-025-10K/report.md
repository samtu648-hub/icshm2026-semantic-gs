# GEO-025-10K - Fresh continuous Scaffold + VGGT-Omega P0 initialization

Status: `not_started_parent_coordinate_gate_failed`.

This run may start only after full training-side geometry, operational-v2
fusion, and Scaffold initialization reload pass. The only major comparison
variable versus `GEO-000-10K` is the initialization PLY; cameras, split, seed,
resolution, loss, evaluator, and continuous 0-to-10k schedule remain fixed.

`GEO-025-P0` generated all planned files but failed its first strict final
coordinate audit record. Per the pre-registered stop policy, no fusion PLY or
candidate scene was created and this 10k run never started. No RGB metric or
quality conclusion exists for VGGT-Omega initialization.
