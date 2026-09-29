# GEO-026-P0-F64 — VGGT canonical-depth float64 P0 audit

Status: `completed_passed_for_fusion_and_standard_interface`.

This is a new experiment and does not rewrite the failed `GEO-025-P0`
coordinate gate. It reuses the immutable 303-window/7,575-record output without
running VGGT-Omega again. The hard-gate representation is the representation
actually consumed by fusion: stored float32 depth combined with stored float64
official intrinsics and cam2world, backprojected and projected in float64.

The redundant serialized float32 `world_points` array remains schema- and
fidelity-checked, and its historical 0.001-pixel roundtrip is reported, but it
does not decide this new gate because fusion reconstructs points from depth and
official cameras. All thresholds and source hashes are frozen in `config.yaml`
before the full audit.

The full CPU audit completed over all 7,575 records and 2,009,011,200 valid
pixels. Coverage was exact: 303 windows, 303 unique training images, no
duplicate pair, and no validation/guard/test access. The canonical float64
depth/camera path reached maximum pixel and depth roundtrip errors of
`4.206412995699793e-12` pixels and `4.973799150320701e-14`, respectively,
well inside the frozen limits. Serialized float32 world points remained within
the fixed coordinate-fidelity bound, while their diagnostic-only historical
0.001-pixel roundtrip was exceeded by 734 pixels with maximum
`0.0021170910478645055`; this preserves and explains rather than relabels the
GEO-025-P0 failure.

The accepted records were then fused with the previously fixed operational-v2
settings. Fusion reduced 707,330 raw candidates to 100,000 points, including
24,814 line-structure points; support min/median/max was 2/4/4. The resulting
PLY SHA256 is
`b1debba524689d1cacbb22d4379fa8402b054d9143f9f848e1f22b1bb7059cbd`.
The pinned Standard reader loaded all 100,000 points as finite and recovered
the exact 303/74 train/validation cameras from the candidate scene.

This result is a numeric, fusion, and interface pass only. No Standard
training, RGB score, semantic training, or semantic score has been produced.
The original GEO-025-P0 failed result remains unchanged.

Artifact hashes:

- audit result: `81a75af0d1519326d233e8db9a09c7a776a7fe10c37119fc6329329658ad8470`
- accepted records: `dc643dd7e962bdcac542123f3c8fcd4ed8b9a53fc1114710b977dc7247f10748`
- fusion manifest: `eb87eff73414020adc80846a57668ed911e8e0e6c82ed41ae4f32ade91595ff6`
- fused NPZ: `637204c4ba09c3696a7e6cdb0c24211746adf060d38691f7db2c8f2dd7242f58`
