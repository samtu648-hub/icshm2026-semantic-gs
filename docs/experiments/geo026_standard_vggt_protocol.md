# GEO-026/027 — VGGT geometry initialization for Standard 3DGS

## Purpose and historical boundary

`GEO-025-P0` remains a failed preserved GeometryPrior-v2 run: its serialized
float32 world-point roundtrip exceeded the original 0.001-pixel hard gate at
three shallow-depth pixels. This protocol does not relabel that result. It
defines a new canonical representation audit and, only after that audit and
fusion pass, a matched Standard 3DGS comparison aligned with the current
Standard + Mask2Former research direction.

The motivation for changing the downstream backbone is later evidence, not a
retroactive edit to GeometryPrior v2. GeometryPrior v2 fixed Scaffold as the
common receiver before forward results. GEO-027 therefore forms a new matched
pair; it is not added to the old Scaffold ranking.

## Stage A: GEO-026 numeric representation audit

The inputs are the exact preserved GEO-025-P0 `progress.json`, dry-run window
manifest, failure record, diagnosis, and all 7,575 NPZ files. Their four
metadata hashes are fixed in the GEO-026 config. No model, checkpoint, RGB,
label, or GPU is accessed.

Fusion consumes depth, confidence, valid mask, float64 official PINHOLE
intrinsics, and float64 official cam2world. It does not consume the serialized
float32 `world_points` array. GEO-026 therefore recomputes world points from
depth and official cameras in float64 and applies these hard gates to every
valid pixel:

- projection roundtrip error at most `1e-7` pixel;
- depth roundtrip absolute and relative tolerance `1e-10`;
- exact 303-window/7,575-record coverage and 303 training-image isolation;
- original schema, dtype, finite-value, mask, positive-depth, intrinsics and
  pose gates;
- serialized float32 world points remain within the original `2e-5`
  absolute/relative coordinate-fidelity check.

The old serialized-world 0.001-pixel count is retained as a diagnostic. A
GEO-026 pass cannot change the GEO-025 result.

## Stage B: fusion and Standard interface gate

Only a complete GEO-026 pass may materialize an accepted run manifest. Fusion
then reuses the already executed operational-v2 implementation and fixed
parameters: stride 8, three consistency neighbours, 5% relative-depth
tolerance, ordinary support 3 including source, line support 2 including
source, and at most 100,000 points. The fused PLY must use official P0 XYZ plus
original RGB only and must load through the pinned Standard 3DGS reader.

## Stage C: matched Standard 0→10k pair

The two arms are independently fresh, continuous 0→10,000 runs:

- `GEO-027-STD-CTRL-10K`: original training-only DLT PLY;
- `GEO-027-STD-VGGT-10K`: fused VGGT P0 PLY.

Both use Standard 3DGS commit
`54c035f7834b564019656c3e3fcc3646292f727d`, `rgb_r1_spatial_v1`, 303 training
RGB, 74 spatial validation cameras, full-frame 1320×989 PINHOLE images,
official cameras, seed 0, native loss/optimizer/densification, and the same
PSNR/SSIM/LPIPS-Alex evaluator. Checkpoints are recorded at 1k/3k/5k/10k.
The only major variable is `sparse/0/points3D.ply`.

Both arms additionally bind the same recorded R1 source-tree diff, the exact
`dataset_readers.py`, `train.py`, and `render.py` hashes, and the rasterizer and
simple-knn submodule commits. A source mismatch blocks both arms rather than
being repaired differently between them.

At 10k the candidate may advance only if all engineering gates pass and:

- mean PSNR improves by at least 0.2 dB **or** mean LPIPS-Alex improves by at
  least 0.01;
- mean SSIM is no more than 0.002 below control;
- worst-10%-view PSNR and mean 140–146 PSNR are each no more than 0.1 dB below
  control.

Failure or non-promotion stops the branch. Passing authorizes a separately
registered matched 30k Standard pair. A matching Mask2Former model is trained
only after the 30k RGB result passes; no semantic score is inferred from the
geometry or 10k RGB result.

## Preparation and execution result on 2026-09-19

GEO-026 completed all 7,575-record hard gates. The canonical float64
depth/camera roundtrip maximum was `4.206412995699793e-12` pixels and
`4.973799150320701e-14` depth; all 2,009,011,200 valid pixels were audited.
The diagnostic serialized-world path had 734 pixels above the historical
0.001-pixel limit, confirming that GEO-025-P0 must remain failed while the new
canonical representation can independently pass.

Operational-v2 fusion produced 100,000 points from 707,330 raw candidates.
The candidate PLY hash is
`b1debba524689d1cacbb22d4379fa8402b054d9143f9f848e1f22b1bb7059cbd`.
The pinned Standard reader loaded both scenes successfully: control 72,292
points and candidate 100,000 points, each with 303 training and 74 validation
cameras. After the host driver recovered, the shared-lock, source, disk and GPU
admission gates passed and both arms completed one fresh continuous 0-to-10k
process. All four checkpoints restored, all checkpoint and evaluated-PLY
floating values were finite, each endpoint produced all 74 validation renders,
and both 10k PLYs passed the official external-camera rendering gate.

At 10k, candidate minus control was `+1.129312` dB mean PSNR, `+0.002961`
mean SSIM, and `+0.004454` mean LPIPS-Alex. The candidate also improved the
separately aggregated worst-10%-view PSNR by `+0.304391` dB. However, mean
PSNR on the pre-registered 140.png--146.png subset fell by `2.229914` dB,
far beyond the allowed 0.1 dB regression. GEO-027 therefore does not advance;
no matched 30k pair or Mask2Former training is authorized. This remains a
single-split, single-seed development result and does not change the preserved
GEO-025-P0 failure.
