# VIS-004-DEV-THRESHOLD-FLYTHROUGH — SEM-111 / 0.20 / 0.30 same-pose semantic comparison

## Status

Completed on 2026-09-03 as an explicitly developer-only visualization.

This developer-only visual experiment renders the existing 144-pose VIS-001
trajectory three times with the same frozen RGB, SEM-001 and SEM-102 inputs:
SEM-111's historical 0.80 threshold, the SEM-120 developer-score leader 0.20,
and the nearby 0.30 Pareto alternative. It is a visual comparison, not another
quality audit, model fit, threshold search or formal replacement decision.

## Contract

- Reuses the 144 VIS-001 `CAMERA_FROM_WORLD`, official-reference,
  `SIMPLE_RADIAL` poses and reads no reference RGB or labels.
- Renders frozen Scaffold RGB once per pose, runs the frozen SEM-102 2D teacher
  on that RGB, then applies the exact existing protected-fusion rule at the
  three registered thresholds.
- Does not change RGB geometry, colour, opacity, checkpoints or any parameters.
  SEM-001 deck, tower and foundation pixels must remain exact in every frame.
- Adds no display-only mask, morphology, threshold or other cleanup. The video
  therefore shows precisely the same rule that the SEM-120 development sweep
  scored on its labelled views.

## Command

```bash
"$R1_ENV_ROOT/bin/python" scripts/render_vis004_dev_threshold_flythrough.py \
  --config experiments/runs/VIS-004-DEV-THRESHOLD-FLYTHROUGH/config.yaml \
  --sem001-config experiments/runs/SEM-001/config.yaml \
  --sem102-config experiments/runs/SEM-102-RENDERED-SEG-FINAL/config.yaml \
  --sem120-result "$ICSHM_OUTPUT_ROOT/r2/sem_120/dev_threshold_sweep/full-20260903-r1/result.json" \
  --sem001-checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --sem102-final-result "$ICSHM_OUTPUT_ROOT/r2/sem_102/final_training/final-20260902-r1/result.json" \
  --sem102-checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_102/final_training/final-20260902-r1/checkpoints/segformer_b2_final.pth" \
  --sem102-model-dir "$R1_MODEL_ROOT/segformer_b2" \
  --rgb-run-dir "$ICSHM_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --trajectory-dir "$ICSHM_OUTPUT_ROOT/results/frozen_scaffold_sem001/flythrough_v1/cameras" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/vis_004/dev_threshold_fusion/flythrough-20260903-r1"
```

## Result

The output contains all 144 pose-matched semantic triplets and a 6.0-second,
24-fps, 3960×1032 H.264 yuv420p comparison video:
`semantic_sem111_vs_dev020_vs_dev030.mp4`. Its SHA-256 is
`863c2321790cdd80a2c8587e407e792deeff313aff99cc038b379b3714413ebb`; the
render manifest SHA-256 is
`a1fc5854419581b1a528cf6108efe24714d5a8b292d7b6e4288e79e14b40d98b`.

The manifest verifies `CAMERA_FROM_WORLD`, `SIMPLE_RADIAL`, 144 unique poses,
no reference RGB/labels, frozen RGB/checkpoints, class range and exact protected
SEM-001 class handling. A recoverable interrupted invocation wrote the semantic
streams before video encoding; the registered `--resume` path validated and
preserved those files rather than overwriting them, then encoded the completed
144-frame stream. Per-frame SHA-256 entries are retained in the manifest.

Visual review confirms the quantitative trade-off rather than hiding it. In
the high-difference frame `frame0091`, 0.20 expands yellow into a broad
low-confidence region behind/right of the tower, while 0.30 keeps the dominant
main-cable lines and removes most of that expanded region. Across the novel
trajectory, 0.20 adds 347,371 `stay_cable` pixels beyond 0.30; 0.30 still adds
1,307,344 beyond SEM-111's 0.80. Deck/tower/foundation maps are exactly the
same across all three streams. The video has no GT, so this is a visual design
observation, not a new accuracy or generalisation claim.

Accordingly, 0.20 remains the pre-declared score leader, while 0.30 is the
recommended **development visual baseline** for the next mechanism: it retains
the main cable recovery visible in the video while avoiding the most obvious
low-confidence expansion. Neither label changes SEM-111's historical formal
candidate status or the SEM-001 rollback.
