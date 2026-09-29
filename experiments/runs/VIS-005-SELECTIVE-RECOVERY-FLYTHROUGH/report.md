# VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH — 0.30 / SEM-123 recall / SEM-123 conservative same-pose comparison

## Status

Completed on 2026-09-03 as a developer-only visual check.

## Contract

The video will use exactly the 144 novel `CAMERA_FROM_WORLD` and
`SIMPLE_RADIAL` poses from VIS-001. It renders frozen Scaffold RGB once per
pose, uses the frozen SEM-001 sidecar and SEM-102 teacher, and then applies
only three already fixed rules:

- hard protected threshold 0.30 (the current developer visual baseline);
- SEM-123 score leader `p20_d02_any`;
- SEM-123 conservative Pareto `p20_d04_sem001_stay_cable`.

It reads neither labels nor reference RGB, has no presentation-only cleanup,
and cannot alter these rules after looking at frames. Deck, tower and
foundation are audited against the hard-0.30 stream in every output frame;
the recovery streams must also never remove an existing hard-0.30
`stay_cable` pixel.

## Command

```bash
"$R1_ENV_ROOT/bin/python" scripts/render_vis005_selective_recovery_flythrough.py \
  --config experiments/runs/VIS-005-SELECTIVE-RECOVERY-FLYTHROUGH/config.yaml \
  --sem001-config experiments/runs/SEM-001/config.yaml \
  --sem102-config experiments/runs/SEM-102-RENDERED-SEG-FINAL/config.yaml \
  --sem123-result "$ICSHM_OUTPUT_ROOT/r2/sem_123/band_recovery_dev_sweep/full-20260903-r2/result.json" \
  --sem001-checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --sem102-final-result "$ICSHM_OUTPUT_ROOT/r2/sem_102/final_training/final-20260902-r1/result.json" \
  --sem102-checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_102/final_training/final-20260902-r1/checkpoints/segformer_b2_final.pth" \
  --sem102-model-dir "$R1_MODEL_ROOT/segformer-b2-cityscapes" \
  --rgb-run-dir "$ICSHM_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --trajectory-dir "$ICSHM_OUTPUT_ROOT/results/frozen_scaffold_sem001/flythrough_v1/cameras" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/vis_005/selective_recovery/flythrough-20260903-r1"
```

## Result

All 144 pose-matched triplets and the 6.0-second, 24-fps, 3960×1032 H.264
video completed. The video SHA-256 is
`a41dcdb66ee71daea92de3e9acd97c372350d2f81899896cd49e4812dab6990a`; the
render manifest SHA-256 is
`225784f050a5921793942921b205c94d19357e0f648066451951602aaab1dce5`.

The manifest verifies the official `CAMERA_FROM_WORLD` + `SIMPLE_RADIAL`
contract, 144 unique poses, no labels/reference RGB, frozen RGB/checkpoints,
valid five-class PNGs, and exact deck/tower/foundation handling. It also
verifies that neither recovery rule deletes an existing hard-0.30
`stay_cable` pixel. Recovering with `p20_d02_any` adds 242,108 `stay_cable`
pixels over hard 0.30 on this novel trajectory; the conservative
`p20_d04_sem001_stay_cable` adds 98,369.

Visual review of the highest-addition frames (recall candidate frame0091;
conservative candidate frame0031) finds the new yellow pixels concentrated
along already predicted cable/tower regions, without the broad extra yellow
background patch that VIS-004 exposed for hard 0.20. This is a useful visual
sanity check, not a labelled quality measurement: it cannot establish
accuracy, generalisation, a formal replacement or a final submission choice.
SEM-111 remains the historical formal candidate and `RGB-030-FULL
Scaffold-GS + SEM-001` remains the rollback.
