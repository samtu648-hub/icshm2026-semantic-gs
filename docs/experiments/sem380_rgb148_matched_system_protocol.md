# SEM380 RGB148 Matched Mask2Former System Protocol

Date: 2026-09-16

## Question

Test whether the frozen RGB148 reconstruction plus an independently trained, input-matched Mask2Former is worth retaining as a complete system candidate against RGB140 plus SEM378.

This is a complete-system comparison. It does not isolate the effect of region RGB supervision from RGB148's larger Gaussian population, changed optimization trajectory, or semantic training stochasticity. No RGB147 semantic control, capacity-matched reconstruction, spatial holdout, or multi-seed arm is added in this run.

## Frozen Arms

- Baseline: frozen RGB140 plus frozen SEM378.
- Candidate: frozen RGB148 plus new SEM380.
- Default remains RGB130 plus SEM371 regardless of this run's result.

RGB148 and RGB140 use the same released official 400-camera PINHOLE reference frame and native 1320x989 resolution. The candidate reads the already completed RGB148 endpoint and does not modify geometry, color, opacity, cameras, or reconstruction files.

## Attempt Record

The `r1` attempt stopped during CPU-only preparation because the frozen 18-artifact manifest was resolved relative to the repository instead of the explicit artifact root. It created no GPU process, checkpoint, or semantic prediction. Its failure record is retained. The corrected `r2` attempt changes only that path resolution and remains bound to this protocol.

## Matched Semantic Training

SEM380 must match SEM378 in all semantic-training choices:

- the same 300 labels, dataset manifest, and `released_all400_rgb_all300_sem_v1` split;
- independent initialization from the same official ADE20K Mask2Former Swin-L weights;
- seed 20260805;
- 6000 successful updates;
- uniform view selection, native 768x768 random crops, and horizontal flip probability 0.5;
- no resize or color augmentation in the executed tensor pipeline;
- the same optimizer, schedule, losses, AMP retry policy, and 3000/6000 checkpoints;
- native 768/512 sliding-window probability averaging and five-class argmax inference.

The only major semantic-training variable is the frozen RGB input. SEM379's cable-region sampler, DINO features, fusion, new losses, morphology, and postprocessing are forbidden.

Before full training, the run must verify the config, manifest, split, all label hashes, all 544 RGB input hashes, official trajectory metadata, core source hashes, pretrained weights, initial model state, resolved model recipe, and the first engineering updates. The full 6000 sampling trace must match SEM378 exactly.

## Evaluation

Quantitative results use all 300 labeled training views and are training-fit evidence only. Report five-class mIoU, per-class IoU/precision/recall/FP/FN, paired per-view changes, two-pixel annotated `stay_cable` region-boundary metrics, and cable FN/FP transition counts.

The fixed nine views and existing same-location difficult/structure boxes remain unchanged. Emphasize 159/160, 146, 057/064, trajectory frames 96-98, and unlabeled view 350. The 100 released unlabeled views and 144-frame trajectory have no semantic GT and are qualitative change evidence, not unknown-view accuracy.

The review must retain the photo, GT where available, both RGB renders, both semantic predictions, and error/change overlays. RGB and semantic changes must be reported jointly as descriptive association, not causal attribution. A filled `stay_cable` sector annotation is not an individual physical-wire mask.

Report reconstruction and semantic model sizes, reconstruction and semantic training time, rendering/inference time, and peak memory. RGB148's additional Gaussians and rendering cost are part of the complete-system tradeoff.

## Stop Boundary

Complete this one training, inference, evaluation, review, and documentation chain. Do not automatically start capacity matching, line reconstruction, Scaffold migration, parameter sweeps, DINO, fusion, or an RGB147 semantic control.
