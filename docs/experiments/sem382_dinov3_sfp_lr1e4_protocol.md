# SEM382 DINOv3 SFP learning-rate protocol

## Question

Does increasing the initial learning rate of the complete trainable SEM381 SFP,
including its LayerNorm parameters, from `1e-5` to `1e-4` recover the quality
lost by SEM381 while every other training and inference input remains fixed?

This is a one-variable optimization comparison against SEM381. It does not
retest DINOv3 versus Swin-L as a new architecture comparison.

## Frozen inputs and method

- Direct control: `SEM-381-DINOV3-VITB-SFP-ALL300`; performance reference:
  `SEM-378-RGB140-ABSGRAD-ALL300`.
- Reconstruct the model from the same official ADE checkpoint, local DINOv3
  source and adapter seed. Do not load or continue either SEM381 checkpoint.
- Require initial trainable-state SHA256
  `cc677af8686be7df215b0d8b1a3f82efe6ff114d23d9c6b5fd543d538b86b995`
  and initial/final frozen-DINO SHA256
  `032a9ca331800355f69e7be93db33f564fff28a3f9f43736c89b422d3e3b7445`.
- Reuse the exact SEM381 6000-entry image/crop/flip plan with SHA256
  `732cf8aabe69e809dbb7846588b5593daa7c1abf980ca8e0e672228d0548c8d0`.
- Keep RGB140, all 300 labels, seed `20260805`, native 768 crops, horizontal
  flip, batch 1, Mask2Former losses, AdamW, weight decay, global-norm clipping
  `0.01`, poly power `0.9`, and 768/512 sliding inference unchanged.
- Keep hidden states 3/6/9/12 and SFP strides 4/8/16/32 with channels
  192/384/768/1536 unchanged. DINOv3 stays in eval mode without gradients.

## Single treatment and optimizer audit

- All 3,842,496 trainable SFP parameters use initial LR `1e-4`.
- This set must include every parameter under both SFP adapters and all four SFP
  LayerNorm modules. Parameter names, tensor counts, element counts, initial
  learning rates and weight decays are recorded before training.
- All 20,250,310 trainable decoder parameters remain at initial LR `1e-4`.
- All 85,660,416 DINOv3 parameters remain frozen and absent from the optimizer.
- The per-update journal records actual SFP and decoder LR values. Initial/final
  norms and update norms are reported separately for SFP and decoder.

## Ordered execution

1. Bind config, code, SEM381/SEM378 endpoints, RGB, labels, split, sampling plan,
   local DINOv3 files, initial-state hashes and frozen artifacts.
2. Reconstruct the model in a fresh process, audit optimizer membership, and run
   eight successful native-768 updates as a numerical/resource check. This is
   not a quality gate and does not select a treatment.
3. Reconstruct the same initial state again and run exactly 6000 successful
   updates. Save checkpoints at 3000 and 6000; do not resume from SEM381 or from
   the 3000 checkpoint.
4. Infer all 544 fixed RGB140 inputs independently from both checkpoints. Treat
   6000 as the primary endpoint and 3000 only as training-progress analysis.
5. Verify finite checkpoints, frozen DINO state, fresh fixed-view probabilities,
   fresh 544-view RGB rendering, frozen artifacts and immutable controls.
6. Independently compare SEM382-6000, SEM382-3000, SEM381 and SEM378 using
   aggregate/per-class/per-view metrics, corrected/harmed pixels, cable FP/FN,
   2 px boundary F1, fixed regions, 100 unlabeled views and 144 trajectory frames.
7. Inspect views 159/160/146/057/064, trajectory 96--98 and unlabeled view 350
   before writing the conclusion. End the task after delivery; do not launch a
   follow-up training run.

## Interpretation

All scores on the 300 labeled images are training fit, not independent held-out
accuracy. The 100 released unlabeled images and 144 trajectory frames have no GT
and support qualitative statements only. Loss reduction or a favorable 3000
checkpoint cannot override the preregistered 6000 primary endpoint. A broad
`stay_cable` label region is not evidence of individual physical cable recovery.

Preserve failures for source/hash drift, missing optimizer parameters, DINO
gradients/state changes, non-finite updates, changed sampling/RGB, OOM or output
collisions. Do not alter the treatment to rescue a failure.
