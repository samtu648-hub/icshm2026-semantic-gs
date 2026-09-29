# SEM381 DINOv3 ViT-B/16 + SFP protocol

## Question

On the exact frozen RGB140 inputs and all300 label fit used by SEM378, does a
frozen official DINOv3 ViT-B/16 feature source with a trainable simple feature
pyramid improve the five-class Mask2Former result enough to justify its costs and
local trade-offs?

This is an architecture-package comparison. DINOv3 pretraining, ViT token
features, the required SFP, parameter count and Swin removal change together, so
the result is not a pure pretraining or backbone-capacity attribution.

## Frozen control

- Parent/control: `SEM-378-RGB140-ABSGRAD-ALL300`.
- RGB: byte-identical 544 RGB140 PNGs, including 400 released views and the same
  144 official-coordinate trajectory frames.
- Labels/split: the same 300 labels, manifest
  `data/manifests/contest_dataset_v1.json`, split
  `released_all400_rgb_all300_sem_v1`.
- Seed and sampling: seed `20260805`; the full 6000-view/crop/flip sequence must
  exactly match SEM378. No SEM379 GT region sampling is allowed.
- Optimisation and loss: Mask2Former author set/mask/dice loss, AdamW,
  poly-0.9, global-norm clipping 0.01, batch 1, no accumulation, native 768
  crops, 512 sliding inference and checkpoints at 3000/6000 successful updates.

## Single treatment

- Load `facebook/dinov3-vitb16-pretrain-lvd1689m` from the local directory named
  by `DINOV3_VITB16_DIR`; network access is forbidden.
- Require the registered 342,662,192-byte safetensors file with SHA256
  `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
- Freeze all 85,660,416 DINOv3 parameters and keep that module in evaluation
  mode. Use hidden states 3/6/9/12 after removing CLS plus four register tokens.
- Train a deterministic SFP that maps stride-16 token grids to Swin-compatible
  `res2..res5` shapes: strides 4/8/16/32 and channels 192/384/768/1536.
- Build the existing ADE-initialized Mask2Former, retain its compatible pixel and
  transformer decoder tensors, discard Swin-L, and preserve the same random
  five-class classifier adaptation as SEM378.
- DINOv3 uses zero learning rate/no gradients; SFP uses the registered backbone
  multiplier (1e-5); the decoder uses 1e-4.

## Ordered execution

1. Bind config, code, RGB, labels, SEM378 endpoints, sampling plan, local DINOv3
   file hashes and DINOv3 license before GPU work.
2. In a fresh process, prove no network loading, no missing/unexpected DINO keys,
   exact feature shapes, frozen encoder state, finite loss/gradients and eight
   successful native-768 updates. Record peak allocated/reserved VRAM.
3. If the preflight fits the RTX 5070 without reducing resolution or capacity,
   start one independent 6000-successful-update run from the registered sources.
   Preserve any failure; do not silently retry with a changed method.
4. Reload the 6000 checkpoint and infer all 544 images with the same 768/512
   sliding protocol. Verify both checkpoints are finite and freshly rerender all
   RGB140 views byte-exact after training.
5. Independently recompute all300 confusion matrices, per-class IoU/precision/
   recall/FP/FN, per-view deltas, corrected/harmed/wrong-to-wrong pixels,
   stay-cable FN/FP changes, 2px boundary F1, runtime, VRAM and model size.
6. Produce the fixed nine views, all 100 unlabeled views and all 144 trajectory
   frames at matching locations, then inspect 159/160/146/057/064, trajectory
   96--98 and unlabeled 350 before writing the conclusion.

## Interpretation and stop rules

All 300 scored images are training fit. The 100 released unlabeled images and
144 trajectory frames have no GT and permit qualitative statements only. A wide
`stay_cable` annotation is not evidence of per-cable geometric reconstruction.
One seed cannot establish stability or hidden-view generalization.

OOM, non-finite updates, source/hash drift, decoder-load mismatch, changed RGB,
changed sampling or an unfrozen DINO encoder stops the run and preserves the
failure. Do not add RGB148, region sampling, experts, fusion, post-processing,
lower resolution, layer scans, learning-rate scans or endpoint selection inside
SEM381. SEM378 remains the rollback path regardless of the result.
