# SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300

Status: completed spatial-prior mechanism ablation; not promoted.

- Parent/direct control: `SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300`
- Performance reference: `SEM-378-RGB140-ABSGRAD-ALL300`
- Protocol: [SEM383 DINOv3 spatial-prior SFP](../../../docs/experiments/sem383_dinov3_spatial_sfp_protocol.md)
- Config: [config.yaml](config.yaml)
- Output: `outputs/quality/SEM-383/r2`
- Scope: all300 training-view fit; unlabeled100 and trajectory144 qualitative only.
- Git: `49cdcdb1c9915a9544d568754af95ea20a85ec9c`
- Seed: `20260805`
- Split: `released_all400_rgb_all300_sem_v1`

## Registered treatment

SEM383 keeps the complete SEM382 SFP and adds one 514,048-parameter RGB
spatial-prior residual. Four convolutional levels at strides 4/8/16/32 are
projected to the existing SFP channel contract and fused through per-channel
gates initialized to `1e-3`. Existing SFP, the new branch and the Mask2Former
decoder use initial LR `1e-4`; DINOv3 remains fully frozen with LR zero.

This is a custom deformable-free spatial-prior residual inspired by the
convolutional spatial prior in Meta's official adapter. It is not the Meta
DINOv3 Adapter and not ViT-Adapter: it omits deformable Extractors, ConvFFN,
level embeddings and custom attention CUDA operators.

Checkpoint 6000 is the preregistered primary endpoint. Checkpoint 3000 is
retained only to analyze training progress and was not selected after observing
quality.

## Execution

- Attempt `r1` completed preparation but the no-GT input diagnostic stopped
  before any model forward or training with `ValueError: Artifact escapes root:
  350.png`. The failed diagnostic is preserved. Attempt `r2` only corrected the
  artifact-root handling and is the valid run.
- Preparation bound the config, split, 544 RGB140 images, 300 labels, three
  diagnostic photos, four local DINOv3 files, SEM382/SEM378 controls, the exact
  6,000-entry sampling plan and 18 frozen artifacts. Shared initial trainable
  tensors matched SEM382; the complete initial trainable-state SHA256 was
  `a548d9998ca4cf2b992a292f9a768a61f61030d3f202c8f009fe2e62d8c73d3b`.
- The native-768 preflight completed eight successful updates. It verified all
  4,356,544 SFP/spatial parameters and 20,250,310 decoder parameters at initial
  LR `1e-4`, all 85,660,416 DINOv3 parameters frozen and absent from the
  optimizer, and finite nonzero gradients in every registered spatial stage,
  projection and gate.
- Formal training completed 6,000 successful updates in 840.509 s (14.009
  min), with two standard AMP skips. Peak memory was 2,542.958 MiB allocated
  and 2,764 MiB reserved. SFP/spatial update L2 was 11.308871 (14.193195%
  relative); decoder update L2 was 16.239658 (5.487263% relative).
- The 3,000-step checkpoint is 441,297,587 bytes with SHA256
  `fd3461cd30e1d1bac347ae7c64e42eb87bf27b80b5642ac616be407edbc19479`.
  The 6,000-step checkpoint has the same size and SHA256
  `d35895a4f6adb335ea1333724f3cbf3c5ccd9cfc1715809862c5fc48f70ba8d3`.
- Each checkpoint produced 300 labeled, 100 unlabeled and 144 trajectory
  predictions. Measured forward time was 185.532 s at step 3,000 and 185.607 s
  at step 6,000; median per-image times were 0.339902 s and 0.340069 s.
- Both checkpoints were finite. Nine fresh probability replays per checkpoint
  and all 544 fresh RGB140 renders were exact. The DINOv3 state SHA256 remained
  `032a9ca331800355f69e7be93db33f564fff28a3f9f43736c89b422d3e3b7445`;
  optimizer membership, spatial gradients, RGB140, SEM382, SEM378 and all 18
  frozen artifacts passed verification.

The first bound verification attempt exposed that two sequential model
constructions in one process did not reset Python/NumPy/Torch/CUDA RNG state;
it stopped before evaluating the second checkpoint and is preserved in
`verification-failure-r1.json`. A read-only wrapper reset seed `20260805`
before each construction without modifying the bound verifier. A second attempt
then reached fresh RGB reload and exposed the target environment's missing
`plyfile`; it is preserved in `verification-failure-r2.json`. Final verification
used the existing single-file `plyfile` module from a local environment; no
bound source, model, prediction or frozen artifact was changed.

The same deterministic replay behavior is retained in
`scripts/replay_verify_sem383.py`. It wraps the bound verifier and resets
Python, NumPy, CPU Torch and CUDA Torch RNGs before every model construction;
the bound `scripts/verify_sem383.py` used by `r2` remains unchanged.

Evidence: `prepare/result.json`, `preflight/result.json`,
`mask2former/result.json`, `inference-3000/result.json`,
`inference-6000/result.json`, `verification.json`, `audit.json`,
`input-diagnostic-candidate/result.json`, `visual-review.json` and
`comparison/index.html` under the output directory.

## Input-domain diagnostic

Views 311, 350 and 356 were run through SEM378, SEM382 and SEM383 using both
the prepared photo and same-camera RGB140 render. This stage read no labels,
created no optimizer and replayed historical RGB140 predictions exactly.

- SEM383 versus SEM382 hard-label photo/render disagreement changed from
  0.015510 to 0.015173 on 311, 0.018218 to 0.015682 on 350, and 0.019640 to
  0.018343 on 356.
- Mean Jensen-Shannon divergence decreased on 311 and 350, from
  0.004782/0.008924 to 0.004356/0.008511. On 356 it increased from 0.010919
  to 0.011734, while probability MAE also increased from 0.009211 to 0.009371.

The diagnostic is mixed and has no GT. It localizes input-domain sensitivity
but does not prove a domain-generalization improvement or a unique causal
explanation.

## All300 training fit

The preregistered 6,000-step endpoint reached **95.158715%** mIoU with
background and **94.094714%** foreground-only mIoU. This is training-view fit,
not held-out accuracy.

- Against SEM382, mIoU increased by only **0.036088 pp**. Background, deck,
  `stay_cable`, tower and foundation IoU increased by 0.005085, 0.038805,
  0.055207, 0.055475 and 0.025869 pp respectively.
- Against SEM378, mIoU remained **0.138690 pp lower**. The five class IoUs
  remained lower by 0.023312, 0.059099, 0.268012, 0.199287 and 0.143739 pp.
- Versus SEM382, 185/300 views improved and 115 degraded. Across 467,646
  changed pixels, 240,171 were corrected, 217,638 were harmed and 9,837 changed
  from one wrong class to another. All four foreground GT classes had more
  corrected than harmed pixels, while background had more harmed than corrected.
- Versus SEM378, only 81/300 views improved and 219 degraded. Across 972,760
  changed pixels, 430,891 were corrected, 524,121 were harmed and 17,748 were
  wrong-to-wrong. Every GT class had more harmed than corrected pixels.
- Relative to SEM382, `stay_cable` false negatives decreased by 10,845 and
  false positives decreased by 4,914. Relative to SEM378, false negatives
  increased by 23,897 and false positives increased by 53,991.
- The 2 px cable boundary precision/recall/F1 were
  0.886066/0.850873/0.868113. F1 improved 0.249390 pp over SEM382 but remained
  1.228645 pp below SEM378.

The analysis-only 3,000-step endpoint reached 94.450995% mIoU, 0.671632 pp
below SEM382. Continuing to the fixed 6,000-step endpoint added 0.707720 pp and
improved 292/300 views relative to step 3,000. All five class IoUs improved,
including `stay_cable` by 1.148851 pp and foundation by 1.197867 pp. This
supports completion of the registered budget but is not after-the-fact endpoint
selection.

## Visual review

All nine fixed views, all 100 unlabeled change maps and all 24 six-frame
trajectory sheets were inspected. Views 311/350/356 and selected high-change
unlabeled views were reviewed in detail.

- Against SEM382, five fixed-view all-image mIoUs improved and four degraded.
  The largest loss was 1.078114 pp on 064; the largest gains were 0.533834 pp
  on 160 and 0.382044 pp on 054.
- Against SEM378, eight of nine fixed views remained lower. Only view 160 was
  higher, by 0.110338 pp. The fixed views do not establish a consistent
  structural advantage.
- The complete no-GT change-map review showed mainly thin boundary and broad
  cable-envelope redistribution. Detailed views 351/355/357/358/390 also
  showed isolated tower- or foundation-colored components or removals. These
  may be artifacts, but without GT their correctness cannot be scored.
- The 144-frame mean adjacent label-change proxy was 0.029083, close to SEM382
  (0.029132) and SEM378 (0.029185). The maximum was 0.123682 at transition 98.
  There was no sequence-wide collapse, but frames 96--98 retained fragmented
  cable-envelope interiors and showed no stable temporal advantage.
- Wide `stay_cable` sectors follow the released annotation style and do not
  establish reconstruction of individual physical wires.

## Final repository validation

- `python3 -m unittest tests.test_sem383_protocol`: 5/5 tests passed.
- `python3 -m py_compile scripts/replay_verify_sem383.py
  tests/test_sem383_protocol.py`: passed.
- `jq empty` on the final visual-review, comparison and registry files: passed.
- `bash scripts/repo_health.sh`: 561 tests ran with 76 skipped and all executed
  tests passing, 238 JSON-compatible YAML files validated, and the repository
  health check passed. The optional local dataset was absent and skipped as
  expected in a clean clone.
- `git diff --check`: passed for tracked worktree changes.

## Decision

Do not promote SEM383 over RGB140+SEM378 and do not treat it as a new system
candidate. The added spatial prior is a valid, reproducible mechanism ablation:
it slightly improves all five all300 class IoUs, paired-view balance, cable
FP/FN and boundary F1 relative to SEM382. The gain is only 0.036088 pp on one
seed and on the 300 training views, while SEM383 remains below SEM378 overall,
in every class, in 219/300 paired views, corrected-versus-harmed pixels and
boundary F1. Fixed, no-GT and trajectory review does not provide a stable
compensating advantage.

Retain SEM383 and both preserved verification failures as evidence. Keep
RGB140+SEM378 as the system candidate and rollback; RGB130+SEM371 remains the
project default. Stop this line. No endpoint selection, architecture/LR scan,
extra steps, DINO unfreezing, resolution change, fusion or post-processing is
authorized. Further DINO work requires a new frozen protocol and preferably
independent held-out GT or multi-seed evidence.
