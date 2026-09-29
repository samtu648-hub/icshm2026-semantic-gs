# SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300

Status: completed; positive optimizer ablation versus SEM381; not promoted over SEM378.

- Parent/direct control: `SEM-381-DINOV3-VITB-SFP-ALL300`
- Performance reference: `SEM-378-RGB140-ABSGRAD-ALL300`
- Protocol: [SEM382 SFP LR 1e-4](../../../docs/experiments/sem382_dinov3_sfp_lr1e4_protocol.md)
- Config: [config.yaml](config.yaml)
- Output: `outputs/quality/SEM-382/r1`
- Scope: all300 training-view fit; unlabeled100 and trajectory144 qualitative only.
- Git: `49cdcdb1c9915a9544d568754af95ea20a85ec9c`
- Seed: `20260805`
- Split: `released_all400_rgb_all300_sem_v1`

## Registered treatment

Only the initial learning rate of every trainable SFP parameter, including all
SFP LayerNorm parameters, changes from `1e-5` to `1e-4`. The Mask2Former
decoder remains at `1e-4`, DINOv3 remains fully frozen, and all initialization,
data, sampling, augmentation, loss, optimizer, schedule, clipping, crop and
inference settings remain matched to SEM381.

Checkpoint 6000 is the preregistered primary result. Checkpoint 3000 is retained
only to analyze training progress and will not be selected after observing
quality.

## Execution

- Preparation bound the config, split, 544 RGB140 images, 300 labels, local
  DINOv3 files, SEM381/SEM378 endpoints, the full SEM381 sampling plan and 18
  frozen artifacts. The initial trainable-state SHA256 was
  `cc677af8686be7df215b0d8b1a3f82efe6ff114d23d9c6b5fd543d538b86b995`.
- The native-768 preflight completed eight successful updates. It verified that
  all 3,836,352 SFP adapter parameters and all 6,144 SFP LayerNorm parameters
  used initial LR `1e-4`, the 20,250,310 decoder parameters remained at
  `1e-4`, and all 85,660,416 DINOv3 parameters were frozen and absent from the
  optimizer.
- The formal run completed 6,000 successful updates in 930.752 s (15.513 min),
  with two AMP skips. Peak memory was 2,439.679 MiB allocated and 2,604 MiB
  reserved. SFP update L2 was 11.181830 (17.202486% relative); decoder update
  L2 was 16.293851 (5.505575% relative).
- The 3,000-step checkpoint is 439,230,127 bytes with SHA256
  `f6a34e9cba773607988041eb340aaaac91f0216c66f0d76434c08c1f4b9562c9`.
  The 6,000-step checkpoint has the same size and SHA256
  `ba9c802d6aa5a5bfa0db5af53746f3c5ec053ce43e232f78ce258f5c5ebae9c3`.
- Each checkpoint produced 300 labeled, 100 unlabeled and 144 trajectory
  predictions. Measured forward time was 200.922 s at step 3,000 and 195.183 s
  at step 6,000; median per-image times were 0.368290 s and 0.355020 s.
- Both checkpoints were finite. Nine fresh probability replays per checkpoint
  and all 544 fresh RGB140 renders were exact. DINOv3 state SHA256 remained
  `032a9ca331800355f69e7be93db33f564fff28a3f9f43736c89b422d3e3b7445`;
  RGB140, SEM381, SEM378 and all 18 frozen artifacts remained unchanged.

Evidence: `prepare/result.json`, `preflight/result.json`,
`mask2former/result.json`, `inference-3000/result.json`,
`inference-6000/result.json`, `verification.json`, `audit.json`,
`visual-review.json`, `driver.json` and `comparison/index.html` under the output
directory. The driver was run after the manually completed stages to record the
stage inventory; it did not rerun any computation.

## All300 training fit

The preregistered 6,000-step endpoint reached **95.122627%** mIoU with
background and **94.050875%** foreground-only mIoU. This is training-view fit,
not held-out accuracy.

- Against SEM381, mIoU increased by **0.326073 pp**. Background, deck,
  `stay_cable`, tower and foundation IoU increased by 0.028312, 0.136406,
  0.256667, 0.480002 and 0.728976 pp respectively.
- Against SEM378, mIoU remained **0.174778 pp lower**. The same five class IoUs
  remained lower by 0.028398, 0.097904, 0.323219, 0.254762 and 0.169608 pp.
- Versus SEM381, 274/300 views improved and 26 degraded. Across 776,296 changed
  pixels, 443,694 were corrected, 317,418 were harmed and 15,184 changed from
  one wrong class to another. Every GT class had more corrected than harmed
  pixels.
- Versus SEM378, only 65/300 views improved and 235 degraded. Across 985,206
  changed pixels, 425,612 were corrected, 541,375 were harmed and 18,219 were
  wrong-to-wrong. Every GT class had more harmed than corrected pixels.
- Relative to SEM381, `stay_cable` false negatives decreased by 54,314 and
  false positives decreased by 18,804. Relative to SEM378, false negatives
  increased by 34,742 and false positives increased by 58,905.
- The 2 px cable boundary F1 was 0.865619, up 1.965882 pp from SEM381 but still
  1.478035 pp below SEM378. Precision/recall were 0.881314/0.850473, between
  SEM381 and SEM378.

The analysis-only 3,000-step endpoint reached 94.442818% mIoU, 0.353736 pp
below SEM381. It improved on SEM381 in 87 views and degraded in 213. Continuing
to the fixed 6,000-step endpoint added 0.679808 pp, improved all five class IoUs
and improved 287/300 views relative to step 3,000. This supports the registered
endpoint choice but is not an after-the-fact checkpoint selection.

## Visual review

All nine fixed views and all 24 six-frame trajectory sheets were inspected;
selected high-change no-GT views were also reviewed.

- SEM382-6000 improved all nine fixed-view all-image mIoUs over SEM381,
  including visible recovery of cable-envelope holes in 057 and 064, but every
  fixed-view all-image mIoU remained below SEM378.
- Views 159/160 recovered incomplete SEM381/SEM382-3000 cable and deck regions.
  Their difficult and structure ROIs nevertheless remained below SEM378 except
  for the small registered near boxes. View 146 had a local difficult-ROI gain,
  but its full image and larger structure ROI remained below SEM378.
- On no-GT views 350/351, SEM382-6000 removed tower-colored fragments below the
  bridge. On 355--358 it also removed large parts of the broad cable sector.
  Without GT neither change is an accuracy result.
- The 144-frame mean adjacent label-change proxy was 0.029132, close to SEM378
  (0.029185) and SEM381 (0.028895). The maximum was 0.123508 at transition 98,
  slightly above both controls. Frames 96--98 retained fragmented cable-envelope
  interiors; there was no collapse or clear temporal advantage.
- Wide `stay_cable` sectors follow the released annotation style and do not
  establish reconstruction of individual physical wires.

## Runtime note

The target Python environment still lacked `plyfile`. Verification used the
same narrow runtime workaround as SEM381 by injecting only the existing
single-file module from another local environment. Bound source, checkpoints,
predictions and frozen artifacts were not modified.

## Final repository validation

The three SEM382 protocol tests passed in the target Python environment. Final
repository health passed 237 experiment configurations and 556 tests, with 76
optional-runtime skips; the absent optional local dataset was skipped as
expected for a clean clone. JSON parsing and `git diff --check` also passed.

## Decision

Do not promote SEM382 over RGB140+SEM378. Increasing the complete SFP learning
rate to `1e-4` is a successful optimizer ablation relative to SEM381: it
improves all five classes, most paired views, cable FP/FN and boundary F1. It
does not fully recover the matched Swin-L reference: all five class IoUs,
235/300 paired views, corrected-versus-harmed pixels and cable boundary F1
remain worse than SEM378.

Retain SEM382 as complete reproducible evidence and keep RGB140+SEM378 as the
system candidate and rollback; RGB130+SEM371 remains the project default. Stop
this training line. No endpoint selection, LR grid, extra steps, DINO unfreezing,
crop change, fusion or post-processing is authorized. Any further DINO work
requires a new frozen protocol and preferably independent held-out GT or
multi-seed evidence.
