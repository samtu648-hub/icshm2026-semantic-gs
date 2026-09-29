# SEM-387-RGB150-FEATURE-SPLATTING-DINOV3

Status: completed on the fixed Dev30 split and not promoted. The exact matched
DINOv3 distillation arm did not improve the label-only Feature Splatting arm,
and both arms were substantially below SEM384 and SEM386.

- Parent RGB: `RGB-150-STANDARD-ABSGRAD-DEV30`.
- Development comparator: `SEM-384-RGB150-DEV270`.
- Split: `dev30_local_composition_v1`.
- Seed: `20260805`.
- Protocol: [SEM387 Feature Splatting + DINOv3](../../../docs/experiments/sem387_feature_splatting_dinov3_protocol.md).
- Output: `outputs/quality/SEM-387/r1`.

The fixed comparison freshly trained a label-only 32D per-Gaussian field and
an exactly matched field that additionally distilled frozen DINOv3 ViT-B/16
final patch tokens from the 100 unlabeled RGB-training views. Both arms kept
RGB150 geometry, opacity, SH, exposure, cameras, densification and pruning
immutable. This is a project adaptation, not a strict Feature Splatting
reproduction or a pure-pretraining causal estimate.

## Source, access and engineering evidence

- The recovered DINOv3 directory matched all four registered model-file hashes
  and the SEM381 `download_manifest.json` SHA256
  `cbd11fe8edd8083efad8979ff4151baabd341631ed222fa5bd1ea64edf3bc2e3`.
  The RGB150 Standard renderer tree also matched all 828 files in its sealed
  source manifest.
- `prepare` completed with 370 training RGB hashes and 270 training-label
  hashes. It read zero holdout RGB or label hashes. Binding SHA256 is
  `f5c392ea2ecc1826efbd9e0d04a476e06e101e9dc72cce12a42c8aeec0505e14`.
- The independent eight-update preflight completed for both arms. Initial state
  and labeled trace matched exactly; the control teacher head had no gradient,
  the candidate teacher head had a finite nonzero gradient, and the frozen
  DINOv3 state was unchanged.
- Signed zero-field output was exactly zero. The 32-channel native-pass
  reassembly maximum absolute error was `2.980232238769531e-07`.
- Both preflight checkpoints passed exact state and actual render replay. RGB
  parameters and the three fixed RGB renders were exact before and after.
- Preflight peak allocated/reserved memory was `7019.455/8562 MiB` on an
  `11738.625 MiB` RTX 5070, so the fixed 12GB resource gate passed.
- Final verification passed all 11 targeted tests in the fixed Torch runtime,
  245 config validations, complete `git diff --check`, and the repository
  health check of 587 tests with 85 optional-runtime skips. The earlier
  `py_compile` check also remains valid because the result update changed only
  records and one registry-state assertion.

## Training evidence

- Both arms completed exactly 6000 successful updates from state SHA256
  `8730706ca66a0834536894c16776c53e25ceea0ad338bef9f0a2511d5a922efe`
  and labeled-plan SHA256
  `7e3684bd2f5b378aa6d24c7105691cb359d161c42adc106fd2f8128ace4ab7ae`.
- Label-only training took `2426.677 s`; DINOv3-distilled training took
  `5103.625 s`. Candidate-only DINO supervision therefore added `2676.948 s`
  and made the arm about `2.10x` as slow.
- Full-training peak allocated/reserved memory was `7745.372/10156 MiB`; the
  12GB resource gate still passed.
- Both arms passed final checkpoint state/render replay, exact RGB parameter
  invariance and three-view RGB pixel invariance. Candidate DINOv3 state SHA256
  remained exactly
  `032a9ca331800355f69e7be93db33f564fff28a3f9f43736c89b422d3e3b7445`.
- Step-3000 checkpoints were retained as descriptive provenance and were not
  quality-selected. The preregistered step-6000 endpoint alone determined the
  comparison.
- No holdout RGB or label byte was accessed during preparation, preflight or
  either full fit. Inference opened Dev30 only after both step-6000 checkpoint
  hashes were recorded, and those hashes were unchanged after inference.

## Fixed Dev30 result

At the preregistered step-6000 endpoint:

- Label-only: five-class mIoU `87.520321%`; foreground mIoU `85.025809%`.
- DINOv3-distilled: five-class mIoU `87.501212%`; foreground mIoU
  `85.002832%`.
- Candidate minus control: `-0.019109 pp` five-class and `-0.022977 pp`
  foreground mIoU.
- Candidate class-IoU deltas were background `-0.003638 pp`, deck
  `-0.012092 pp`, `stay_cable` `-0.082328 pp`, tower `-0.022832 pp`, and
  foundation `+0.025344 pp`.
- Candidate `stay_cable` FP decreased by 355 pixels, but FN increased by 2218.
  Its 2px holdout boundary F1 decreased by `0.075496 pp`.
- The candidate improved 10 and degraded 20 of the 30 holdout images by
  per-image five-class mIoU. Its mean per-image delta was `-0.003294 pp`; the
  best/worst images were `187.png` (`+0.687830 pp`) and `185.png`
  (`-0.925080 pp`).
- The regular/oblique/distant group five-class deltas were
  `-0.041400/-0.060522/+0.111921 pp`; this mixed subgroup behavior does not
  override the pooled endpoint.
- Relative to SEM384, the label-only and candidate five-class mIoU values were
  lower by `6.178285 pp` and `6.197394 pp`. The candidate was `6.240584 pp`
  below SEM386, whose separate human visual review remains pending.

Across the 400 released views, the two arms differed on every image but changed
only `0.203694%` of pixels on average; the maximum was `3.914116%` on unlabeled
view `388.png`. A limited read-only spot check of fixed view `159.png` and the
holdout extrema `185.png`/`187.png` found sparse arm-to-arm changes rather than
a stable structural repair. This check is not a substitute for a broad human
visual review, but no visual claim is needed to reject promotion because the
fixed pooled endpoint and both established comparators are already negative.

The label-only/candidate checkpoint sizes were `285834165/287421877` bytes.
Median per-view inference time was `0.177896/0.177525 s`; complete inference
took `234.795 s` with peak allocated/reserved memory of
`3364.498/4474 MiB`. Mean adjacent trajectory label change was
`0.0318364/0.0318108`; without trajectory GT this is descriptive only.

## Artifact hashes

- `prepare/result.json`: `31eae01113c4f0297c078a85c3276b45ca9be762fa5b3588473a36abb797a704`
- `preflight/result.json`: `9ca8476323fe3722f06f72792d02160eb076065f3c892a49df0b786cc2fd24bc`
- `training/result.json`: `b65c4e88a25914ff5b5ebf479f2eebebd7a6cb42a5ac44dc00503b4fa64609f0`
- `inference/result.json`: `9607fcd284b8232da573f62e0c46e524f813f85c5e2632abaadfbdca47dee2d5`
- Label-only step 3000/6000:
  `fb51d83ba8ca519f0cb4c681ee4020ac155cf4dff654a03ef863cf20f31ed8d6` /
  `1830c5c0af4aa0b2f858ee37b7e7a8d7b7afe3b1407ea1e9b54ca79787088157`.
- DINOv3-distilled step 3000/6000:
  `e9ca173f16b4f20e08f1cce5a087d02649cab8cd0dfc5da3f6d08edef170b7d7` /
  `c3473c036e822f19bbb3c1b2f8d36dc80880398235a58b34335d3d547d5f4bad`.

## Decision and rollback

SEM387 is complete and not promoted. Frozen DINOv3 distillation did not improve
the exactly matched Feature Splatting control, and the shared Feature Splatting
representation was much weaker than SEM384/SEM386 on this split, especially on
`stay_cable`. SEM384 remains the Dev30 baseline; SEM386 remains a separate small
metric improvement pending its existing human review. All SEM387 artifacts are
preserved, but no second seed, all-data fit, layer/loss search, geometry update,
pseudo-labeling, fusion or post-processing follows automatically. Rollback is
to ignore both SEM387 checkpoints; no parent, default or system candidate was
modified.
