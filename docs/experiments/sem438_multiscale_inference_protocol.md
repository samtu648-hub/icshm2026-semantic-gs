# SEM438 frozen multiscale inference protocol

## Question

Test whether the unresolved deck disappearance in released views 355--357 is caused by the frozen SEM437 model's effective inference scale rather than absence of recoverable semantic evidence. This is an inference-only development audit, not training and not a blind-test claim.

## Frozen inputs

Use only the sealed SEM437-12k checkpoint and its frozen RGB170 exports. Do not update model, optimizer, RGB reconstruction, geometry, opacity, color, camera, labels, or checkpoint files. Native 768/512 sliding inference must reproduce the existing SEM437 labels exactly before any candidate is interpreted.

## Preregistered arms

Report all arms; do not select scales or weights from 344--358.

1. `native`: existing 768 tile / 512 stride inference at 1320x989.
2. `global_1p5`: bilinear-resize RGB to 1980x1484, run the same sliding model, bilinear-resize probabilities back and renormalize.
3. `global_2p0`: bilinear-resize RGB to 2640x1978, then the same probability return path.
4. `strip_2p0`: from native argmax, take pixels predicted as deck or `stay_cable`; use their 85th-percentile y as an anchor, falling back to y=494 only if support is empty. Crop a clamped full-width 1320x384 strip centered at the anchor, resize to 2640x768, infer, resize its probabilities back, and use native probability outside the strip.
5. `fixed_ensemble`: probability weights native/global-1.5/strip-2.0 = 0.50/0.25/0.25. Global-2.0 is diagnostic and is not added after seeing results.

## Expansion and retention

The no-GT 344--358 probe authorizes the full audit only if native labels exactly match SEM437, probabilities are finite simplexes, checkpoint/RGB hashes remain fixed, every candidate changes at most 5% of a probe view, and at least one fixed non-native arm adds at least 256 deck pixels in at least two of 355--357. This gate says only that recoverable signal exists.

If expanded, run every arm on all 300 labeled training views, all 100 released unlabeled views, and all 144 trajectory frames. Report full training-fit confusion, per-class IoU, `stay_cable` precision/recall/FP/FN, 2px boundary F1, per-view results, no-GT changed pixels, 344--358 class areas, runtime and memory. The fixed ensemble is retained only if five-class and foreground mIoU each lose no more than 0.05 pp, deck IoU strictly improves, `stay_cable`, tower, foundation and boundary each lose no more than 0.10 pp. These are all300 training-fit gates, not spatial-generalization proof.

Stop and preserve evidence on OOM under the registered resolutions, nonfinite probability, native replay mismatch, checkpoint/RGB hash change, missing input, or severe class collapse. Do not silently lower resolution, change weights, hand-select a bridge crop, or tune from unlabeled views in this ID.
