# RGB147-149 Semantic-GT Structure Supervision Protocol

## Question

Test whether released semantic GT can localize reconstruction supervision while the real photo remains the only color and edge target. This is a three-arm RGB140 Standard/AbsGS comparison, not semantic rendering and not label-contour transfer.

## Fixed Identity

- A: `RGB-147-STANDARD-GT-SUPERVISION-CTRL`, original RGB140 full-frame RGB loss.
- B: `RGB-148-STANDARD-GT-REGION-RGB`, A plus GT-structure-region RGB L1.
- C: `RGB-149-STANDARD-GT-REGION-GRADIENT`, A plus GT-structure-region multiscale photo-gradient consistency. C does not include B.
- Parent: `RGB-140-STANDARD-ABSGRAD-ALL400`.
- Manifest: `data/manifests/contest_dataset_v1.json`.
- Split: `released_all400_rgb_all300_sem_v1`.
- All arms: official fixed 400 cameras, 1320x989, seed 0, 30,000 iterations, original 72,292-point RGB140 initialization, and original RGB140 AbsGS densification.

## Region

For each of the existing 300 aligned GT labels, the common region is the union of deck, `stay_cable`, tower, and foundation (`GT != background`), dilated by 4 native pixels. Losses divide by the effective weighted area and RGB channel count. The 100 unlabeled views use only the original RGB loss; no pseudo-GT is generated.

The region only selects where photo supervision receives extra weight. In particular, a broad `stay_cable` fan polygon does not provide a target color, a filled-cable target, or a polygon-edge target. All targets come from the corresponding real photo.

## Losses

The inherited loss is:

`L_A = 0.8 * mean(|R-P|) + 0.2 * (1-SSIM(R,P))`.

For B:

`L_B = L_A + lambda_rgb * sum(M * |R-P|) / (3 * sum(M))`.

For C, RGB forward differences are computed on the complete render and photo before region weighting. Scale 1 uses the native images. Scale 4 first area-downsamples each complete image and the common mask. Horizontal and vertical terms are each area-normalized, then averaged; scales 1 and 4 are equally weighted:

`L_C = L_A + lambda_grad * mean_s(mean_xy(sum(M_s * |D(R_s)-D(P_s)|)/(3*sum(M_s))))`.

The coefficient rule is fixed before full training: on 12 uniformly spaced labeled views and frozen RGB140 renders, set each coefficient so the median image-space auxiliary gradient norm is 20% of the inherited RGB-loss gradient norm. This is a one-time scale calibration, not a quality sweep. Numeric coefficients and all probe rows must be recorded before A/B/C training.

The frozen `calibration-r2` coefficients are `lambda_rgb=0.26004419260021433` and `lambda_grad=0.3103426341938125`. The preserved `calibration-r1` artifact used the median of per-view ratios rather than the required ratio of medians and is superseded; no training used it.

## Densification And Capacity

The complete selected training loss is backpropagated through the native RGB140 renderer. The original absolute-gradient split statistic, signed clone statistic, pruning, and opacity reset remain unchanged. Therefore B/C can change screen-space gradient accumulation, densification, pruning, topology, final Gaussian count, runtime, and render cost. This experiment does not claim fixed topology or equal capacity.

## Preflight

- Verify every label hash, PINHOLE grid, 1320x989 shape, and class ID.
- Generate photo, frozen RGB140 render, GT region, photo gradient, render gradient, region RGB error, and multiscale gradient-error panels for the fixed nine views.
- Inspect bridge deck, towers, main cable, cable fan, water, shore, 159/160 near and far regions.
- Prove the implementation computes gradients before masking using a constant-image synthetic counterexample.
- Run two native A-off and two hooked A-off repeats from the same initialization. The pinned CUDA rasterizer is not bit-exact across independent repeats, so require the control loss path to return the identical loss object, exact A/B/C camera sequences, and native-to-hooked checkpoint tensor deltas no larger than twice the observed native/native or hooked/hooked repeatability envelope plus a `1e-8` absolute numerical floor. Failed strict checks remain preserved as `preflight-r1` and `preflight-r2`; passed r3 predates the resume-evidence-only code change, so full training is bound to a fresh r4.
- Require finite, nonzero B/C gradients on expected appearance and geometry parameters.
- Preflight is an engineering/content check, not an early quality gate.

## Evaluation And Stop

After the matched 30k runs, evaluate all 400 training images with PSNR/SSIM/LPIPS and per-image deltas; fixed nine views/ROIs; class-region RGB MAE; common region RGB and gradient errors; and the same 144-frame no-GT official trajectory. Report A-to-B, A-to-C, and B-to-C, final Gaussian count, training time, peak memory, model size, and rendering cost. Build photo/A/B/C pages and one synchronized A/B/C trajectory video.

All 400 scores are training fit. The 144-frame trajectory is qualitative only. Auxiliary loss decrease or sharper edges alone is not reconstruction success. Stop after these three arms; do not add RGB143/RGB146, DINOv3, Scaffold migration, semantic retraining, post-processing, coefficient scans, or extra combinations.
