# SEM390/SEM391 Corrected Second-Seed Confirmation Protocol

## Purpose

Repeat the closed SEM388/SEM389 attempt under new IDs with the initialization gate corrected for a seed-dependent fresh classifier head. SEM388 and SEM389 remain immutable failed/unrun records. This pair changes no RGB reconstruction, split, model, loss, optimizer, schedule, inference, context ratio, or image size and provides development evidence rather than a blind-test claim.

## Fixed Pair

- Control: `SEM-390-RGB150-DEV270-SEED20260919-MEMORY`.
- Candidate: `SEM-391-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY`.
- Seed: `20260919` before every fresh model build for Python, NumPy, CPU Torch, and all CUDA devices.
- Data: frozen RGB150 and `dev30_local_composition_v1`; 270 semantic fitting views and 30 holdout views.
- Training: fresh official ADE Swin-L state, unchanged SEM378/SEM384 loss/optimizer/poly/AMP behavior, four-stage Swin activation recomputation, and exactly 6000 successful updates.
- Inference: unchanged native 768 crop / 512 stride sliding policy at checkpoint 6000 only.

## Initialization Contract

Each preflight independently reconstructs the seed-`20260805` SEM384 fresh ADE state and first requires its complete state digest to match the recorded SEM384 initial state. It then resets every RNG to `20260919` before constructing the actual arm model.

- The reconstructed old-seed and current-seed pretrained-load reports must be identical.
- Exactly 650 state tensors must exist. The 648 non-head tensors must be byte-identical across seeds.
- Only `sem_seg_head.predictor.class_embed.weight` and `.bias` may differ across seeds, and both are required to differ.
- The current head must have shapes `[6, 256]` and `[6]`, be finite, and have SHA256 recorded. Six rows represent five semantic classes plus no-object.
- SEM391 must match SEM390 across all 650 tensors and the complete pretrained-load report at the same seed.
- Formal training reconstructs a fresh model and repeats exact all-state/load-report equality against the sealed preflight evidence for SEM390 and against SEM390's sealed training initialization for SEM391.
- A non-head difference, load-report difference, missing cross-seed head difference, malformed head, or nonfinite head stops and preserves the arm before update 1. Cross-seed complete-state equality is forbidden as a hard gate.

## Frozen Sampling

SEM390 freezes a new 6000-step native plan from seed `20260919`; SEM391 is bound entry-for-entry after the SEM390 plan hash is known.

- SEM390 uses native `768x768` crops for all 6000 updates.
- SEM391 retains every SEM390 view, original crop-coordinate draw, flip, and sampler RNG consumption. Odd updates use the native crop; even updates use the complete `1320x989` frame resized to `1025x768`.
- RGB resize is Pillow bilinear, label resize is Pillow nearest, and horizontal flip follows resize.
- Letterbox, padding, filtering, GT-guided selection, image rejection, rescue resizing, and endpoint selection are forbidden.
- The intervention changes context and object scale together and is not described as a pure context-only effect.

## Access And Execution Gates

1. Prepare and hash both independent new-ID plans without reading holdout RGB or labels. Save a paired audit of all 6000 view/crop/flip records, odd/even modes, dimensions, export hashes, and zero holdout access.
2. Run independent fresh-model eight-update preflights. SEM390 must execute eight native inputs; SEM391 must execute four native and four context inputs. Both must pass initialization, recipe, trace, finite-loss, gradient, RGB-hash, and 12 GB gates. Discard both models.
3. If either preflight fails, preserve the evidence and stop. Formal training is executable only after both preflights pass.
4. Train SEM390 and then SEM391 sequentially to 6000 successful updates. Preserve checkpoints 3000 and 6000, but evaluate only 6000. Do not rescue a numerical or resource failure.
5. Holdout RGB and labels remain closed until both checkpoint-6000 files and all training artifacts are sealed and hash-verified. Then run both inference jobs.

## Evaluation

For training and holdout roles record five-class and foreground mIoU, all class IoUs, `stay_cable` precision/recall/FP/FN, and 2px boundary precision/recall/F1. Also record regular-local, oblique-local, and distant-local results; paired per-image changes; correction/damage/class-swap counts; paired bootstrap intervals; runtime; peak allocated/reserved GPU memory; and checkpoint sizes.

Generate three local contact sheets covering all 30 holdout views with RGB, ground truth, control prediction, candidate prediction, and change map. Review all 30 for large-area collapse, remote isolated blocks, cable continuity, tower continuity, cable shortening, and thin-structure damage. Record uncertain and adverse views explicitly.

## Frozen Confirmation Gate

- On each seed, candidate-minus-control holdout five-class mIoU is nonnegative and foreground mIoU is strictly positive.
- The mean five-class mIoU difference over seeds `20260805` and `20260919` is strictly positive.
- On each seed, 2px `stay_cable` boundary F1 does not decline.
- On each seed, no overall holdout class IoU declines by more than `0.20 pp`.
- On each seed, none of the three subgroup five-class mIoUs declines by more than `0.20 pp`.
- Review of all 30 views finds no new large-area collapse, remote isolated semantic blocks, or systematic cable/tower discontinuity.

Only a pass on every quantitative and visual condition authorizes registration of a new-ID frozen RGB140 all-data migration. Failure retains the negative or ambiguous result and stops migration. No threshold, seed, ratio, size, endpoint, metric, or visual criterion may change after holdout opens.

## Current State

Both preparations and preflights passed, but SEM390 formal training failed its final frozen-trace audit after 6000 updates. The actual training trace matched the seed-`20260805` SEM384 plan at all 6000 entries rather than the registered seed-`20260919` SEM390 plan. The shared runner had correctly seeded model initialization but passed an inherited SEM370 recipe seed to the training function's private sampler. The failed output and non-authoritative checkpoints are preserved; SEM391 formal training was not started; holdout remained closed. Both IDs are closed. Any seed-thread correction requires another new-ID preregistered pair.
