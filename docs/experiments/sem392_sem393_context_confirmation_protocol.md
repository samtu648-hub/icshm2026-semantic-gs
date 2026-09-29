# SEM392/SEM393 Seed-Thread-Corrected Confirmation Protocol

## Purpose And History

This new-ID pair executes the originally requested second-seed RGB150 confirmation after two preserved engineering failures. SEM388 failed an invalid cross-seed complete-state gate. SEM390 later passed the corrected initialization preflight but its formal runner inherited the old SEM370 recipe seed for the private sampler and failed the final trace audit. SEM388 through SEM391 remain immutable and provide no quality evidence. SEM392/SEM393 correct only seed validation and seed propagation; no scientific or runtime variable changes.

## Fixed Pair

- Control: `SEM-392-RGB150-DEV270-SEED20260919-MEMORY`.
- Candidate: `SEM-393-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY`.
- Seed: `20260919` for Python, NumPy, CPU Torch, all CUDA devices, the model, the training recipe, and the private sampler.
- Data: frozen RGB150 and `dev30_local_composition_v1`; 270 fitting views and 30 holdout views.
- Training: fresh official ADE Swin-L state, unchanged SEM378/SEM384 loss, optimizer, poly schedule, AMP policy, four-stage Swin recomputation, and exactly 6000 successful updates.
- Inference: unchanged native 768 crop / 512 stride sliding policy at checkpoint 6000 only.

## Initialization And Seed Gates

- Rebuild seed `20260805` and require its complete state digest to match SEM384.
- Rebuild seed `20260919` after resetting every RNG. Exactly 650 tensors must exist; the 648 non-head tensors and pretrained-load report must match across seeds.
- Only classifier weight and bias may and must differ across seeds. Their shapes must be `[6,256]` and `[6]`, values finite, and hashes recorded.
- At seed `20260919`, SEM393 must match SEM392 across all 650 tensors, both head tensors, and the full pretrained-load report.
- Formal training repeats exact initialization validation against sealed evidence.
- Before any formal GPU/model construction, regenerate the native 6000-step trace from the registered config seed and require exact equality with every frozen plan's step/view/top/left/flip fields. The training recipe seed must equal the registered config seed.

## Sampling

SEM392 freezes a new seed-`20260919` native plan. SEM393 is derived after its hash is known.

- SEM392: 6000 native `768x768` updates.
- SEM393: identical view, original crop draw, flip, and sampler RNG consumption at every update; odd updates native, even updates complete `1320x989` frames resized to `1025x768`.
- RGB is resized bilinearly, labels nearest-neighbor, and flip follows resize.
- Letterbox, padding, filtering, GT-guided selection, image rejection, rescue resizing, and endpoint selection are forbidden.
- Context and object scale change together; this is not a pure-context claim.

## Execution And Access

1. Prepare both independent new-ID outputs and audit all 6000 paired records, dimensions, RGB hashes, and zero holdout access.
2. Run two independent fresh eight-update preflights: eight native for SEM392 and four native/four context for SEM393. Require corrected initialization, exact trace, finite losses, all-stage finite gradients, and the 12 GB gate.
3. Stop and preserve both IDs if either preflight fails.
4. Only after both pass, train SEM392 then SEM393 sequentially. Preserve checkpoints 3000/6000; evaluate only 6000. No rescue for numerical/resource/contract failure.
5. Holdout remains closed until both checkpoint-6000 files and training artifacts are sealed and hash-verified.

## Evaluation And Frozen Confirmation Gate

Record complete train/holdout and three-group five-class/foreground metrics, all class IoUs, `stay_cable` precision/recall/FP/FN and 2px boundary metrics, per-image changes, correction/damage/class swaps, paired bootstrap intervals, runtime, memory, and checkpoint sizes. Generate three ten-view contact sheets covering all 30 holdout views and review large collapse, remote isolated blocks, cable/tower continuity, shortening, and thin-structure damage.

- Each seed: five-class delta nonnegative; foreground delta strictly positive; 2px `stay_cable` boundary F1 nonnegative.
- Two-seed mean five-class delta strictly positive.
- Each seed: no overall class IoU or group five-class mIoU decline below `-0.20 pp`.
- All 30 views: no new large-area collapse, remote isolated semantic block, or systematic cable/tower discontinuity.

Only all-pass quantitative and visual evidence authorizes a separate new-ID frozen RGB140 all-data migration. Any failure retains a negative/ambiguous result and stops. Thresholds and criteria are immutable after holdout opens.

## Current State

The pair is complete and closed. Preparations, preflights, formal 6000-update fits, sealed-endpoint inference, independent comparison, all-30-view visual review, and the final gate all completed. Both formal initializations passed exact 650-tensor/load-report checks; both traces matched the registered seed-`20260919` plans; SEM393 used exactly 3000 native and 3000 context updates; holdout access during fitting was zero.

SEM393 improved train five-class mIoU by `+0.053723 pp` but reduced holdout five-class/foreground mIoU by `-0.536632 pp` / `-0.659322 pp`. Foundation, `stay_cable`, tower, and regular-local protection checks failed. The earlier seed improved five-class mIoU by `+0.043190 pp`, but the two-seed mean is `-0.246721 pp`. The visual hard gate passed after review of all 30 panels, with regular-local thin-structure damage retained as an adverse observation. The final status is `confirmation_failed`; RGB140 migration is not authorized and no rescue experiment follows from this protocol.
