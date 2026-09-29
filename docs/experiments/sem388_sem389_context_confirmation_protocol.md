# SEM388/SEM389 Second-Seed Context Confirmation Protocol

## Purpose

Confirm or reject SEM386's small Dev30 gain with one new matched seed. This protocol changes no RGB reconstruction, split, model, loss, optimizer, schedule, inference, context ratio, or image size. It is development evidence on the same fixed holdout, not a blind-test claim.

## Fixed Pair

- Control: `SEM-388-RGB150-DEV270-SEED20260919-MEMORY`.
- Candidate: `SEM-389-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY`.
- Seed: `20260919` for Python, NumPy, CPU Torch, and all CUDA devices.
- Data: frozen RGB150 and `dev30_local_composition_v1`; 270 semantic fitting views and 30 holdout views.
- Initialization: fresh official ADE Swin-L state in each arm. No fitted semantic checkpoint is reused.
- Training: unchanged SEM378/SEM384 losses, author optimizer, poly schedule, AMP retry policy, and exactly 6000 successful updates.
- Memory: native `BasicLayer` activation recomputation on all four Swin stages in both arms. The runtime-only policy is recorded separately from the unchanged resolved model YAML.
- Inference: unchanged native 768 crop / 512 stride sliding policy at the sealed 6000-step endpoint only.

SEM384/SEM386 used seed `20260805`. Their memory execution was asymmetric because recomputation was introduced only to make SEM386 fit; the pinned Swin component test preserved forward output and RNG exactly and matched gradients within tolerance. The new pair removes that engineering asymmetry by applying recomputation to both arms.

## Frozen Sampling

SEM388 freezes a new 6000-step native plan from seed `20260919`. SEM389 is derived entry-for-entry from that plan.

- Every arm consumes the parent sampler on every successful update, preserving view, crop-coordinate, flip, and Python sampling-RNG consumption.
- SEM388 uses the sampled native `768x768` crop on all updates.
- SEM389 uses that same native crop on odd updates. On even updates it retains the same view and flip but uses the complete `1320x989` frame resized to `1025x768`.
- Full-frame RGB uses Pillow bilinear interpolation and labels use Pillow nearest-neighbor interpolation. Horizontal flip follows resize. Letterbox, padding, filtering, GT-guided selection, and image rejection are forbidden.
- The intervention changes context and object scale together and must not be described as a pure context-only effect.

## Access And Execution Gates

1. Prepare and hash both immutable plans without reading holdout RGB or labels.
2. Run an independent fresh-model eight-update preflight for each arm. SEM388 must execute eight native inputs; SEM389 must execute four native and four context inputs. Both must validate finite losses, all-stage finite gradient participation, frozen RGB hashes, initialization, recipe, trace, and the 12 GB resource limit. Discard both preflight models.
3. If either preflight fails, preserve the failure and stop. Do not rescue by changing any scientific or runtime setting.
4. Train both arms sequentially to exactly 6000 successful updates. A numerical/resource failure is retained; no partial checkpoint is selected.
5. Do not access any holdout RGB or label until both 6000-step checkpoints are sealed and hash-recorded. Only then run inference for both arms.

## Preregistered Confirmation Gate

All conditions below are evaluated at the 6000-step endpoint. `pp` means percentage points. Overall class changes refer to the five holdout classes aggregated over all 30 views; subgroup limits refer to each subgroup's five-class mIoU, not per-class subgroup diagnostics.

- On each seed, candidate-minus-control holdout five-class mIoU is nonnegative and foreground mIoU is strictly positive.
- The mean candidate-minus-control five-class mIoU over seeds `20260805` and `20260919` is strictly positive.
- On each seed, 2px `stay_cable` boundary F1 does not decline.
- On each seed, no overall holdout class IoU declines by more than `0.20 pp`.
- On each seed, none of regular-local, oblique-local, or distant-local five-class mIoU declines by more than `0.20 pp`.
- Repeated visual review of all 30 holdout views finds no new large-area collapse, remote isolated semantic blocks, or systematic cable/tower discontinuity.

Subgroup per-class changes remain mandatory diagnostics but are not silently substituted for the frozen gate above. No threshold, seed, ratio, image size, endpoint, metric, or visual criterion may be changed after either holdout result is opened.

Only if every condition passes may the identical 50% context recipe be registered for a frozen RGB140 all-data migration. Failure retains the negative/ambiguous result and stops that migration.

## Execution Outcome

The SEM388 preparation completed without holdout access, but its fresh-model preflight stopped before update 1. The implementation compared the complete seed-`20260919` state against the seed-`20260805` SEM384 state. Independent reconstruction showed that 648 of 650 tensors were identical and only the newly initialized five-class classifier weight and bias differed; both pretrained-load reports were identical. The failure is therefore an initialization-validation contract error rather than a model-quality, numerical, resource, or context result.

Per the preregistered stop rule, the failure is preserved, SEM389 preflight was not run, and neither 6000-step fit was started. The holdout remained unopened. A seed-aware contract must use new experiment IDs and be registered before any continuation; these IDs are closed and will not be repaired in place.
