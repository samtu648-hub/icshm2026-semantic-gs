# SEM385 Fixed Full-Frame Context-Mix Protocol

## Purpose

Test whether missing full-frame context contributes to SEM384's development-holdout errors. This is one matched semantic-training intervention, not a new split, architecture, inference policy, or RGB reconstruction.

## Parent And Roles

- Parent: `SEM-384-RGB150-DEV270`.
- RGB: frozen RGB150 renders from the 370-view RGB training role.
- Semantic fitting: the same 270 labels from `dev30_local_composition_v1`.
- Development holdout: the same 30 labels and RGB inputs, inaccessible until the 6000-step checkpoint is sealed.
- Seed: `20260805`; endpoint: 6000 successful updates.
- Initialization, model, loss, optimizer, learning-rate schedule, AMP retry, and checkpoint roles remain identical to SEM384.

## Single Variable

Start from SEM384's frozen 6000-step view, native-crop, and horizontal-flip plan with SHA256 `b17b813ae84d6523f4fc6b322643348e44663e47788cf02c2e9ffa521f0b7dc0`.

- Odd successful updates retain the corresponding SEM384 `768x768` native crop.
- Even successful updates retain the same view and flip but replace the crop with the complete `1320x989` frame resized to `1025x768`.
- The context resize fixes the short edge to 768, preserves aspect ratio, and remains below `max_size=2560`.
- RGB uses Pillow bilinear interpolation; class IDs use Pillow nearest-neighbor interpolation.
- The parent sampler is still called at every update, so view, crop-coordinate, flip, and Python RNG consumption remain aligned with SEM384.
- No letterbox, label padding, GT-guided selection, image rejection, or model-output selection is allowed.

This yields exactly 3000 native crops and 3000 full-frame context inputs. It changes context and object scale together; the result must not be described as a pure context-only causal estimate.

## Engineering Gate

An independent fresh-model eight-update preflight must contain four native and four context inputs, use dynamic `Instances.image_size`, complete finite forward/backward updates, and remain within the local 12 GB GPU. The preflight model is discarded. OOM or numerical failure is preserved and stops the experiment; input size, activation checkpointing, model capacity, endpoint, and context fraction cannot be changed as a rescue.

## Evaluation

After sealing the fresh 6000-step checkpoint, run the unchanged SEM384 native 768/512 sliding inference on all 400 RGB150 renders. Report train and holdout five-class/foreground mIoU, all class IoUs, `stay_cable` FP/FN and 2-pixel boundary metrics, the three fixed holdout groups, corrected/harmed pixels versus SEM384, and training time/memory. This single split and seed are development evidence only.

## Commands

```bash
python3 scripts/run_sem385_context_mix.py --artifact-root <artifact-root> --stage prepare
python3 scripts/run_sem385_context_mix.py --artifact-root <artifact-root> --stage preflight
python3 scripts/run_sem385_context_mix.py --artifact-root <artifact-root> --stage train
python3 scripts/run_sem385_context_mix.py --artifact-root <artifact-root> --stage inference
```

No quality gate may stop a numerically valid run or alter the fixed endpoint. A later RGB detail-preservation arm remains unauthorized until this result is complete and reviewed.

## Execution Outcome

The 2026-09-17 fresh-model preflight stopped at update 2, the first `1025x768` context input, with CUDA OOM after update 1's native crop completed. No checkpoint was retained, no 6000-step training or inference started, and holdout RGB/labels remained unopened. The failure is preserved at `outputs/quality/SEM-385/r1/preflight/result.json`. Per the frozen failure policy, SEM385 is closed without changing its input size, memory policy, model, mixture, or endpoint.
