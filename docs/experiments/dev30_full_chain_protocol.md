# Fixed Dev30 Full-Chain Protocol

## Purpose

Measure reconstruction and semantic behavior on a reusable 30-view development holdout that is excluded from the complete optimization chain. This protocol is a development evaluation, not a blind test and not a strictly guarded spatial split.

## Frozen Split

- Split ID: `dev30_local_composition_v1`.
- Source: official COLMAP camera centers, optical axes, released covisibility tracks, and LabelMe composition statistics only.
- Model outputs, rendered images, model errors, and quality scores are forbidden during selection.
- Groups: regular-local `108.png` through `117.png`; oblique-local `145.png` through `154.png`; distant-local `180.png` through `189.png`.
- Holdout-list SHA256: `d32f0091a95ba59f0e6fe0f86dd2368f51c5341289e1525c87ff911bf8b5b7ac`.
- The numeric names are not the selection rule. Each group is the deterministic geometry/composition winner defined in the split artifact.

## RGB Access

- RGB training uses 270 labeled and 100 unlabeled released views.
- The 30 holdout RGB images are excluded from photometric optimization, feature matching, re-triangulation observations, point-cloud initialization, and point colors.
- Official poses remain fixed and are available for coordinate-compatible holdout rendering.
- Initialization is re-triangulated from training observations; no fitted all-400 checkpoint is reused.

## Semantic Access

- Semantic fitting uses only the 270 training labels and their frozen RGB-150 renders.
- Holdout RGB and labels are inaccessible while fitting and while constructing the sampling plan.
- The 6000-step sampling plan is generated from the sorted 270-view training list with seed `20260805` and frozen before training.
- The model starts from the same ADE pretrained Swin-L state as SEM378. SEM378's fitted weights are not loaded.
- Holdout inference and ground-truth evaluation begin only after the 6000-step checkpoint is sealed.

## Metrics

- RGB: PSNR, SSIM, and LPIPS, reported separately for train, holdout, and each holdout composition group.
- Semantic: five-class IoU, mIoU with and without background, per-view results, `stay_cable` FP/FN, and 2-pixel `stay_cable` boundary precision/recall/F1.
- Results are also grouped by regular, oblique, and distant local blocks.
- Training metrics are fit diagnostics. Holdout metrics estimate this fixed development protocol only.

## Execution

```bash
python3 scripts/run_rgb150_dev30.py --artifact-root <artifact-root> --stage preflight
python3 scripts/run_rgb150_dev30.py --artifact-root <artifact-root> --stage train
python3 scripts/run_rgb150_dev30.py --artifact-root <artifact-root> --stage evaluate
python3 scripts/run_sem384_dev270.py --artifact-root <artifact-root> --stage prepare
python3 scripts/run_sem384_dev270.py --artifact-root <artifact-root> --stage train
python3 scripts/run_sem384_dev270.py --artifact-root <artifact-root> --stage inference
python3 scripts/build_dev30_full_chain_review.py
```

No quality gate can stop a numerically valid run, and no holdout result can change the frozen membership or endpoint.
