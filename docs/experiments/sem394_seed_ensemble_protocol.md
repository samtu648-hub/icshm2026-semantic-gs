# SEM-394 frozen dual-seed probability ensemble protocol

Experiment ID: `SEM-394-RGB150-DEV270-SEED-ENSEMBLE`.
Registered on 2026-09-19 before any evaluation label of this run was read.

## Purpose

ADR-133 closed the SEM392/393 second-seed confirmation with a negative result, but the
two frozen native-crop models (SEM384, seed `20260805`; SEM392, seed `20260919`) are both
sealed and independently scored on the fixed `dev30_local_composition_v1` split. This
read-only experiment asks one question: does the single pre-registered candidate

```
P_ensemble = 0.5 * P_SEM384 + 0.5 * P_SEM392
```

improve the dev30 holdout over SEM392 (the stronger parent) enough to justify training an
RGB140 second seed for a full-data ensemble validation? No model is trained, no parent
artifact is modified, and no alternative fusion is tried.

## Frozen inputs

- Split: `experiments/splits/dev30_local_composition_v1.json` (270 train / 30 holdout
  labeled, 100 unlabeled), identical to both parents.
- SEM384 checkpoint `outputs/quality/SEM-384/r1/mask2former/model-6000.pt`,
  SHA256 `d9373e3330e5fc06f6c914638c1cc948b07faf5514d8b4677322dd7310a5d8a0`.
- SEM392 checkpoint `outputs/quality/SEM-392/r1/mask2former/model-6000.pt`,
  SHA256 `1763e203f02d9ac12c453abcd4f8d6521762af79232bb6bfa706b1f45a0e810a`.
- SEM384 inference result `outputs/quality/SEM-384/r1/inference/result.json`,
  SHA256 `a61ba1471be8e3399935aa20c1330de6e7c648015823278990b8ac53a9c39f78`.
- SEM392 inference result `outputs/quality/SEM-392/r1/inference/result.json`,
  SHA256 `cab0712b74912242a79418b8d347e902556874d7acab4099d54a8eee30e7424f`.
- Frozen RGB150 evaluation renders, `outputs/quality/RGB-150/full-r1/evaluation/images`
  (400 views, 1320x989).

## Stage 1: reproduction of the parent inference

For each parent, in one fresh GPU process under the shared GPU lock:

1. Verify the four input SHA256 values above and that the current
   `src/quality_mask2former.py` equals the hash recorded in both parent inference
   results (`debd14ba...`).
2. Build the model exactly as the parent inference stage did: official ADE Swin-L
   Mask2Former config (no activation checkpointing at inference), official pretrained
   weights, then the parent `model-6000.pt` state dict, eval mode, CUDA.
3. For all 400 views in the parent's sorted order, run the unmodified
   `native_predict` (native 768 window, stride 512, per-tile five-class score
   normalization, float32 accumulation, per-pixel count division, final normalization,
   cuda float16 autocast). No TTA, thresholds, class weights, or post-processing.
4. Validate per view: probabilities finite, shape `(5, 989, 1320)`, non-negative,
   per-pixel class sum within `2e-5` of 1, and the recomputed argmax prediction is
   pixel-identical to the parent's sealed prediction PNG (whose SHA256 must match the
   parent result manifest).
5. Save the five-class probabilities losslessly (float32 compressed NPZ) and the
   prediction PNG under `outputs/quality/SEM-394/r1/parent-<id>/`, and seal a
   `reproduce-result.json` with all artifact hashes. No label is read in this stage.

If exact reproduction fails, the parent-bound source snapshot is located and used; if it
still fails, SEM394 is registered as an engineering failure. Parent predictions,
checkpoints, and the fusion formula are never modified to make the check pass.

## Stage 2: fixed ensemble

Only after both parents' 400-view probabilities and predictions are complete and
hash-sealed:

1. Load both parent probability files per view, convert to float64, and compute
   `P_ensemble = 0.5 * P_SEM384 + 0.5 * P_SEM392` in float64.
2. Argmax in float64 with the fixed smallest-class-ID tie rule.
3. Verify the ensemble probabilities are finite, non-negative, and sum to 1 within
   tolerance; store them as float32 compressed NPZ (documented precision) and record that
   the stored-precision argmax equals the float64 argmax on every view.
4. Save all 400 ensemble prediction PNGs under `outputs/quality/SEM-394/r1/ensemble/`
   and seal `ensemble/result.json` with hashes. No label is read in this stage.

No other weight, logit average, confidence selection, per-class weight, or threshold is
tried under this ID.

## Stage 3: evaluation

Only after the 400 ensemble predictions are sealed, read the 270 training and 30 holdout
labels (the 100 unlabeled views keep predictions only) and compute, from independently
recomputed confusion matrices:

- train and holdout five-class / foreground mIoU, per-class IoU, precision, recall;
- `stay_cable` FP, FN, and 2px boundary precision / recall / F1;
- regular-local, oblique-local, distant-local group metrics;
- per-view improvement / regression counts and corrected / damaged pixels versus both
  SEM384 and SEM392;
- an independent recomputation of both parents' per-view confusions (600 checks) and of
  every ensemble confusion, with aggregate cross-checks;
- input-integrity diagnostics on the 30 holdout views: SEM384/SEM392 disagreement pixel
  share (~0.1633167%), only-SEM384-correct (~0.0605269%), only-SEM392-correct
  (~0.0994245%), disagreement-with-both-wrong (~0.0033653%), and the unreachable
  GT-selected hard-oracle mIoU (~94.5300558%). The oracle is a complementarity upper
  bound and input check only, never a candidate score;
- a fixed paired view bootstrap with replacement over the 30 holdout views, 10000 draws,
  RNG seed 394, SEM392 as baseline, reporting the five-class mIoU delta median, the 95%
  percentile interval, and `P(delta > 0)` (foreground mIoU, `stay_cable` IoU, and boundary
  F1 deltas are reported alongside).

## Pre-registered promotion gates (all relative to SEM392)

1. holdout five-class mIoU improves by at least `+0.10 pp`;
2. holdout foreground mIoU does not decrease;
3. no class IoU drops by more than `0.10 pp`;
4. no group (regular/oblique/distant) five-class mIoU drops by more than `0.15 pp`;
5. `stay_cable` 2px boundary F1 drops by no more than `0.10 pp`;
6. the paired bootstrap median five-class delta is positive and `P(delta > 0) >= 0.95`.

## Outcome rules

- All gates pass: SEM394 is registered as worth training an RGB140 second seed for a
  full-data ensemble validation. That training is not started here and the current
  default pipeline (RGB130+SEM371) and system candidate (RGB140+SEM378) are untouched.
- Any gate fails: SEM394 is closed with the negative result. No weight search, no
  remedial fusion, no rerun under the same ID.
- Reproduction failure that cannot be resolved from the parent-bound source: SEM394 is
  registered as an engineering failure and closed.

All outputs go to `outputs/quality/SEM-394/r1` only; the SEM384/SEM392 directories are
never written.
