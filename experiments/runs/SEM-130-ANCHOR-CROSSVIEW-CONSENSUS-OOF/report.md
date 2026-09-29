# SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF — strict spatial-OOF anchor consensus

## Status

Completed on 2026-09-03; the calibration gate failed, so SEM-130 stopped
before the training check and before the 74-view development set. It is a
negative result, not a candidate for replacement.

SEM-130 is a training-only optimisation experiment. It does not alter frozen
Scaffold geometry, RGB, opacity, the SEM-001 checkpoint, or any SegFormer
checkpoint. It is not a replacement candidate and cannot open the 74-view
developer labels unless a separately registered successor is justified by its
independent training check.

## Why this mechanism

SEM-123 showed that selectively restoring the fixed 0.20--0.30 probability
band can recover true `stay_cable` pixels, but it still reintroduces some
yellow-background errors. SEM-070 showed that global anchor deletion is not
safe: the same anchor can support legitimate cable pixels. Therefore SEM-130
does the inverse, local operation: it preserves every hard-0.30 cable pixel
and only accepts a *new* proposed cable pixel when the same dominant frozen
anchor is supported by other cameras.

For target camera `v`, sources are restricted to other OOF cameras in `v`'s
existing spatial SEM-101 fold. Their one frozen OOF model was trained without
all cameras in that fold, including `v`; the target camera itself is removed
from the support count. A source view supports an anchor only if it dominates
at least 8 pixels and its mean OOF `stay_cable` probability across them is at
least 0.30. The only registered sweep is the minimum number of such independent
source views: `1, 2, 3, 4, 6, 8`.

## Protocol

1. Hash-check the frozen sources and build the fold-local anchor support table
   from 219 training cameras with no label, RGB image, unlabelled image,
   validation view or external camera input.
2. Use deterministic 108-view calibration labels to choose one source count
   by the registered score and gates.
3. Read 111 disjoint check labels once for that frozen choice only if one
   calibration candidate qualifies.
4. Do not open 74-view developer labels in this experiment. A passed check can
   only justify a separately registered developer audit with no source-count
   retuning.

## Command

```bash
"$R1_ENV_ROOT/bin/python" scripts/run_sem130_anchor_crossview_consensus_oof.py \
  --config experiments/runs/SEM-130-ANCHOR-CROSSVIEW-CONSENSUS-OOF/config.yaml \
  --split experiments/splits/sem_001_scaffold_static_v1.json \
  --oof-split experiments/splits/sem_005_oof_spatial_v1.json \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --data-root "$ICSHM_DATA_ROOT" \
  --rgb-run-dir "$ICSHM_OUTPUT_ROOT/r1/runs/RGB-030-FULL" \
  --source-root "$R1_THIRD_PARTY_ROOT/scaffold-gs" \
  --sem001-checkpoint "$ICSHM_OUTPUT_ROOT/r2/sem_001/full/checkpoints/semantic_sidecar.pth" \
  --sem001-training-root "$ICSHM_OUTPUT_ROOT/r2/sem_110/sem001_training_predictions/full-20260902-r1" \
  --sem101-oof-root "$ICSHM_OUTPUT_ROOT/r2/sem_101/rendered_segformer_oof/full-20260902-r3" \
  --sem101-oof-summary "$ICSHM_OUTPUT_ROOT/r2/sem_101/rendered_segformer_oof_summary/full-20260902-r3/result.json" \
  --sem110-result "$ICSHM_OUTPUT_ROOT/r2/sem_110/protected_fusion_oof/full-20260902-r1/result.json" \
  --sem123-result "$ICSHM_OUTPUT_ROOT/r2/sem_123/band_recovery_dev_sweep/full-20260903-r2/result.json" \
  --extension-root "$ICSHM_OUTPUT_ROOT/r2/sem_070/torch_extensions_preflight_20260902_retry2" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_130/anchor_crossview_consensus_oof/full-20260903-r1"
```

## Executed result

All 219 authorised training cameras first contributed only frozen
dominant-anchor / spatial-OOF probability evidence: 189,073 frozen anchors
received 345,375 fold-local source-support observations. No label, real RGB,
unlabelled image, development view, hidden input, or external camera was read
in that phase. Source support was correctly restricted to a different camera
in the target's own OOF fold; the target is absent from that fold model's
training data.

One initial process interruption exposed a nested-config field lookup after
the source table was complete and immediately after calibration began. It wrote
no candidate metric or prediction and did not open check/development labels.
The field lookup was corrected, then the immutable source-table cache and the
same registered input manifest were resumed; no rule, source input or threshold
was changed in response to labels.

The 108-view calibration hard-0.30 baseline had mIoU `0.796303`,
`stay_cable` IoU/recall `0.821512/0.959647`, and 1,776,579
background→`stay_cable` pixels. Unfiltered SEM-123 `p20_d02_any` recovered
243,164 pixels, but fell to `0.793761/0.810253/0.962530` and increased
background→`stay_cable` to 1,984,403.

Every registered source-count rule improved on that raw recovery, but none
reached hard 0.30 in both mIoU and cable IoU. The closest rule, eight other
supporting cameras, retained 94,503 recovery pixels and obtained mIoU
`0.795336`, cable IoU/recall `0.817223/0.960852`, and 1,856,303
background→`stay_cable` pixels. It still lost 0.097pp mIoU and 0.429pp cable
IoU against hard 0.30, so no count qualified. The other counts 1/2/3/4/6
likewise failed only those two quality gates.

The key diagnostic is that the stricter count does not selectively remove
background: at count 8 it retained 12,932 true `stay_cable` additions but also
79,724 background additions. Thus repeated OOF evidence for the same frozen
anchor is shared by the persistent annotation/prediction error, not independent
physical evidence that separates cable from yellow background.

No check label or 74-view label was opened. Result SHA-256:
`48e4f481dd5cfb6e035bd1da81fb329375788a25e2567afd44254f0661280875`.
The post-stop frozen RGB isolation audit on training views 001/017/031 reports
artifact and tensor hashes equal with maximum pixel difference zero; its SHA-256
is `a8467c8803370da63ed363a2507b6ded6e34713606cdf7c11ad645f05a865969`.

## Decision

Drop the entire SEM-130 source-count grid. Do not tune its 8-pixel source
minimum, 0.30 mean probability, source-count values, OOF-fold restriction, or
the SEM-123 recovery predicate inside this experiment. A future semantic route
needs a signal that is not another view's version of the same two-dimensional
annotation error—for example independently curated physical wire supervision
or a new, separately registered geometric measurement. SEM-111 remains the
historical qualified candidate and SEM-001 remains the immediate rollback.

## Rollback

Whatever the outcome, `RGB-030-FULL Scaffold-GS + SEM-001` remains the
immediate rollback. SEM-111 remains the historical qualified protected-fusion
candidate; neither SEM-123 nor SEM-130 may be described as a blind or formal
replacement.
