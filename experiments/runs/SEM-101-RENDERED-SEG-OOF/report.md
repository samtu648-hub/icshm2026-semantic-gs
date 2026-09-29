# SEM-101-RENDERED-SEG-OOF — rendered-RGB SegFormer spatial OOF

## Status

Completed on 2026-09-02. The fixed training-side rendered-domain OOF gate
passed on all five criteria. This qualifies only a separately registered final
rendered-domain 2D model; it is not a three-dimensional semantic candidate and
is not a 74-view result. The fallback remains
`RGB-030-FULL Scaffold-GS + SEM-001`.

The first rendering attempt wrote 124 training RGB PNGs to the retained,
Git-ignored `full-20260902` directory before the command service interrupted it
at its bounded execution window. A second persistent-session attempt similarly
stopped after 18 PNGs in retained `full-20260902-r2`. Neither wrote a completed
rendered-dataset manifest, OOF prediction, metric, model checkpoint or label
access. The retry uses the unchanged config/data/split/seed in
`full-20260902-r3`, with fixed 40-view runtime shards and a separate finalizer;
this is runtime repair, not an experimental-variable change.

## Question

`SEM-100` showed that a SegFormer trained on real images retains most bridge
`stay_cable` recall on frozen Scaffold renderings, but adds substantial
background→`stay_cable` false positives. The one changed variable here is to
train/evaluate the same 2D SegFormer recipe on same-camera frozen renderings,
rather than real RGB, using only the 219 labelled training views and the frozen
three-fold spatial OOF split.

## Frozen protocol

- Render exactly 219 `sem_001_scaffold_static_v1` training cameras with frozen
  `RGB-030-FULL`, official `CAMERA_FROM_WORLD`, identity Sim(3), and appearance
  index 0. Rendering opens no RGB reference or label and produces a hash
  manifest plus repeated-render/artifact RGB=0 evidence.
- Reuse SEM-051's SegFormer-B2 implementation, pretrained revision, seed
  20260805, three spatial folds, final-3,000 checkpoint budget, weighted CE,
  768 crop, flip, AdamW, AMP, and 768/512 sliding inference. The three folds
  consume only the rendered training images and their authorized labels.
- Require every one of the 219 training images to have exactly one prediction
  from its spatial OOF checkpoint. Fixed 74-view, unlabelled, hidden data and
  external-camera inputs have no runner mode and remain closed.
- Before any result exists, freeze the OOF qualification gate: mIoU ≥0.70,
  `stay_cable` IoU ≥0.70, precision ≥0.80, recall ≥0.90 and
  background→`stay_cable` fraction ≤2%. All are required; there is no
  threshold, checkpoint, seed, architecture or loss choice after results.

## Decision boundary

If the fixed training-side OOF gate fails, record the failure and do not train a
final model or open 74-view. If it passes, it only supports a separately
registered final rendered-domain model and then a once-only 74-view/external
camera test. It never changes the frozen RGB artifact, SEM-001 checkpoint or
current fallback.

## Completed evidence

The completed `full-20260902-r3` rendering manifest contains exactly the 219
authorised training-camera PNGs. Its SHA-256 is
`a8e08740ac8fa30f115f26cb822b604addca2b2f4c0a316ac4914f1fdd9ceb67`.
Its access record reports no validation, unlabelled, hidden-label or external
camera access. Re-rendering `001.png`, `017.png`, and `031.png` reproduced both
the frozen RGB artifact hashes and PNG hashes, with maximum absolute RGB
difference zero.

The three fixed spatial folds produced 95, 84, and 40 OOF predictions,
respectively, so all 219 training views have exactly one prediction from a
checkpoint that did not train on that view. All three final-3,000 checkpoints
reloaded successfully. Total training time was 907.39 s (0.2521 GPU h), total
OOF inference time was 171.57 s, and peak allocated GPU memory was 3,203 MiB.

Merged OOF metrics are: mIoU including background `0.849734`, mIoU excluding
background `0.818663`, `stay_cable` IoU `0.836124`, precision `0.860914`, and
recall `0.966708`. There are 3,383,866 background→`stay_cable` pixels, a
background fraction of `0.014137` (below the fixed 0.02 limit). Thus mIoU,
`stay_cable` IoU, precision, recall, and background-fraction gates all pass.
The Git-ignored summary result has SHA-256
`88fd816d8fa9d56c0a78dece570ca38ca02d43d66a801aee671e6c4de1681923`; it also
stores fixed rendered-RGB / GT / OOF prediction / error panels for `001`,
`017`, and `031`.

This is evidence that learning in the frozen rendered-RGB domain resolves the
training-side domain drift identified by SEM-100. It is not evidence that a
final model generalises to the closed 74-view split or official external
cameras. No threshold, seed, architecture, loss, checkpoint choice, real RGB
input, unlabelled image, validation view, or external-camera input was changed
or opened after these results.

## Outcome

The OOF qualification is complete and may be used to register one distinct
final rendered-domain model experiment. That future experiment must freeze this
recipe before training, retain the above access boundary until its final model
is frozen, and perform a once-only 74-view plus official external-camera audit.
It cannot replace SEM-001 automatically and has not started in SEM-101.
