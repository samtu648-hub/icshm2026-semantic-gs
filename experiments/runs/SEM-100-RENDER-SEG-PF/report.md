# SEM-100-RENDER-SEG-PF — frozen rendered-RGB semantic-domain preflight

## Status

Completed training-only preflight; not a quality candidate and not a 74-view
result.  `SEM-001` remains the only three-dimensional semantic fallback.

The first host attempt created only `training_only_access.json` and stopped before
loading the frozen Scaffold model because the isolated 2D environment lacked the
upstream `colorama` import.  It rendered no image, loaded no teacher checkpoint,
ran no prediction and read no label.  The retained partial output is
`r2/sem_100/render_seg_pf/preflight-20260902`.  The completed retry kept the
same code/config/data/OOF plan and uses the frozen Scaffold interpreter with the
existing 2D environment placed first on `PYTHONPATH`, preserving Transformers
4.57.1 while supplying the upstream renderer's pure-Python dependencies.  This
is an environment repair, not a model or experimental-variable change.  The
retry completed in 13.80 s at Git HEAD
`49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus the recorded dirty worktree.

## Result

All eight fixed training-side OOF views completed: fold 0 `017/018/019`, fold 1
`001/002/003`, and fold 2 `031/032`.  No validation, unlabelled, hidden-test,
or external-camera input was opened; all real and rendered predictions existed
before the training labels were read.  No parameter or checkpoint was written.

The frozen teacher retains genuine `stay_cable` evidence on frozen rendered RGB:
recall is 0.959029 versus 0.951849 on the corresponding real RGB OOF inputs.
However, its rendered-domain cable precision falls from 0.928544 to 0.799024,
and background→`stay_cable` rises from 47,875 to 163,214 pixels.  The rendered
input has mIoU 0.789742 (real input 0.833488), `stay_cable` IoU 0.772650 (real
input 0.886885), and a +115,339 background→`stay_cable` pixel delta.  These are
eight-view diagnostic comparisons only; they must not be ranked against the
74-view SEM-001 result or used to select a checkpoint.

The visual samples agree with the metrics: bridge cable evidence is mostly
preserved, while rendered view `031` adds a water/remote-background yellow
false-positive region.  Frozen RGB isolation passed: all six frozen artifact
hashes match before/after, three repeated RGB render PNG hashes match, and the
maximum RGB pixel difference is zero.  All OOF checkpoint hashes and the
five-class simplex checks passed.

## Question

`SEM-005` proved that a spatially OOF SegFormer-B2 can detect `stay_cable` in
real RGB, but its result is not an external-camera semantic renderer: at
inference an external pose provides a frozen Scaffold RGB rendering, not the
target photograph.  This preflight asks the narrow training-only question:
does the existing frozen OOF teacher retain meaningful semantic evidence when
its input is the frozen Scaffold RGB rendering at exactly the same official
camera?

## Frozen protocol

- Exactly eight named training views, all existing OOF views and spread across
  the three frozen `SEM-051-OOF` folds: fold 0 `017/018/019`, fold 1
  `001/002/003`, fold 2 `031/032`.
- Each view's existing real-RGB OOF probability file is hash-verified.  The
  same fold checkpoint is hash-verified, reloaded, and run only on the frozen
  Scaffold RGB rendered at that view's official `CAMERA_FROM_WORLD` pose.
- RGB geometry, colors, opacity, appearance, RGB checkpoint, SegFormer
  checkpoint and all SEM-001 parameters remain frozen.  There is no optimizer,
  no new checkpoint, pseudo-label, threshold selection, 3D semantic training,
  validation, unlabelled input, hidden input or external-camera mode.
- All real/rendered predictions finish before any label is opened.  Labels are
  then used only for authorised training-side diagnostic metrics and GT panels.

## Decision boundary

This is not a quality candidate and its eight-view metrics cannot choose a
model or replace `SEM-001`.  It passes only if OOF provenance, five-class
simplex, RGB isolation, artifact hashes and training-only access are intact.
The preflight supports a separate `SEM-101` preregistration for rendered-domain
2D training, because cable recall remains high, but it exposes a material
render-domain false-positive gap.  It does not authorize a 219-view quality
claim, a 74-view run, or reuse of the frozen real-RGB teacher as an external
camera model.  Any `SEM-101` must define its rendered-RGB training/OOF recipe
and formal external-camera gates before execution.  The fallback remains
`RGB-030-FULL Scaffold-GS + SEM-001`.
