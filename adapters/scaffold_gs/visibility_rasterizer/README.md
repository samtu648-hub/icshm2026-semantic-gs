# SEM-002 contribution adapter

This directory is a semantic-only CUDA extension, not a patch to the frozen
Scaffold-GS RGB renderer.  It was written against the pinned Scaffold-GS
upstream commit `59c833b56bbbf510f3f64d40f81721995caced66` and mirrors the
opaque workspace layouts in its `diff-gaussian-rasterization` submodule:

- `cuda_rasterizer/rasterizer_impl.{h,cu}` for `GeometryState`,
  `BinningState` and `ImageState` alignment/layout;
- `cuda_rasterizer/forward.cu` for the exact alpha cutoff, transmittance stop
  rule and front-to-back ordering.

The adapter reads frozen `means2D`, `conic_opacity`, tile ranges and point lists
from one no-gradient upstream preprocessing pass.  It then emits semantic
numerators and gradients only for logits of contributors selected by either a
front-to-back cumulative-mass gate or a relative-to-maximum gate.  Geometry,
opacity, color, covariance and all RGB MLP parameters are detached inputs;
gate selection is therefore stop-gradient with respect to RGB.

The compiled extension is cached only below the Git-ignored experiment output
root.  `SEM-020-EQ` verifies the disabled-gate renderer against SEM-001 and
also checks the adapter's all-contributor numerical agreement before S1 runs.
