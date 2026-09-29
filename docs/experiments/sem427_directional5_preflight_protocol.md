# SEM427 first-order view-conditioned semantic preflight

## Question

Can the failed SEM426 static per-Gaussian class distribution be replaced by the
smallest continuous view-dependent alternative without changing frozen RGB152?
This is an engineering and representation-signal preflight, not a quality test.

## Single changed variable

SEM426 stores five logits per Gaussian. SEM427 stores five affine functions of
the registered viewing direction: bias plus normalized Gaussian-to-camera
`x/y/z`. Geometry, opacity, SH appearance, native rasterizer, residual-background
rule, loss, class weights, Adam settings, seed, split and sample crops remain
fixed. This differs from SEM401--406: no position-offset, ray-depth or hard
responsibility partition is introduced.

## Isolation and gates

- Reuse the first eight samples of the sealed SEM426 6000-step plan.
- Read only the 256 training labels. Keep all 30 confirmation and 17 buffer
  RGB/labels sealed.
- Require eight finite, non-zero updates; separately require non-zero bias and
  directional gradient norms.
- Require a non-zero per-Gaussian probability response between two registered
  training cameras after training (`max abs delta > 1e-7`).
- Require probability-sum error at most `5e-7`, exact checkpoint state/render
  replay, exact frozen RGB state and fresh-render equality, and peak reserved
  CUDA memory below 12,288 MiB.
- Run an independent artifact audit. Only if every gate passes may a new full
  spatial-comparison experiment be registered; this preflight itself has no
  promotion or quality claim.

## Failure policy

Preserve any failure and close this exact affine-direction recipe. Do not tune
learning rate, basis order, loss, weights, geometry, fusion or post-processing
under SEM427.
