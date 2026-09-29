# SEM440 real-label weak-structure tail protocol

## Purpose

Test one final targeted hypothesis for the released no-GT failure around view 357: SEM437 already fits ordinary real tower-free views well, but has not learned their joint low-scale and low-contrast form.

## Frozen method

- Resume the exact SEM437-12k model, optimizer, AMP scaler, Torch RNG and sampler RNG.
- Keep the existing fixed 0.1-times low learning-rate phase; do not restart or decay it to zero.
- Run exactly 2,000 additional successful updates and save the 13k and 14k endpoints on one trajectory.
- Odd tail updates use the continuing all300 native-crop sampler.
- Even tail updates use a complete 1320x989 RGB170/real-label pair from the frozen 11-view pool, resized to 512x384 with bilinear RGB and nearest label interpolation.
- The pool is defined from labels before training as deck present and tower absent: 049--052, 206--207, 258--261 and 282.
- Do not access 301--400 or trajectory RGB during fitting. Do not use pseudo-labels, geometry changes or post-hoc tuning.

## Evidence and gates

Report the parent 12k control and both 13k/14k endpoints. Independently recompute all300 metrics, per-class IoU, stay_cable precision/recall and 2px boundary F1. A candidate must lose no more than 0.05 pp five-class/foreground mIoU, 0.10 pp in any foreground class, and 0.15 pp boundary F1 relative to SEM437-12k.

Because 357 has already been inspected and has no GT, its use is explicitly development-only. A target-behavior candidate must predict at least 256 deck pixels with a largest 8-connected deck component of at least 128 pixels, and visual review must place it on plausible bridge support rather than water or sky. These criteria do not establish accuracy.

If neither endpoint passes both protection and target-behavior checks, close this repair line. Do not respond by tuning another scale, ratio, seed or longer endpoint.
