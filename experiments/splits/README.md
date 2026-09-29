# Evaluation splits

This directory stores lightweight, version-controlled camera-list splits. It never stores images, labels, camera parameter files, renderings, checkpoints, or absolute paths.

`rgb_r1_spatial_v1.json` is created from official COLMAP camera centers by `scripts/create_rgb_split.py`. Each validation fold holds out one spatial cluster and excludes a fixed guard band from all RGB training, including the 100 unlabeled images. This prevents close cameras from inflating the R1 backbone comparison.

Regenerate the split only if the released camera set changes. Any methodological change to its clustering or guard policy requires a new split ID rather than silently overwriting the old one.
