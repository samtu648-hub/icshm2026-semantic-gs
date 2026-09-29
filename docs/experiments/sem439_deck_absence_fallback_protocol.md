# SEM439 zero-deck anomaly fallback protocol

## Question

Can the recoverable 1.5x deck signal found by SEM438 be restricted to views where the native model loses deck completely, without changing ordinary labeled views or protected tower/foundation pixels?

## Frozen inputs

- SEM438 r3 native and global-1.5x PNGs for the same 544 frozen RGB170 renders.
- SEM437-12k checkpoint and RGB/export hashes.
- No network inference, training, geometry, color, opacity, camera, threshold scan, or weight tuning.

## Fixed rule

For each view, trigger only when native Class ID 1 count equals zero and global-1.5x Class ID 1 count is at least 256. On a triggered view, replace with Class ID 1 only pixels where global-1.5x predicts Class ID 1 and native is background or `stay_cable`. Keep all other native labels; never overwrite tower or foundation.

## Evaluation

The all300 predictions must be exactly native. Report every trigger in released 301--400 and the 144-frame trajectory, including added deck pixels and changed fraction. Views 355 and 356 must each add at least 256 deck pixels; 357 must remain native because its fallback also has zero deck. No-GT images are qualitative development evidence only.

Retain the fallback only if all identities hold, per-trigger changed fraction is at most 0.2%, and visual review shows the additions follow plausible bridge-deck support rather than isolated water/sky blobs. Even when retained, SEM437 native remains the accuracy-backed rollback because the trigger has no labeled counterpart.
