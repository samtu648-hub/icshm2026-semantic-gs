# SEM-210-XYZ-DECK-FIELD-OOF — continuous world-coordinate deck field

## Status

Completed on 2026-09-06. No calibration threshold passed; check and the reused
74-view development set remained unopened.

## Question and single change

SEM-201 established that a conservative interior deck surface is precise enough
to avoid aggregate regression, but sparse long-baseline voting reached only 261
true added deck pixels on calibration. SEM-210 changes the representation from
discrete source pixels to a continuous, view-independent field that maps a
visible frozen RGB-030-FULL surface point in official world coordinates to a
deck probability.

Each spatial-OOF fold trains only on labels from its training cameras, sampled
on the frozen surface every eight pixels. The held-out camera contributes only
its external camera and frozen renderer depth; its RGB and label are unavailable
to prediction. Model architecture, optimizer, seed, iterations and the nine
calibration thresholds are fixed in the registered config.

Only foundation-safe SEM-160 background may become deck. Existing deck,
`stay_cable`, tower, foundation and rejected foundation pixels remain exact.
All 219 OOF probability maps are sealed before calibration is read. Check opens
only if a calibration threshold passes every fixed gate, and the reused 74-view
development set remains closed.

## Result

The three folds trained on 2.33M/2.62M/3.62M frozen-surface samples. Their final
losses were 0.04447/0.04250/0.04876, all checkpoints reloaded exactly, and all
219 OOF maps were sealed. The best registered threshold was 0.995, but relative
to SEM-160 it produced:

- mIoU -0.13828 pp
- deck IoU -0.64840 pp
- deck recall +0.03776 pp
- final deck precision 94.93%

It added 60,053 pixels with only 4.72% true-deck precision and raised the
background-to-deck error ratio to 1.2913. The continuous field learned bridge
location but not the visible deck surface boundary. Probability/result hashes
are `8200538c...fcda` and `0df56387...fd4d`. Do not train a final XYZ field.
