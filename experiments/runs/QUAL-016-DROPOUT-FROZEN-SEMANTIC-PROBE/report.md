# QUAL-016-DROPOUT-FROZEN-SEMANTIC-PROBE

Completed2026-09-21; four frozen
RGB/semantic combinations,120 predictions, no training. Original RGB quality
gate remains failed and SEM419/420 remain not run.
[Protocol](../../../docs/experiments/dropout_frozen_semantic_probe.md).

Both semantic models were trained only on the v1 training labels. All120
predictions were sealed before target labels were opened; checkpoint and input
hashes remain exact, with120 independently counted confusion matrices.

- Seed20260805: dropout-minus-control foreground-2.814603pp,
  five-class-2.260634pp, cable IoU+2.231582pp.
- Seed20260919: foreground-1.697606pp, five-class-1.369089pp,
  cable IoU+4.905427pp.
- Region1 foreground changes:-5.023855/-2.794804pp. Cable gains do not compensate
  for other structural-class losses.
- Runtime69.492 seconds; peak allocated1479.326MiB. No parameter updates.

Conclusion: `matched_retraining_supported=false`. Close this fixed dropout
recipe; no rate/weight search or extra semantic fits. The probe changes rendered
inputs under fixed RGB152-trained semantic models, so it is not a matched-domain
training comparison and cannot prove all future dropout methods ineffective.
