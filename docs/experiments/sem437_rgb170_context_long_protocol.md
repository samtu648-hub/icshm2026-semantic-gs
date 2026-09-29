# SEM437 RGB170 context-mix long-training protocol

## Question

Test whether the fixed SEM386 full-frame context intervention transfers to the frozen RGB170 all-300 semantic model and improves the tower-free, distant midspan failure pattern visible around released views 344--358. This is development research, not a blind-test claim.

## Frozen control and single variable

SEM436 is the matched control. RGB170 bytes and model remain frozen. Mask2Former Swin-L, ADE initialization, seed 20260805, batch 1, losses, optimizer, AMP policy, 6000-step base phase, fixed 0.1 learning-rate tail through 12000, endpoints 6000/8000/10000/12000, inference and labels remain unchanged.

Exactly odd-numbered successful updates retain the parent native 768x768 crop. Exactly even-numbered updates retain the same parent view and flip but replace the crop with the complete 1320x989 image resized aspect-preservingly to 1025x768. RGB uses bilinear interpolation and labels nearest-neighbor. The parent sampler is still called every step, preserving its RNG use and view/flip trajectory. This method changes context and object scale together.

Native Swin BasicLayer activation recomputation is enabled only to fit the dynamic full-frame input on the local 12GB GPU. It adds no weights and is disabled for inference. A discarded-model eight-update preflight must cover four native and four context updates before training.

## Preregistered diagnostics and evaluation

Before candidate training, the sealed SEM436-12k model is applied to both original photographs and RGB170 renders for 344--358. This no-GT probe is explanatory only; the context recipe is already frozen and the probe cannot select it.

The candidate follows one 12k trajectory and reports every endpoint. Primary quantitative reporting covers all300 training-fit five-class and foreground mIoU, all foreground classes, stay_cable precision/recall/FP/FN and 2px boundary F1. The comparison must also include 344--358 per-frame class area, control/candidate changed pixels, central continuity review, the full released 301--400 set, and all 144 trajectory frames. No claim of new-view accuracy is allowed without GT.

Stop and preserve evidence on OOM under the registered memory policy, nonfinite optimization, RGB hash change, trace mismatch, checkpoint corruption or severe class collapse. Do not lower resolution, batch size or model capacity within this ID.
