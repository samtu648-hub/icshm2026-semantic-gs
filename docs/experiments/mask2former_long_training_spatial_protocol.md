# Mask2Former 6k--12k spatial-development curve

Experiment IDs: `SEM-434-RGB152-MASK2FORMER-LONG12K` and
`SEM-435-RGB154-MASK2FORMER-LONG12K`.

## Question and evidence boundary

The experiment asks whether the current ADE-pretrained Mask2Former Swin-L is
under-trained at 6000 successful updates and whether longer optimization improves
spatially separated rendered views rather than only the training labels. Both
spatial folds have already been opened for development, so all results are
development curves. They are not blind, hidden-test or official scores.

## Recovery decision

SEM396 and SEM412 save model weights and bindings at step 6000, but do not save
AdamW state, GradScaler state, torch RNG or sampler RNG. Their author polynomial
learning rate is also effectively exhausted at the endpoint. Direct resume is
therefore rejected. Each new run starts from the same fresh ADE initialization,
replays the same first 6000 samples and schedule, records step-6000 model and
prediction equality to its parent, and continues without recreating the optimizer.
If historical equality fails despite identical initialization and samples, the new
run's own sealed step 6000 is the only valid control for its later endpoints.

## Frozen recipe

- Model: Mask2Former Swin-L, ADE pretrained, five project classes.
- Input: frozen RGB152 for v1 and frozen RGB154 for v2; native 768 crop, batch 1.
- Seed: 20260805; losses, flip augmentation, AdamW groups, AMP retry and inference
  are unchanged from SEM396/SEM412.
- Steps 1--6000: author polynomial factor `(1-(step-1)/6000)^0.9`.
- Steps 6001--12000: fixed 0.1 times each optimizer group's initial LR, i.e.
  `1e-5` for ordinary/decoder groups and `1e-6` for Swin backbone groups.
- Endpoints: 6000, 8000, 10000 and 12000 on one continuous trajectory.
- Full resumable state is saved at 6000 and 12000; model checkpoints are saved at
  every evaluation endpoint.
- v2 uses this exact recipe regardless of the v1 result. No endpoint-specific
  restart, learning-rate rescue, input-size change or class-specific tuning is
  allowed.

RGB geometry, color, opacity, cameras and renders are read-only. The pre/post RGB
hash set must match. Any numerical error, nonfinite update, RGB change, serious
class collapse, external GPU conflict or resource failure stops the current run
with evidence preserved; it does not authorize a smaller model, crop or batch.

## Evaluation

After all four checkpoints in a fold are sealed, evaluate every endpoint with the
same native 768/512 sliding inference. Report training and confirmation five-class
and foreground mIoU, their gaps, each class IoU, `stay_cable` precision/recall,
FP/FN and 2px boundary F1, all three spatial regions, the worst view, time, memory
and disk. An independent process must reconstruct confusion and boundary counts
from saved PNGs.

The first sign of overfitting is reported when training fit improves while spatial
confirmation worsens and the train--confirmation gap expands. A full-data transfer
is considered only if one longer endpoint improves or preserves both five-class
and foreground confirmation mIoU in both folds, does not incur more than 0.5pp
loss in deck, tower, foundation or `stay_cable` 2px boundary F1 in either fold,
and does not show a material gap expansion. Among qualifying endpoints, prefer the
earliest one within 0.05pp of the best mean confirmation mIoU.

If both folds qualify, train a matching all-300 RGB140 model under a new ID and
compare it with SEM378 before reviewing the 100 released unlabeled images and the
144-frame trajectory. If the folds disagree, do not transfer; a second seed or a
new LR-tail experiment may be proposed but is not automatically run. If both
overfit, stop at 6000. Scaffold RGB136 is considered only if the Standard folds
suggest a duration-by-backbone interaction; an all-300 Scaffold fit alone cannot
answer spatial generalization.
