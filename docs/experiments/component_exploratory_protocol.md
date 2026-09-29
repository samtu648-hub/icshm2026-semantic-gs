# Fixed component-quota exploratory pair after SEM416

2026-09-21, registered before any new training or quality access. The user has
explicitly authorized continuing both trials and worthwhile follow-ups without
another request, asking not to be excessively defensive. SEM416's prevalence
gate failed: the final mask loss gives <4 effective points to 83/1529 small
component observations (5.43%) at both late snapshots. The matcher has 334/1529
low-coverage observations. Do not rewrite that gate as passing or claim a
general loss-sampling deficiency.

SEM417/418 remain not run under their original conditional protocol. Separately
register SEM421/422 and QUAL015 for a direct exploratory test of the same quota,
without changing its 16 points, <=1024 full-image component area, 5% cap, private
seed, matcher, loss, model, crop, optimizer, 6000 endpoint or inference. This
tests whether protecting the minority of poorly covered components has useful
net effects despite the weak prevalence rationale. It is not a new mechanism
or independent confirmation, and no parameter search follows either outcome.

Use frozen RGB154, v2 255/30/21 roles, semantic seed20260805 and matched fresh
ADE initializations. After the independent DropGaussian queue releases the GPU,
execute two discarded eight-update preflights, fresh paired6000 fits, seal both
checkpoints and traces, then fixed inference and independent PNG/count/scale
audit. Reuse the unchanged QUAL014 sampler implementation and preflight; retain
separate source/config binding and execution logs for this added pair.

The original retention rule applies: foreground mIoU > control, boundary F1
>= control, cable precision loss <=0.1pp, all foreground IoU losses <=0.5pp.
Always report small-component recall/zero detections, FP/FN, all classes/regions
and cost. Preserve original RGB and first-seed models and record before/after
three training-view RGB checks. No blind/stability claim, no second candidate,
no automatic combination with DropGaussian. Estimate 70--100 minutes and4GiB.
