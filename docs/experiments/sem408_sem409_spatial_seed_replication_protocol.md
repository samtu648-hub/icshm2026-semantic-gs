# Paired semantic-seed replication protocol

Experiment IDs: `SEM-408-RGB152-SPATIAL256-SEED20260919` and
`SEM-409-RGB153-SPATIAL256-SEED20260919`.

Execution note: SEM408 completed all 6,000 updates but failed its post-training
contract before any confirmation access because the inherited runner required a
new-seed full state to equal the old-seed SEM378 state. SEM409 training was not
started. Both IDs are closed and preserved. The corrected, newly registered
pair is `SEM-410-RGB152-SPATIAL256-SEED20260919` and
`SEM-411-RGB153-SPATIAL256-SEED20260919`; it uses the already sealed SEM392
seed-20260919 initialization contract, whose 650-tensor state hash exactly
matches SEM408's observed fresh initialization. No model, data, optimizer,
schedule, sample, budget, or inference variable changes.

## Question

Does the large downstream semantic advantage observed for the RGB152 AbsGS
backbone over the matched RGB153 Standard backbone persist under a second
fresh Mask2Former initialization and sampling sequence?

## Frozen comparison

Both arms reuse `spatial_confirmation_v1` byte-for-byte: 256 semantic-training
labels, 30 confirmation labels, and 17 excluded buffer views. The semantic
recipe remains the SEM396/SEM407 ADE-pretrained Swin-L Mask2Former recipe with
a 768 crop and 6,000 successful updates. Both arms use seed 20260919, which
determines both the fresh model initialization and the precomputed 6,000-step
view/crop/flip plan.

The only between-arm variable is the frozen RGB reconstruction input:

- SEM408 uses RGB152 AbsGS renders;
- SEM409 uses RGB153 original-Standard renders.

Preparation must prove byte-identical sample lists between the two arms.
Training must prove byte-identical initial model-state hashes between the two
arms. Confirmation images and labels remain inaccessible until both step-6000
checkpoints are sealed. No threshold, checkpoint, or hyperparameter may be
changed after either confirmation result is opened.

## Decision rule

Report paired differences for five-class and foreground mIoU, `stay_cable`
IoU/precision/recall, two-pixel boundary F1, and all three spatial regions.
The AbsGS-backbone conclusion is considered seed-replicated only if SEM408
exceeds SEM409 in overall five-class mIoU, `stay_cable` IoU, and boundary F1,
with the direction of the region-1 gap agreeing with SEM396 versus SEM407.

This is a historically exposed spatial development split, not blind testing.
The second seed tests semantic-training stability only; it does not establish
independence from the chosen spatial split.
