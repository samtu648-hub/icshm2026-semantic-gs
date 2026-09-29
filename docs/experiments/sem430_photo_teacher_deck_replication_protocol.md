# SEM430 deck-only photo-teacher replication

SEM429's preregistered combined deck/tower rule passed its aggregate gates, but the sealed class breakdown showed that deck was strongly positive while tower was harmful. SEM430 therefore changes exactly one thing: it removes the tower branch. The remaining rule is unchanged: photo normal/flip agreement, a four-connected stable deck component of at least 128 pixels, and a frozen render prediction of background.

The rule is repeated using the SEM412 checkpoint and RGB154 render/photo pairs on `spatial_confirmation_v2`, whose 30 confirmation views do not overlap v1. Inference artifacts and the gate are sealed before v2 labels are decoded. Promotion requires changed-pixel precision at least 75%, positive net corrections, nonnegative five-class and deck IoU deltas, and bitwise identity of every non-deck predicted mask.

After evaluation, the already sealed SEM429 public outputs are filtered to deck-only without new inference. No semantic or RGB parameter is updated. Passing authorizes designing a separate paired semi-supervised training experiment; it does not itself prove that training will improve hidden views.
