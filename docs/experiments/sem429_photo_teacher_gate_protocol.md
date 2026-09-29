# SEM429 photo-teacher gate protocol

SEM429 is a zero-training diagnostic for the proposed photo-teacher/render-student route. It does not change the frozen RGB models or the Mask2Former checkpoints.

The paired real photo and reconstruction render use the same official COLMAP camera and native 1320×989 grid. The same frozen semantic checkpoint predicts the photo twice: normally and after a horizontal flip, with the second probability map flipped back. A pseudo-label is accepted only when both photo predictions agree, belongs to `deck` or `tower`, lies in a four-connected stable teacher component of at least 128 or 64 pixels respectively, and the frozen render prediction is `background`. Thus the predicted `stay_cable` and `foundation` masks are bitwise invariant. This restriction is fixed before the 30 validation labels are read and directly addresses QUAL005: photo input recovered missing bridge structure but sometimes damaged cable and foundation.

The spatial-confirmation-v1 model SEM396 is used on its 30 historically exposed, spatially held-out views. All teacher outputs, masks, candidates, checkpoint hashes and source hashes are sealed before validation labels are decoded. The unchanged rule is then applied with the full-data SEM378 checkpoint to released unlabeled views 301–400. Public-unlabeled outputs are qualitative pseudo-label candidates, not ground-truth metrics.

Training is authorized only if changed-pixel precision is at least 60%, net corrections are positive, five-class mIoU does not decrease, and the protected prediction masks remain exact. Failure stops this gate definition; thresholds, component sizes, classes and source transitions are not adjusted inside SEM429.
