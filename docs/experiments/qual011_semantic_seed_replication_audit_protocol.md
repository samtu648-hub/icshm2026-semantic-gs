# Two-seed semantic-backbone replication audit

Experiment ID: `QUAL-011-SEMANTIC-SEED-REPLICATION-AUDIT`.

This read-only audit joins the sealed SEM396/SEM407 seed-20260805 pair and the
sealed SEM410/SEM411 seed-20260919 pair. It performs no training, inference,
threshold search, checkpoint selection, or metric recomputation from labels.

The AbsGS-backbone finding is seed-replicated only if, in each seed separately,
AbsGS exceeds Standard on overall five-class mIoU, `stay_cable` IoU, and
two-pixel boundary F1, and the same three differences are positive in region 1.
The audit also reports foreground mIoU, precision/recall, regions 2/3, two-seed
means, and execution invariants. No claim of statistical significance is made
from two seeds.

All four results are historically exposed spatial development evidence, not
blind or official hidden-set evaluation. Semantic-seed replication does not
remove dependence on the single spatial split or single RGB training seed.
