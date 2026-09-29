# SUBMIT-020-SEM161-FREEZE — current-best preservation

## Result

Completed on 2026-09-06 before starting the new multi-view geometry research
line. The current `RGB-030-FULL + SEM-161` candidate is pinned by hashes for
its RGB geometry/MLPs, SEM-001 sidecar, frozen 2D checkpoints, fusion results,
developer prediction manifest and external-camera contract.

This is an additive hash lock, not a new model and not a copy over the existing
`SUBMIT-010-BUNDLE`. New experiments may read these parents but cannot mutate
or replace them. The rollback order remains SEM-161, SEM-151, SEM-141,
SEM-111, then SEM-001.

The frozen local developer result is mIoU 0.833870. Deck IoU/precision/recall
are 0.914667/0.950392/0.960525. These are reused-development metrics, not a
hidden-test claim.

