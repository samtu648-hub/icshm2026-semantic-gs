# SEM-408-RGB152-SPATIAL256-SEED20260919

Registered on 2026-09-20 as the RGB152 arm of a second paired semantic-seed
replication. It uses seed 20260919 and otherwise freezes the SEM396 recipe,
data roles, budget, and inference.

Status: stopped and preserved on 2026-09-20; no confirmation access.

All 6,000 successful updates completed in 1,638.16 seconds with four AMP
retries. The post-training contract then rejected the run because the inherited
runner incorrectly required the fresh seed-20260919 full state to equal the
old-seed SEM378 full state. The observed initialization SHA256,
`a9f66a9ada969f1bde050ac3cc184ae332d72877b25caeda56b7e9436de9f7d2`,
exactly matches the previously sealed SEM392 seed-20260919 preflight state.
Thus this is a contract-design failure, not a numerical or quality result.

The failed result and step-6000 checkpoint are retained, but inference is
forbidden under this ID. The corrected pair proceeds only under SEM410/SEM411.
