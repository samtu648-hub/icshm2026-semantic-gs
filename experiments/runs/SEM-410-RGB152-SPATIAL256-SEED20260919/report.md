# SEM-410-RGB152-SPATIAL256-SEED20260919

Registered on 2026-09-20 as the corrected RGB152 arm of the second paired
semantic-seed replication. It preserves SEM408's scientific variables and
replaces only the invalid old-seed full-state equality check with the sealed
SEM392 same-seed initialization contract.

Status: completed on 2026-09-20.

Preparation reproduced SEM408's 6,000-sample plan exactly. Before the first
update, the 650-tensor initial state and pretrained-load report matched the
sealed SEM392 seed-20260919 reference exactly; the state SHA256 is
`a9f66a9ada969f1bde050ac3cc184ae332d72877b25caeda56b7e9436de9f7d2`.
Training completed 6,000 successful updates in 1,635.15 seconds with four AMP
retries and 8,097.57/8,588 MiB peak allocated/reserved memory.

## Fixed evaluation

| Role | Five-class mIoU | Foreground mIoU | stay_cable IoU | 2px boundary F1 |
|---|---:|---:|---:|---:|
| Training fit | 95.451181% | 94.460335% | 95.412531% | 87.927525% |
| Spatial confirmation | 89.409963% | 86.926324% | 88.897017% | 78.185904% |
| Region 1 | 85.902002% | 82.653560% | 76.335758% | 76.922595% |
| Region 2 | 90.200062% | 87.907708% | 92.312661% | 69.750025% |
| Region 3 | 91.444097% | 89.364440% | 92.889340% | 92.592448% |

Relative to paired SEM411, confirmation five-class mIoU, `stay_cable` IoU,
and boundary F1 are higher by 15.669094/7.918670/9.884617 percentage points.
Region 1 repeats the primary effect at +37.101256/+38.895006/+39.058851 pp.
Region 2 boundary F1 is lower by 6.626270 pp, so the result supports difficult-
region protection rather than uniform superiority on every regional metric.

The checkpoint SHA256 is
`f5bad0766182e1dee7b7bef2a7e2e7ebd2d2fe6b48c678a923071d6b9a1bd4d9`
and remained unchanged after confirmation access. Training-result SHA256:
`3aeeb72dfb018e417b641c95ddf496c7e7c8bbe21763c1e8f9b4fcfd7ea40495`.
Inference-result SHA256:
`31fcac5a39ba2440e659057a89a6b37f11424d4bb823a582e63c333ffa874507`.
