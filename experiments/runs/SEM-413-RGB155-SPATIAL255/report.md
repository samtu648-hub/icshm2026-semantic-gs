# SEM-413-RGB155-SPATIAL255

Completed on 2026-09-20 as the Standard semantic arm of the disjoint spatial-
fold replication. Its initial state and every one of the 6,000 view/crop/flip
samples match SEM412 exactly. Training took 1,737.907 seconds with seven AMP
skips; the sealed checkpoint hash is
`5ebf86378202c7101f2944ff4920584c28962addb140898ceafa082498a7175a`.

## Results (%)

| Role | Five-class mIoU | Foreground mIoU | `stay_cable` IoU | Precision | Recall | 2px boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 95.465728 | 94.473848 | 95.715180 | 97.584183 | 98.038242 | 88.377008 |
| Confirmation | 89.443071 | 87.005705 | 80.698006 | 86.304734 | 92.549489 | 70.061941 |
| Region 1 | 80.647941 | 75.915001 | 56.341405 | 74.144875 | 70.117242 | 69.210688 |
| Region 2 | 92.885166 | 91.179503 | 92.603639 | 96.128817 | 96.190808 | 87.416118 |
| Region 3 | 87.552620 | 84.880191 | 78.659766 | 82.919624 | 93.869317 | 59.240964 |

SEM412-minus-SEM413 confirmation differences are +0.501025 pp five-class
mIoU, +0.594970 pp foreground mIoU, +2.917695 pp `stay_cable` IoU,
+3.369702 pp precision, -0.025657 pp recall, and +5.405989 pp boundary F1.
The pre-registered semantic direction gates all pass. Region-1 overall mIoU is
1.109985 pp lower under AbsGS even though cable IoU and boundary F1 improve;
this mixed result is retained as a claim boundary.
