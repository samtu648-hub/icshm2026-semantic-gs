# SEM-412-RGB154-SPATIAL255

Completed on 2026-09-20 as the AbsGS semantic arm of the disjoint spatial-fold
replication. The run used 255 training labels, seed 20260805, and 6,000 paired
768-pixel crops. Its 650-tensor initial state hash
`90b1b339261659d51baac01d6a6afc4fadc819589d5cabd5d72447088bd8e16f`
matches the sealed SEM378 initialization contract. Training took 1,735.801
seconds with four AMP skips; the sealed checkpoint hash is
`a2438e15d7053fceb36fc2394f7c3830a1eb5fb2f08afff339d9883f81bf564f`.

## Results (%)

| Role | Five-class mIoU | Foreground mIoU | `stay_cable` IoU | Precision | Recall | 2px boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 95.464580 | 94.472453 | 95.746151 | 97.641300 | 98.013116 | 88.445101 |
| Confirmation | 89.944096 | 87.600675 | 83.615702 | 89.674436 | 92.523833 | 75.467930 |
| Region 1 | 79.537956 | 74.528533 | 58.425038 | 75.931863 | 71.703875 | 73.085033 |
| Region 2 | 92.893267 | 91.189437 | 92.640820 | 96.179395 | 96.180294 | 89.126859 |
| Region 3 | 89.055970 | 86.661011 | 83.043737 | 88.050028 | 93.592057 | 68.137106 |

The first inference attempt stopped after 175 views when one FP16 tile produced
exact-zero semantic mass; it is preserved under
`inference-r1-failed-empty-mass`. The recovery re-evaluated only that unchanged
tile in FP32 and completed all 379 views without fabricating background. The
checkpoint hash was identical before and after confirmation access.
