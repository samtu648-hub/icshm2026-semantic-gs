# SEM-051-OOF — three spatial OOF SegFormer-B2 checkpoints

Status: completed; all 219 training-side views have exactly one OOF prediction.

The three checkpoints share one architecture, loss, seed, budget, crop, and inference implementation. They differ only in the frozen spatial fold. Each of the 219 SEM-001 training-side labelled views must receive exactly one prediction from a checkpoint that did not train on that view or its local guard band.

## Real results

All three runs used code freeze `59937945ced2f3adb22ef57f832804cced86791d`, split SHA-256 `ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`, seed 20260805, 3,000 iterations, BF16, and the same weighted-CE/crop/inference implementation. Fixed validation and unlabeled RGB remained unopened.

| Fold | Train / guard / OOF | mIoU | stay_cable P / R / IoU | Train / OOF inference | Peak MiB | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 114 / 10 / 95 | 0.836199 | 0.877238 / 0.975866 / 0.858611 | 297.57 s / 75.96 s | 3221 | `1d5312de9f69a97b1caf0c7e6e52afa7570853dc5914cbe990977f58a63de99a` |
| 1 | 128 / 7 / 84 | 0.851310 | 0.917897 / 0.970161 / 0.892695 | 295.25 s / 67.57 s | 3226 | `31fb4706428ab37b8e6c59fbb8586a6b6d6a7b686c0751bc35a1f796a72d08e0` |
| 2 | 177 / 2 / 40 | 0.842864 | 0.864121 / 0.906693 / 0.793554 | 295.54 s / 32.57 s | 3221 | `f6b7d2e36a49462c5f2d0963b756a11411d56b27e4b1db21b895f00428afe1e1` |

Total training time was 888.35 s (0.2468 GPU h); OOF inference was 176.10 s. Result SHA-256 values are `5f31d07a6ee55d8a650bc15fda4e45d41abdb7281078da8f52d13e7b3ff1dd1d`, `699ff8906555c02142e024e721fdb34bf8ca120efd3152f6d9dcbb829d63769e`, and `efbc9d937f01546c7ed3aa3220f2d3301d7a8df7b19330358cc898dc247685e0`.

Conclusion: the OOF engineering and completeness gates pass. These are 2D fold results, not final 3D model metrics. Qualification is decided only by SEM-052.
