# SEM-396-RGB152-SPATIAL256

- Status: completed on 2026-09-20.
- Parent recipe: SEM378 ADE-pretrained Swin-L Mask2Former, unchanged except for the new data roles and RGB152 input.
- Split: `spatial_confirmation_v1`, 256 semantic train / 30 confirmation / 17 excluded buffer.
- Seed: 20260805; fixed endpoint: 6,000 native 768-crop successful updates.
- RGB parent: sealed RGB152 `full-r2` checkpoint and fixed renders.
- Training: 1,532.54 seconds, six AMP retries, peak allocated/reserved 8,084.52/8,564 MiB.
- The sampling plan matched all 6,000 updates, the RGB renders remained byte-frozen, and no confirmation label was read before the step-6000 checkpoint was sealed.
- Step-6000 checkpoint SHA256 before and after confirmation access: `eab4aa600104c0c4f1341c89306d41750c4552d33bb607c6c46390735ce6e8eb`.

## Fixed evaluation

| Role | Five-class mIoU | Foreground mIoU | 2px boundary F1 |
|---|---:|---:|---:|
| Training fit, 256 labels | 95.375428% | 94.367004% | 88.187853% |
| Spatial confirmation, 30 labels | 89.204606% | 86.667377% | 79.519325% |
| Region 1, 152--161 | 85.003155% | 81.554443% | 77.050142% |
| Region 2, 255--264 | 90.909214% | 88.762301% | 73.171793% |
| Region 3, 181--190 | 91.425195% | 89.340557% | 92.736153% |

Confirmation IoU by class is background 99.353523%, deck 89.964324%, `stay_cable` 89.355622%, tower 88.602339%, and foundation 78.747223%. `stay_cable` confirmation precision/recall are 97.273043%/91.651491%, with 34,511 false positives and 112,135 false negatives. Region 1 is the clearest failure mode: `stay_cable` IoU is 76.743439% and recall is 79.902389%; foundation IoU is 74.666695%.

The training-to-confirmation gaps are 6.170822 percentage points for five-class mIoU, 7.699627 points for foreground mIoU, and 8.668528 points for 2px boundary F1. The result confirms that the all-data training-fit scores materially overstate performance on spatially separated views. It also supplies a justified target for a future single-variable boundary-loss experiment, especially thin-structure recall and region-1 boundaries. The poor RGB152 region-1 reconstruction is a competing bottleneck, so a semantic boundary loss cannot be assumed to recover the entire gap.

This is a historically exposed spatial development confirmation, not a blind test. It does not replace the default RGB130+SEM371 or all-data candidate RGB140+SEM378 automatically, and it cannot be compared directly with their all-data training-fit scores as if the data roles were identical. No boundary-loss candidate has been trained under this ID.

Prepare, training, and inference result SHA256 values are `43071033f8b7d804c6cc2cf5a2b89372910e0a757a023626711fadcdd378f7e9`, `f63f71f9a937abe47eeba374f88d7b34f5cf29309f3d06322c3323717512b90f`, and `7cf83ea33c54a3e2a6d3ba83c432b31ea3b5ba7e3bf2a221075b8b94c968add4`.
