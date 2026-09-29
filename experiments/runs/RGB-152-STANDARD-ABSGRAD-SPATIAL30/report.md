# RGB-152-STANDARD-ABSGRAD-SPATIAL30

- Status: completed on 2026-09-20.
- Parent recipe: RGB140 Standard 3DGS + AbsGS, unchanged except for the new data roles.
- Split: `spatial_confirmation_v1`, 353 RGB train / 30 confirmation / 17 excluded buffer.
- Seed: 0; fixed endpoint: 30,000.
- Prepared scene: 383 cameras and 78,045 training-only DLT points. Initial PLY SHA256 `b7bddeda3759a11e833bd3d60ee554b8e005688ab14f9d41b99d99e2ac59521a`.
- Final checkpoint SHA256: `d63ac062b96faf80f883255dfd3ec244446257846c0a10cde7205caefef67be2`.
- Final PLY: 789,281 Gaussians; SHA256 `77ab7c8059e453eebd05f05354dcdc98791fcb616cbbcccb6127b6d6aa01a57a`.
- Training time: 2,677.72 seconds in the native worker; peak allocated/reserved GPU memory 1,950.61/6,154 MiB.
- Confirmation and buffer observations and point colours remained excluded from initialization and training.

## Fixed evaluation

| Role | Views | PSNR dB | SSIM | LPIPS-Alex |
|---|---:|---:|---:|---:|
| Training fit | 353 | 23.296830 | 0.897137 | 0.242863 |
| Spatial confirmation | 30 | 19.383911 | 0.830742 | 0.350885 |
| Region 1, 152--161 | 10 | 17.405002 | 0.770710 | 0.464581 |
| Region 2, 255--264 | 10 | 20.742694 | 0.838378 | 0.321818 |
| Region 3, 181--190 | 10 | 20.004037 | 0.883138 | 0.266257 |

The 30-view confirmation PSNR is 3.912919 dB below the training fit, with the largest degradation in region 1. This is credible evidence of a spatial reconstruction gap for the unchanged recipe on this development split. It is not a direct matched comparison with RGB140's all-data training-fit score and is not a blind-test claim.

## Preserved recovery history

`preflight-r1` stopped before model loading because the invocation duplicated the `r1` component of the artifact root. `preflight-r2` then passed, but `full-r1` stopped before update 1 because the shared worker still enforced its legacy 370/30 counts. The worker recovery only added explicit registered partition/count arguments while retaining legacy defaults. Because that changed a preflight-bound source hash, `preflight-r3` revalidated the final code and `full-r2` restarted from initialization. All failed attempts are preserved; neither produced a checkpoint or quality result.

Final result SHA256: `95730c3b460d9d9d52b4ad72850a5b6e041077e41de7e63fa7c563552e96c314`. Evaluation result SHA256: `06163daf5c7629e6b1621d870cff38b28d03b9b2cbac92b323d15c4d4aee6fe9`.
