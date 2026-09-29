# QUAL-010-MATCHED-SPATIAL-BACKBONE

Registered as a read-only comparison after both matched chains were sealed.
No training, inference, threshold search, or checkpoint selection is performed
under this ID.

Status: completed on 2026-09-20.

## Matched result

| Spatial confirmation metric | AbsGS | Standard | AbsGS minus Standard |
|---|---:|---:|---:|
| RGB PSNR | 19.383911 dB | 18.139885 dB | +1.244026 dB |
| RGB SSIM | 0.830742 | 0.816594 | +0.014148 |
| RGB LPIPS | 0.350885 | 0.386085 | -0.035200 |
| Five-class mIoU | 89.204606% | 71.980289% | +17.224317 pp |
| Foreground mIoU | 86.667377% | 65.573421% | +21.093956 pp |
| `stay_cable` IoU | 89.355622% | 73.667201% | +15.688421 pp |
| 2px boundary F1 | 79.519325% | 61.965084% | +17.554241 pp |

AbsGS is better on 21/30 paired views by PSNR, 20/30 by five-class mIoU,
and 21/30 by `stay_cable` IoU. The aggregate advantage is dominated by region
1: AbsGS improves PSNR by 3.924884 dB, five-class mIoU by 39.044441 pp, cable
IoU by 50.416176 pp, and boundary F1 by 46.133311 pp. Regions 2 and 3 are
near-neutral semantically, with all five-class differences within 0.028 pp;
this localization argues against a global training or evaluation mismatch.

## Cost

Relative to Standard in this matched split, AbsGS uses 20.34% more Gaussians
and PLY bytes, 33.33% more runner training time, and 35.09% more median render
time. The cost is material, but the difficult-region protection is also
material and would dominate a same-scene new-view score if similar views occur
in the hidden camera set.

## Decision

The read-only audit records `completed_provisional_absgs_advantage`. It
provisionally favors the RGB140/RGB152 AbsGS recipe over original Standard for
the final system, but does not yet promote it: the large gain is one historical
development split and one seed. The next evidence is a paired second semantic
seed, followed, if needed, by the already camera-only selected disjoint spatial
folds.

Machine result SHA256:
`55dbb4614bdea9832a4aa0a0c1103ffbdba23c2405cc1e8333cae0c98c54e2dd`.
