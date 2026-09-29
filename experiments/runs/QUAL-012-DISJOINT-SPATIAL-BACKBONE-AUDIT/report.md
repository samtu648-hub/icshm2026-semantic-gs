# QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT

Completed on 2026-09-20 as a read-only aggregation of two zero-overlap spatial
confirmation folds. All four carried-forward v2 direction gates pass.

## Cross-fold confirmation deltas: AbsGS minus Standard

| Fold | RGB PSNR | Five-class mIoU | `stay_cable` IoU | 2px boundary F1 |
|---|---:|---:|---:|---:|
| v1 (semantic two-seed mean) | +1.244026 dB | +16.446706 pp | +11.803546 pp | +13.719429 pp |
| v2 | +0.220856 dB | +0.501025 pp | +2.917695 pp | +5.405989 pp |

The direction replicates, but its magnitude is strongly region-dependent. In
v2, RGB SSIM changes by +0.000656 while LPIPS is 0.003657 worse; semantic
`stay_cable` recall changes by -0.025657 pp. Region 1 has -1.109985 pp
five-class mIoU but +2.083633 pp cable IoU and +3.874344 pp boundary F1.
Regions 2/3 have +0.008101/+1.503351 pp five-class mIoU and
+1.710742/+8.896142 pp boundary F1.

The v2 AbsGS cost premium is 42.78% training time, 21.19% Gaussians/PLY bytes,
and 46.36% median rendering time. The project decision is therefore to select
AbsGS as the final quality-oriented reconstruction backbone while retaining
Standard as the efficiency fallback. These are historically exposed
development folds, not blind-test or universal-generalization evidence.

Machine result SHA256:
`0f96eda6845b539516d5acc32574adffa914a543e6b40b28233b523e0b0ea56b`.
