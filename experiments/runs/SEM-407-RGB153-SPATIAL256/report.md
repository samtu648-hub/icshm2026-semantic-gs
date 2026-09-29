# SEM-407-RGB153-SPATIAL256

Registered on 2026-09-20 as the semantic half of the RGB152/153 matched spatial
comparison. It reuses the SEM396 Mask2Former initialization, seed, 6,000-step
sampling trace, labels, optimizer, schedule, endpoint, and inference. The only
input change is RGB153 original-Standard renders in place of RGB152 AbsGS
renders.

Status: completed on 2026-09-20.

Preparation verified the sealed RGB153 result and copied only its 353 training
renders. The 6,000 view/crop/flip samples match SEM396 exactly. The fresh ADE
initial model state SHA256 is also exactly equal to SEM396:
`90b1b339261659d51baac01d6a6afc4fadc819589d5cabd5d72447088bd8e16f`.

Training completed 6,000 successful updates in 1,657.02 seconds with seven AMP
retries and peak allocated/reserved memory 8,084.52/8,564 MiB. The sealed
checkpoint SHA256 is
`f6f66dfa54afa37ac6c04e465c63e7c95fc355e4cecd1713c88a5701bc68e7c1`
and remained unchanged after confirmation access.

## Fixed evaluation

| Role | Five-class mIoU | Foreground mIoU | stay_cable IoU | 2px boundary F1 |
|---|---:|---:|---:|---:|
| Training fit, 256 labels | 95.336137% | 94.318646% | 95.361245% | 87.626938% |
| Spatial confirmation, 30 labels | 71.980289% | 65.573421% | 73.667201% | 61.965084% |
| Region 1, 152--161 | 45.958714% | 34.013837% | 26.327263% | 30.916832% |
| Region 2, 255--264 | 90.881891% | 88.724010% | 92.971131% | 73.207893% |
| Region 3, 181--190 | 91.446060% | 89.365197% | 92.791097% | 92.105426% |

The train fit is nearly identical to SEM396, and regions 2/3 remain nearly
identical. Region 1 collapses together with RGB153: relative to SEM396, region-1
five-class mIoU, `stay_cable` IoU, and boundary F1 are lower by
39.044441/50.416176/46.133311 percentage points. This localized agreement
between RGB and semantic failure supports a reconstruction-driven explanation,
not an overall Mask2Former training drift.

The 30-view result remains historically exposed spatial development evidence,
not blind testing. The single-seed comparison provisionally favors AbsGS but
requires replication before final promotion.

Training result SHA256:
`3608c3ae66d31e5caaae11e647d7f0ef5890e8bd86de460e55a0481586dc8d27`.
Inference result SHA256:
`7a5226b89bb0bb89c6cd29d850b9ca34637342dc49d0eace8fdc5901028d24c9`.
