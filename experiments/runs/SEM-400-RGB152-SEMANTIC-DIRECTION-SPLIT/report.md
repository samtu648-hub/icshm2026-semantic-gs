# SEM-400-RGB152-SEMANTIC-DIRECTION-SPLIT

- Status: completed; boundary gate failed; direction-placement candidate closed without second seed.
- Parent: RGB152 with the SEM398/399 semantic-conflict mechanism diagnosis.
- Sole placement policy variable: deterministic child displacement line, pseudorandom control versus multi-view semantic XYZ-gradient principal line.
- Shared: 4,096 parent IDs, symmetric ellipsoid-surface construction, child count/attributes/logits, optimizer lifecycle, schedules, and budgets.
- New refinement split: 292 RGB train, 201 semantic train, 30 camera-only confirmation, 31 spatial buffer.

## Engineering and direction evidence

Native preflight passed exact RGB replay, probability normalization, finite Class ID 2 gradients, semantic-to-geometry blocking, and point-capacity checks. The common stage completed 2,010 semantic updates, 201 semantic-conflict views, 292 RGB evidence views, and then a separate 201-view XYZ-gradient direction pass without updating geometry.

There were 6,269 jointly eligible parents. The sealed semantic-conflict list contained 4,096 parents. All selected parents received nonzero direction evidence; the median principal eigengap fraction was `0.871515`, and the median unsigned angle between candidate and control lines was `59.678°`. All three preregistered direction gates passed.

Both arms ended at 793,377 Gaussians and used identical 876-view restoration traces. Median parent-to-child radius was `0.0100783` scene units for the pseudorandom control direction and `0.00782585` for the semantic direction. Although both arms use the same one-sigma ellipsoid-intersection rule, anisotropic parent scales make Euclidean radius direction-dependent. Therefore the tested variable is the complete displacement-line policy, not a pure angle change at a fixed Euclidean radius.

## Fixed new refinement confirmation

| Metric | Deterministic control | Semantic direction | Candidate − control |
|---|---:|---:|---:|
| Five-class mIoU | 74.165536% | 74.183963% | +0.018427 pp |
| Foreground mIoU | 70.009145% | 70.029097% | +0.019952 pp |
| `stay_cable` IoU | 10.758857% | 10.761165% | +0.002308 pp |
| `stay_cable` precision | 60.154444% | 60.521120% | +0.366676 pp |
| `stay_cable` recall | 11.584425% | 11.573594% | -0.010831 pp |
| 2px cable boundary F1 | 35.007646% | 34.890695% | **-0.116951 pp** |
| RGB PSNR | 23.466659 dB | 23.458606 dB | -0.008054 dB |
| RGB SSIM | 0.91068265 | 0.91068202 | -0.00000063 |

Five-class mIoU, cable IoU/precision, PSNR, and SSIM protection gates passed. The required boundary gain of at least `+0.5 pp` failed and was negative. Region mIoU changed by `+0.080026 pp`, `+0.028724 pp`, and `-0.345135 pp`; region boundary F1 changed by `+0.154099 pp`, `-0.135617 pp`, and `-0.287788 pp`.

Training-fit differences were also nearly neutral: five-class mIoU `+0.013740 pp`, cable IoU `+0.036299 pp`, and boundary F1 `+0.019627 pp`. The direction signal is numerically strong and distinct, but moving two otherwise identical-responsibility children along that line does not create useful boundary separation.

The frozen decision is `closed_not_promoted`. No second seed, fixed-distance direction retry, direction threshold/offset/restoration scan, or automatic semantic-responsibility initialization is authorized. This result supports a narrower diagnosis: placement direction alone is insufficient while both children inherit the same semantic state.

Endpoint SHA256 values are control `a9f71eb09cd0efa0823fed4e207307717572d406e823a2d87151f04aaa8f06e2` and candidate `9002ab4aa838b0c7212c8431426000032ca027d060d3d0c25c418315d754053e`. The complete audit is `outputs/quality/SEM-400/evaluate-r1/result.json`.
