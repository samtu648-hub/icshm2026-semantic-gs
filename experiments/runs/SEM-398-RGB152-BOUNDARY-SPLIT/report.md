# SEM-398-RGB152-BOUNDARY-SPLIT

- Status: completed positive first-seed mechanism result; SEM-399 second-seed target gates failed; current recipe closed with no system promotion.
- Parent: RGB152 Standard+AbsGS spatially isolated reconstruction, 789,281 Gaussians.
- Common stage: 256 training labels, 2,560 frozen-geometry semantic updates, then one sealed semantic-evidence pass and one sealed 353-view RGB-evidence pass.
- Sole paired variable: rank the same eligible parents by RGB absolute screen gradient (control) or multi-view `stay_cable` semantic-gradient conflict (candidate).
- Both arms: 4,096 native binary splits, 793,377 final Gaussians, identical split RNG seed, identical 1,059-view restoration trace, semantic-to-geometry gradient blocked.

## Engineering and mechanism evidence

The native preflight exactly reproduced RGB152 PNGs on a training and a confirmation view. Five-class probability sums had maximum error `1.43e-6`; the semantic loss produced a finite Class ID 2 gradient while all geometry gradients remained blocked. There were 151,103 large Gaussians before applying the photometric gate.

The shared gate retained 6,067 parents. Control and candidate selected 4,096 each, with 2,740 in common and 1,356 different (33.105% of the budget), passing the preregistered minimum difference of 512. Candidate-selected semantic-conflict quartiles were `[0.299990, 0.419547, 0.601380]`, versus `[0.081959, 0.309183, 0.510595]` for control; candidate RGB-score median was lower (`0.000510083` versus `0.000595596`), confirming a real ranking trade rather than a duplicate RGB ranking. The two restoration sampling files have the same SHA256 `c742c1d64ceae7e10c789f640d51c27a733ed6507ebe9afbfb1358d6cde8aafc`.

The initial sandbox-only preflight could not see the GPU and is preserved as `preflight-r1`. Native-host `preflight-r2` completed. A supplemental evaluation invocation that accidentally coupled output and source attempt names stopped before loading endpoints and is preserved as `evaluate-r2`; `evaluate-r3 --source-attempt r1` then completed without changing either sealed endpoint.

## Fixed 30-view spatial confirmation

| Endpoint | Five-class mIoU | Foreground mIoU | `stay_cable` IoU | Precision | Recall | 2px cable boundary F1 | RGB PSNR | RGB SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Common unsplit | 51.615982% | 40.831205% | 12.254528% | 51.961830% | 13.820250% | 19.898871% | 19.384419 | 0.831530 |
| RGB-ranked control | 50.178673% | 39.087711% | 11.940666% | 48.088887% | 13.707532% | 18.328082% | 19.352227 | 0.831161 |
| Semantic-conflict candidate | 52.992731% | 42.551058% | 12.452798% | 49.594311% | 14.257274% | 20.041279% | 19.458840 | 0.832702 |
| Candidate − control | **+2.814057 pp** | **+3.463348 pp** | **+0.512132 pp** | **+1.505423 pp** | **+0.549743 pp** | **+1.713197 pp** | **+0.106613 dB** | **+0.001541** |

All six preregistered gates passed: boundary F1 gained at least 0.5 pp; five-class mIoU, cable IoU and cable precision did not fall; PSNR/SSIM stayed inside their protection limits. Region mIoU candidate versus control was `+4.877994 pp`, `+0.014375 pp`, and `+1.706077 pp` for regions 1/2/3. Region cable-boundary F1 changed by `+4.640865 pp`, `+0.372510 pp`, and `+2.140267 pp`.

## Training-fit counterevidence and interpretation

On the 256 training labels, candidate versus control five-class mIoU changed by `-0.064451 pp`, cable IoU by `-0.311365 pp`, and cable boundary F1 by `-0.273341 pp`; training RGB PSNR/SSIM changed only `+0.000769 dB/+0.000041`. The confirmation gain therefore is not explained by a larger training fit and may reflect useful structural regularization, but one seed and a historically exposed split cannot establish robustness.

The absolute semantic field is weak: even the candidate reaches only 52.99% confirmation mIoU, far below SEM396/397 Mask2Former. Thus SEM-398 is evidence that semantic conflict is a useful **split-selection signal**, not a production semantic system and not proof of more accurate 3D geometry. It does not replace RGB152, SEM396/397, RGB140+SEM378, or the default RGB130+SEM371.

The original frozen decision was `positive_mechanism_candidate` and authorized exactly one matched second seed. SEM-399 executed that replication: overall confirmation mIoU and RGB remained positive, but the required boundary gain, cable-IoU retention, and cable-precision retention gates failed. The combined decision is therefore `replication_failed_closed`. No third seed, budget/threshold/prior/restoration-step scan, stronger-teacher experiment, or all-data migration is authorized from this result.

Sealed endpoint SHA256 values are control `f07f34bf6268e142af31dd7d7d4a872be18be48f5409376bda9f4f20bbb47420` and candidate `e5a6576e49449eab8028978e55a747818e73be868d4f7884d357a5d2ca9205c0`. The complete unified audit is `outputs/quality/SEM-398/evaluate-r3/result.json`.
