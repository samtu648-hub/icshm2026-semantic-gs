# RGB-153-STANDARD-SIGNED-SPATIAL30

Registered on 2026-09-20 as the matched original-Standard control for the
existing RGB152 AbsGS spatial confirmation. The split, scene, 78,045-point
initialization, seed 0, 30,000-step budget, cameras, and evaluation are fixed.
The only reconstruction-method variable is original signed-gradient Standard
densification versus the parent AbsGS absolute-gradient split.

Status: completed on 2026-09-20.

The preflight verified the exact `spatial_confirmation_v1` split hash, 353/30/17
roles, 78,045-point initialization SHA256, and the same official Standard 3DGS
source used by RGB130. The recorded command contains the signed threshold
`0.0002` and no AbsGS absolute-gradient threshold.

Training completed 30,000 updates in 2,010.37 native-worker seconds
(2,013.89 seconds including the runner), with peak allocated/reserved memory
1,683.08/5,000 MiB. The final point cloud contains 655,889 Gaussians. The
checkpoint and PLY SHA256 values are
`b20f9d78e2afa7707a39701bbcab910feb103bc4235fb369a3b207ead72b0a1b`
and `11ea2964673dc545d9c6923b15d5e470fd30fdd18c701a292f3118c296de53da`.

## Fixed evaluation

| Role | PSNR dB | SSIM | LPIPS-Alex |
|---|---:|---:|---:|
| Training fit, 353 views | 23.202296 | 0.894330 | 0.249931 |
| Spatial confirmation, 30 views | 18.139885 | 0.816594 | 0.386085 |
| Region 1, 152--161 | 13.480117 | 0.722473 | 0.579679 |
| Region 2, 255--264 | 20.638726 | 0.840019 | 0.320610 |
| Region 3, 181--190 | 20.300811 | 0.887291 | 0.257966 |

Relative to the matched RGB152 AbsGS chain, confirmation PSNR/SSIM/LPIPS are
lower by 1.244026 dB / 0.014148 / 0.035200. The difference is strongly
localized: AbsGS improves region 1 by 3.924884 dB, while region 2 is close and
original Standard is 0.296774 dB better in region 3. RGB153 uses 16.90% fewer
Gaussians than RGB152 and renders about 25.98% faster, but its region-1 render
loses most bridge structure and cannot support a competitive semantic input.

The first evaluation entrypoint used system Python and stopped before loading
Torch or opening confirmation images. Its result is preserved under
`evaluation-r1-failed-system-python`. The recovery used the frozen CUDA/Python
environment and changed no data, checkpoint, metric, or evaluation setting.

Training result SHA256:
`d6bdeb27d4f157df2e230aa8b0c5088d8621ba5766c291f28e83340612c876e6`.
Completed evaluation SHA256:
`d32422c643eb2e0d9990772fce715c3c53a50270beebfb925cb6e39a5f6e1143`.
