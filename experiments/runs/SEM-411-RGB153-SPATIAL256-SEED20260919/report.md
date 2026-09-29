# SEM-411-RGB153-SPATIAL256-SEED20260919

Registered on 2026-09-20 as the corrected RGB153 arm of the second paired
semantic-seed replication. It preserves SEM409's scientific variables and
uses the same sealed seed-20260919 initialization contract as SEM410.

Status: completed on 2026-09-20.

Its 6,000 samples match SEM410 exactly. Before the first update, all 650 initial
state tensors and the pretrained-load report matched SEM410 and the sealed
SEM392 seed-20260919 reference; the common state SHA256 is
`a9f66a9ada969f1bde050ac3cc184ae332d72877b25caeda56b7e9436de9f7d2`.
Training completed 6,000 successful updates in 1,635.95 seconds with six AMP
retries and 8,084.52/8,564 MiB peak allocated/reserved memory.

## Fixed evaluation

| Role | Five-class mIoU | Foreground mIoU | stay_cable IoU | 2px boundary F1 |
|---|---:|---:|---:|---:|
| Training fit | 95.424967% | 94.429882% | 95.288162% | 87.240880% |
| Spatial confirmation | 73.740869% | 67.741440% | 80.978347% | 68.301287% |
| Region 1 | 48.800747% | 37.450775% | 37.440752% | 37.863744% |
| Region 2 | 90.701027% | 88.517324% | 93.485861% | 76.376295% |
| Region 3 | 91.715468% | 89.701708% | 93.016017% | 92.270308% |

The near-equal training fit and regions 2/3 again contrast with a severe
region-1 failure, reproducing the first seed's reconstruction-driven spatial
pattern. The checkpoint SHA256 is
`733d0dfd5644d7b4b4f4f7c86e53dfae6d155756878dff33f3364dd411847b22`
and remained unchanged after confirmation access. Training-result SHA256:
`63c4f0615b9e78db3e285096edaea5bdfd12f2ba78b5f38c07a6ec170cc53b23`.
Inference-result SHA256:
`18b843239f58a4752d0d40ab7ef3f33577e256cd11f34ab1714a2d926dbdd3ad`.
