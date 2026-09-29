# SEM-397-RGB152-SPATIAL256-BOUNDARY

- Status: completed and closed without promotion on 2026-09-20.
- Parent/control: SEM396 on the same RGB152, split, seed, initialization, 6,000-step sample trace, optimizer, schedule, and inference path.
- Sole method variable: fixed weight-0.05 `stay_cable` soft morphological boundary-Dice term.
- Training: 1,619.40 seconds, six AMP retries, peak allocated/reserved 8,087.71/8,564 MiB.
- Step-6000 checkpoint SHA256: `39bc86983be0f6bdae37eefe1d753fb431365202a2e163a06b42fe8c0c93c3de` before and after confirmation access.

The fresh initialized state matched SEM396 exactly (`90b1b339261659d51baac01d6a6afc4fadc819589d5cabd5d72447088bd8e16f`). All 6,000 sample names, crop positions, and flips matched the frozen plan. RGB renders remained byte-frozen, and no confirmation label was accessed before the checkpoint was sealed.

## Fixed evaluation

| Role | SEM396 five-class mIoU | SEM397 five-class mIoU | Delta | SEM396 boundary F1 | SEM397 boundary F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Training fit | 95.375428% | 95.382440% | +0.007013 pp | 88.187853% | 88.199732% | +0.011879 pp |
| Spatial confirmation | 89.204606% | 89.657826% | +0.453220 pp | 79.519325% | 79.434665% | -0.084660 pp |
| Region 1 | 85.003155% | 86.040523% | +1.037368 pp | 77.050142% | 76.204718% | -0.845425 pp |
| Region 2 | 90.909214% | 90.722167% | -0.187047 pp | 73.171793% | 73.685485% | +0.513693 pp |
| Region 3 | 91.425195% | 91.305865% | -0.119330 pp | 92.736153% | 93.030787% | +0.294634 pp |

Confirmation foreground mIoU improved from 86.667377% to 87.224506% (+0.557129 pp). The gain came mainly from foundation IoU (+1.657839 pp), deck (+0.935171 pp), and region 1 overall performance. It did not deliver the registered thin-structure target: `stay_cable` IoU fell from 89.355622% to 88.866687% (-0.488935 pp), precision fell from 97.273043% to 96.693823% (-0.579221 pp), and recall changed from 91.651491% to 91.651566% (+0.000074 pp, one additional true-positive pixel). False positives increased from 34,511 to 42,092.

## Frozen gate decision

The candidate passed overall confirmation mIoU, the literal strict recall comparison, and region-retention gates. It failed the required +0.5 pp boundary-F1 gate, `stay_cable` precision retention, and per-class IoU retention. Because every gate was required, the fixed decision is `closed_not_promoted`.

The result is informative: a small dense boundary term altered the general semantic solution and improved overall mIoU, but the chosen soft morphological formulation did not optimize the evaluated 2-pixel cable boundary and traded cable precision/IoU for other classes. No second seed, all-data migration, weight scan, threshold repair, or checkpoint selection is authorized from this result. RGB130+SEM371 remains the default and RGB140+SEM378 remains the all-data candidate.

Prepare, training, inference, and comparison SHA256 values are `f737e61bc4ff1dea44932bd27cee47512272dee15dd579ea1f26fd93ac840ba8`, `a9381f7e26f214fac052b4e41f26cc68dd16605c708e7a8d10f2a7c70bebceb0`, `1c31f712dbe7e217797c382b5be2af47ba570afde869bb5b7dc769881de93430`, and `3756426750190fab75bbf935eeadd83df87d9490d73eca377eac6b79e1376dfd`.
