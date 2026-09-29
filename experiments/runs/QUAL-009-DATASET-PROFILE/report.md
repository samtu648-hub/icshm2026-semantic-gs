# QUAL-009-DATASET-PROFILE

Completed on 2026-09-20 as a read-only, paper-facing profile of the released
300 labeled views and 400 official cameras. No model training, prediction
change, threshold search, or hidden-test claim occurred under this ID.

## Dataset composition

The 300 native 1320x989 labels contain 391,644,000 evaluated pixels. The class
distribution is:

| Class | Pixel share | Views present | 8-connected components | Median component area |
|---|---:|---:|---:|---:|
| background | 85.5872% | 300/300 | 5,007 | 2 px |
| deck | 5.0340% | 300/300 | 667 | 20,252 px |
| `stay_cable` | 6.9745% | 300/300 | 1,101 | 2,620 px |
| tower | 1.8016% | 289/300 | 558 | 4,154.5 px |
| foundation | 0.6026% | 286/300 | 1,398 | 1,145 px |

The result rules out a simplistic claim that `stay_cable` is difficult merely
because it is globally the rarest class: it is the second-largest foreground
class by pixels. Its important difficulty is scale heterogeneity. Components
of at most 1,024 pixels account for 365/1,101 (33.15%) of cable components but
only 0.5800% of cable pixels. This aligns with SEM367's independent error
stratification, where the same small-component range contained a
disproportionate share of cable errors. The two results together support a
thin/distant-structure long-tail explanation rather than a global-frequency
explanation.

Within-class pixels lying no more than two pixels inside a class boundary are
7.0576% for `stay_cable`, 6.3632% for deck, 16.0604% for tower, and 22.2894%
for foundation. Boundary exposure is therefore also substantial for the two
smallest structural classes; boundary metrics should accompany mIoU rather
than being presented as a cable-only issue.

## Camera coverage

The 300 labeled views cover 92.67%, 71.90%, and 100% of the full 400-camera
PCA extent along the three axes. For the 100 unlabeled cameras, the nearest
labeled-camera distance has median 0.5420 and maximum 1.8185 in the official
COLMAP coordinate scale. Labels therefore cover most of the dominant camera
axis but noticeably less of the secondary axis. This supports the spatial
confirmation design and cautions against treating a random image split as a
complete new-view generalization test.

## Artifacts and verification

- machine result: `outputs/quality/QUAL-009/r1/result.json`, SHA256
  `1c561f78b3c4b5e0ee263a42ad9c9be944a8c616ee01da97f477d9d997f7a4f0`;
- class-share figure: `outputs/quality/QUAL-009/r1/class_pixel_share.png`;
- cable-scale figure: `outputs/quality/QUAL-009/r1/stay_cable_components.png`;
- camera-coverage figure: `outputs/quality/QUAL-009/r1/camera_coverage.png`;
- all 300 label hashes were verified against the frozen SEM370 manifest;
- a separate manifest-only aggregation reproduced all five pixel totals and
  the 391,644,000-pixel grand total exactly.

Scope: released-data descriptive analysis only. These results are not model
quality scores and make no hidden-test claim.
