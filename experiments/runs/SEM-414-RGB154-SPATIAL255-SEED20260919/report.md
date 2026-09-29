# SEM-414-RGB154-SPATIAL255-SEED20260919

Completed on 2026-09-21; status: completed paired second-seed validation.

Parent: SEM412/RGB154. Seed 20260919; manifest `contest_dataset_v1.json`;
split `spatial_confirmation_v2` (255 semantic training / 30 confirmation).
Git base `49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus the separately recorded
working-tree source hashes. Config: `config.yaml`.

Output: `outputs/quality/SEM-414/r1`. Execution commands, times and exit statuses
are retained in `outputs/quality/SEM-414-415/r1/*-execution.json`.
Protocol: [paired seed replication](../../../docs/experiments/sem414_sem415_spatial_v2_seed_protocol.md).

## Preflight and launch

Both discarded eight-update preflights passed. All 650 initial tensors match
the sealed seed-20260919 reference and the paired arm; only the two expected
classifier tensors differ from the old seed. AbsGS preflight reserved peak:
8,588 MiB. Confirmation RGB/label access was zero. Fresh 001/120/300 RGB outputs
exactly reproduce RGB154 before training.

Execution is retained by user service `icshm-sem414-sem415-v2-seed.service`.
Stage commands, exit statuses and logs are recorded under
`outputs/quality/SEM-414-415/r1`. Both arms ran from fresh initialization in the
fixed serial queue; the pair binding/source snapshot was sealed before launch.

## Training completed

Training completed 6,000 successful updates in 1,764.948 seconds, with five
standard AMP overflow retries. The initial state SHA256 is
`a9f66a9ada969f1bde050ac3cc184ae332d72877b25caeda56b7e9436de9f7d2`;
the actual 6,000 samples match the registered plan and the seed-aware contract
passed before update one. Peak reserved memory: 8,588 MiB. Frozen training RGB
hashes are exact; confirmation label access during fitting is empty. Stage exit
status is 0. Both full fits were sealed and their actual samples compared before
the first new confirmation inference started.

## Fixed results (%)

| Role | Foreground mIoU | Five-class mIoU | stay_cable IoU | 2px boundary F1 |
|---|---:|---:|---:|---:|
| Training fit | 94.486702 | 95.475824 | 95.684445 | 88.018995 |
| Confirmation | 87.404296 | 89.767499 | 81.774993 | 74.173220 |
| Region 1 | 75.927832 | 80.657069 | 58.452860 | 73.391070 |
| Region 2 | 91.645449 | 93.259032 | 93.477449 | 91.297811 |
| Region 3 | 85.885176 | 88.373915 | 79.602430 | 63.324699 |

Against SEM415, confirmation foreground/five-class mIoU, cable IoU and boundary
F1 improve by 0.246371/0.205683/0.940255/1.466619 pp. Cable precision/recall are
86.867098%/93.311098% (+1.055615/+0.005470 pp); cable FP/FN decrease by
17,139/71 pixels. The gain is primarily false-positive control, not a large
recall increase. Tower IoU is 0.173044 pp lower.

Regional foreground deltas are -0.377754/-0.115981/+1.010405 pp. Region-1/2
cable IoU also declines, although all three regional boundary F1 values improve.
Thus the pre-registered overall directions pass without implying uniform
regional or class dominance. See [QUAL013](../QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md)
for the two-seed conclusion.

## Verification, runtime and artifacts

- All 379 views inferred from the saved 6000-step checkpoint after the pair seal.
- Inference stage: 221.018 seconds including process startup; median model
  inference 482.990 ms/view. Peak allocated/reserved: 1,478.076/1,968 MiB.
- Full training peak allocated/reserved: 8,097.570/8,588 MiB.
- 285 per-view confusion matrices independently recounted from PNGs; pooled and
  regional metrics and independently rebuilt four-neighbor boundary counts exact.
- All 379 prediction hashes and all 285 scored label identities verified.
- Frozen RGB model/cache hashes exact; native 001/120/300 rendered before and
  after the pair with maximum pixel difference 0. No RGB training occurred.
- No failed execution or empty-mass inference retry occurred. The five AMP
  retries are bounded numerical overflow recovery, not additional successful updates.
- New artifacts approximately 2.0 GiB, including both fixed checkpoints and inputs.
- Checkpoint SHA256: `2753152ac75620d7597f91ea876718d28065cc0a0959ac9d998ac86e0df9bb37`.
- Training result SHA256: `82b89de95f75f20233beb414ca8d3ac27b7b0da864fc4de0478b1f2323b9b2e9`.
- Inference result SHA256: `d87214ed6af2e45266be4bb66ede58f47b8d13ccdf4e9f3368e0439a2042e6af`.

Scope remains historically exposed, PINHOLE-grid, same-scene development
confirmation with RGB seed 0. The existing full-data RGB140/SEM378 system and
Standard fallback are unchanged; rollback is to use those retained artifacts.
