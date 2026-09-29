# SEM-415-RGB155-SPATIAL255-SEED20260919

Completed on 2026-09-21; status: completed paired second-seed Standard control.

Parent: SEM413/RGB155, paired with SEM414. Seed 20260919; manifest
`contest_dataset_v1.json`; split `spatial_confirmation_v2` (255 / 30).
Git base `49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus recorded source hashes.
Config: `config.yaml`; output: `outputs/quality/SEM-415/r1`.

The same-seed full initial state and every sample must match SEM414. Both
endpoints must be sealed before new confirmation evaluation.
Protocol: [paired seed replication](../../../docs/experiments/sem414_sem415_spatial_v2_seed_protocol.md).

Both eight-update preflights passed with exactly paired 650-tensor initial
states and samples. Standard preflight reserved peak: 8,584 MiB. Fresh RGB155
001/120/300 renders match their frozen PNGs exactly. No new confirmation RGB
or labels were accessed during preflight. Full training subsequently started
fresh after SEM414 exited and completed all 6,000 successful updates.

## Training and fixed results

Training took 1,776.611 seconds, with five bounded AMP overflow retries. Peak
allocated/reserved memory was 8,097.570/8,588 MiB. All 650 initial tensors and
all 6,000 actual samples match SEM414; the common initial state SHA256 is
`a9f66a9ada969f1bde050ac3cc184ae332d72877b25caeda56b7e9436de9f7d2`.
Confirmation labels were not accessed during fitting. Both fits were sealed
before either new confirmation inference began.

| Role | Foreground mIoU (%) | Five-class mIoU (%) | stay_cable IoU (%) | 2px boundary F1 (%) |
|---|---:|---:|---:|---:|
| Training fit | 94.518335 | 95.501525 | 95.691545 | 88.306752 |
| Confirmation | 87.157925 | 89.561816 | 80.834737 | 72.706601 |
| Region 1 | 76.305586 | 80.961721 | 59.278683 | 72.586022 |
| Region 2 | 91.761430 | 93.352175 | 93.536262 | 90.557236 |
| Region 3 | 84.874771 | 87.535877 | 77.949304 | 60.862470 |

Confirmation cable precision/recall: 85.811483%/93.305627%. SEM414 is better
overall but not on every regional or class metric. See
[QUAL013](../QUAL-013-SPATIAL-V2-SEED-REPLICATION/report.md) for paired deltas
and the first-seed comparison; no checkpoint or seed selection is performed.

## Verification and artifacts

- Training and all subsequent stage exit statuses are 0, with full commands and
  logs in `outputs/quality/SEM-414-415/r1`.
- All 379 views inferred; process-inclusive inference stage 220.259 seconds,
  median 482.861 ms/view, allocated/reserved peak 1,478.076/1,968 MiB.
- Independent recount verified 285 confusion matrices, pooled/region summaries,
  boundary counts, 285 label hashes and 379 prediction hashes.
- RGB155 model/cache hashes and three training-camera RGB renders remained exact
  before/after; no empty-mass inference retries or failed attempts occurred.
- New artifacts approximately 1.9 GiB; both 3000/6000 checkpoints retained.
- Checkpoint SHA256: `b65a6d494d872392784041e59258ac1f45a0ed0c57c752fe59edca382ed19480`.
- Training result SHA256: `9fae3b13057c3329f06cde8312d6d8c309873f6212680afbebff71de2b2e17cd`.
- Inference result SHA256: `0f88a94e8547edb7d1705ac05f22b46af6bfba0f92c3ae817cf8fed5d0cc728b`.

This is a development control, not a new production model. Frozen full-data
systems remain available without migration; no further training is queued.
