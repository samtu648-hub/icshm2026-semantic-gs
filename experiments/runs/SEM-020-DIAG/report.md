# SEM-020-DIAG — training-side contribution distribution

- Status: completed training-side-only diagnostic; parent: `SEM-001`.
- Output: Git-ignored `$R1_OUTPUT_ROOT/r2/sem_002/diagnostic/`; merged `result.json` SHA-256 `d72fa033fc0116c0e047bba9329a42962c6b459db0d052ee926115d557582e87`.
- Inputs: all and only the 219 `semantic_train_labeled` cameras.  No validation RGB/labels, guard-band, unlabeled RGB, or hidden test content was read.

The interactive host terminates foreground GPU commands near one minute, so the same read-only diagnostic was completed as nine contiguous, deterministic training-camera chunks and merged by `scripts/summarize_sem002_diagnostic.py`.  The two thresholds were fixed once before any semantic quality evaluation:

| Gate | Frozen threshold | Contribution mass retained | Contributors retained |
|---|---:|---:|---:|
| SEM-021 cumulative prefix | 0.95 | 95.6325% | 65.9677% |
| SEM-022 relative maximum | 0.05 | 91.2577% | 44.1695% |

These values characterize frozen RGB visibility only; they neither select by nor claim validation quality.
