# SEM-003 — restricted background/stay_cable Dirichlet conflict regularizer

- Status: **completed; drop on the pre-registered main-fold gate**.
- Parent/control: SEM-001, SEM-003-DIAG, SEM-003-CTRL, SEM-003-PF.
- Actual run HEAD / code freeze: d84e5c93c40a6d58cca836582a0ec23bfa340cd4 / b4cdc4e3e7721b8180ff409b5da7836cd3b01adb.
- Run-start tracked-diff / porcelain SHA-256: b8cbe809dc5e1d3381440a525bc10911fb9e241a84f0893fe23295accc7442a8 / 29f7498c8a4e38899c98c0a6dd2b1c10d0ccddababb0aec0cc70f96c36090dec.

## Fair comparison

SEM-003 and CTRL used the same converted SEM-001 initialization, 219/74 spatial split, label manifest, RGB checkpoint, camera/undistortion, seed 20260731, optimizer, class weights, resolution and 10k budget. The sole candidate variable was the pre-registered concentration regularizer on 7,681 train-only eligible anchors. Ordinary five-class NLL remained active for all semantic anchors; global visibility gating remained disabled.

| Metric | SEM-001 | CTRL | SEM-003 | SEM-003 − CTRL | SEM-003 − SEM-001 |
|---|---:|---:|---:|---:|---:|
| mIoU with background | 0.616103 | 0.616209 | 0.615198 | −0.001011 | −0.000905 |
| mIoU without background | 0.563337 | 0.563305 | 0.563018 | −0.000287 | −0.000319 |
| background IoU | 0.827169 | 0.827822 | 0.823917 | −0.003906 | −0.003252 |
| deck IoU | 0.900999 | 0.901161 | 0.901597 | +0.000436 | +0.000598 |
| stay_cable IoU | 0.068759 | 0.068865 | 0.066940 | −0.001925 | −0.001819 |
| tower IoU | 0.834870 | 0.834902 | 0.834794 | −0.000108 | −0.000075 |
| foundation IoU | 0.448718 | 0.448293 | 0.448741 | +0.000448 | +0.000023 |
| stay_cable precision | 0.077965 | 0.078136 | 0.075782 | −0.002354 | −0.002182 |
| stay_cable recall | 0.368036 | 0.367260 | 0.364559 | −0.002701 | −0.003477 |
| background→stay_cable pixels | 13,166,331 | 13,106,856 | 13,452,325 | +345,469 | +285,994 |
| background→stay_cable / GT background | 15.0076% | 14.9398% | 15.3336% | +0.3938 pp | +0.3260 pp |
| worst-10% mIoU without background | 0.423440 | 0.423505 | 0.424028 | +0.000524 | +0.000588 |
| 140–146 mIoU without background | 0.601213 | 0.600672 | 0.600530 | −0.000142 | −0.000683 |

The paired per-view artifact contains all 74 candidate-minus-CTRL rows: $R1_OUTPUT_ROOT/r2/sem_003/comparison/candidate_minus_ctrl_per_view.csv, SHA-256 89788f8f72840f7febcddb9e27569cffdb3f4370e776cb6ac69ad468cd9c9f0a. Candidate mIoU improved on 28/74 views, stay_cable IoU on only 11/74, and background→stay_cable pixels decreased on 37/74; the dataset-level target error nevertheless increased.

## Engineering evidence

Training took 3,195.897 s (0.88775 GPU h), peak PyTorch allocation 849 MiB. Conflict loss fell from 190.8683 to 1.0981 without NaN/Inf/OOM. Checkpoints were saved at 1k/3k/5k/10k. The final checkpoint reloaded bit-identically (4,540,781 bytes; SHA-256 c22e0804b38dc923334fc4e8a639c11ee40c4ef5dfddfd0aafc6f5be5871ecf2). Result JSON SHA-256: 901be3a66d00aafb87c442222ee759b212503589159553ba333aec537ae0ec17; stdout/stderr SHA-256: 9b1d5d07cb3118fcdda6216875613a03d815c3969d18000d9fec3c7b95df80c2.

Frozen RGB artifact/PNG hashes remained identical and RGB maximum pixel difference was 0. The official external CAMERA_FROM_WORLD camera produced 1320×989 RGB, five logits and single-channel uint8 class IDs in 0–4. Checkpoint reload passed.

## Decision

**Drop / hypothesis not supported.** SEM-003 fails the primary gate in the opposite direction: it does not improve mIoU by 1 pp, does not reduce background→stay_cable relative to CTRL, and reduces stay_cable IoU. Protected-class and engineering gates pass, and worst-10% improves slightly, but those do not override the failed target gates. No three-fold/three-seed confirmation, threshold/lambda retry, SEM-004 view residual or pseudo-label stage is authorized by this result. SEM-001 remains the frozen semantic baseline.
