# SEM-003-CTRL — Dirichlet representation matched 10k control

- Status: **completed matched control**.
- Parent: SEM-001, SEM-003-DIAG, SEM-003-CTRL-PF.
- Actual run HEAD / code freeze: d84e5c93c40a6d58cca836582a0ec23bfa340cd4 / b4cdc4e3e7721b8180ff409b5da7836cd3b01adb.
- Protocol: 219 training labels, 74 spatial validation views, seed 20260731, continuous 0→10,000 iterations; no global gate and no conflict regularizer.

## Result

| Metric | SEM-001 | SEM-003-CTRL | CTRL − SEM-001 |
|---|---:|---:|---:|
| mIoU with background | 0.616103 | 0.616209 | +0.000106 |
| mIoU without background | 0.563337 | 0.563305 | −0.000031 |
| stay_cable IoU | 0.068759 | 0.068865 | +0.000106 |
| stay_cable precision / recall | 0.077965 / 0.368036 | 0.078136 / 0.367260 | +0.000171 / −0.000776 |
| background→stay_cable pixels | 13,166,331 | 13,106,856 | −59,475 |
| background→stay_cable / GT background | 15.0076% | 14.9398% | −0.0678 pp |
| worst-10% mIoU without background | 0.423440 | 0.423505 | +0.000065 |
| 140–146 mIoU without background | 0.601213 | 0.600672 | −0.000541 |

Per-class CTRL IoU: background 0.827822, deck 0.901161, stay_cable 0.068865, tower 0.834902, foundation 0.448293. The near-equivalence to SEM-001 confirms that the Dirichlet representation/control is valid and that candidate differences can be attributed to the registered regularizer.

Training took 3,088.626 s (0.85795 GPU h), peak PyTorch allocation 846 MiB. Checkpoints were saved at 1k/3k/5k/10k. The final checkpoint reloaded bit-identically (4,540,781 bytes; SHA-256 9fad9ff8f57c59fe85a4bc7e9a576300cf60a9fd43def808ce71a22e91c6b3be). Result JSON SHA-256: 4171b5587556da8d3cd203c30511266d7e815cb427a0c752ba4226148d551ba6; stdout/stderr SHA-256: f9eb4334d663e83c007e178a935f49da4020693019438082edb740ab448d18f8.

Frozen RGB artifact and RGB PNG hashes were unchanged; maximum RGB pixel difference was 0. Checkpoint reload and official external CAMERA_FROM_WORLD 1320×989 RGB + five logits + uint8 0–4 PNG passed.
