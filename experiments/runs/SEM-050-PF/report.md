# SEM-050-PF — SegFormer-B2 engineering smoke

Status: completed preflight; not a quality result.

This run checks paired PINHOLE RGB/label geometry, the single weighted cross-entropy loss, CUDA mixed precision, checkpoint reload, and 1320×989 sliding-window probability output. It never opens the fixed 74-view validation set or any unlabeled image.

## Real run evidence

- Code freeze: `59937945ced2f3adb22ef57f832804cced86791d`.
- OOF split SHA-256: `ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`.
- Public `nvidia/mit-b2` PyTorch weight SHA-256: `4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f`.
- Fold 0 used eight training-side images for 16 iterations; one OOF image was used only for output-shape and metric plumbing.
- Weighted CE was finite and decreased from 1.633807 to 0.410771.
- BF16 training took 1.810 s and peak PyTorch CUDA allocation was 3,221 MiB.
- The final checkpoint reloaded successfully; its SHA-256 is `111e417c054295f32c86ee3f644a0148ad66e41c00c9ab546666efafcfa461f4`.
- Full-resolution sliding inference produced five probabilities and a single-channel 0–4 PNG at 1320×989.
- Fixed 74-view validation, all unlabeled RGB, Scaffold, and SEM-001 parameters were not opened or trained.
- Result SHA-256: `3f560587a1bea2f121a723ca2400ebe1107244078c89b4c706103fa8952107f5`; log SHA-256: `0a648e31b098aa94684ed6373312ed7fdb40c165fb9a7410f578513ab7a2e4ca`.

Registered command (paths resolved through the project/output roots):

```bash
env PYTHONNOUSERSITE=1 "$SEM005_PYTHON" scripts/run_sem005_segformer.py \
  --config experiments/runs/SEM-050-PF/config.yaml \
  --split experiments/splits/sem_005_oof_spatial_v1.json \
  --fold 0 --prepared-scene "$ICSHM_OUTPUT_ROOT/rgb_r1_spatial_v1/scene_400_eval" \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --model-dir "$ICSHM_OUTPUT_ROOT/r2/sem_005/third_party/nvidia_mit_b2_3bb39e8" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_005/preflight" --mode smoke
```

Conclusion: the CUDA, loss, geometry, reload, and output-schema gates pass. The one-image smoke metrics are deliberately excluded from model-quality decisions. SEM-051-OOF is authorized by this engineering gate.
