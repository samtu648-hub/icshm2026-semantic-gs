# SEM-052-PL-AUDIT — pseudo-label qualification

Status: OOF thresholds frozen; fixed 74-view audit not yet opened.

Per-class thresholds are frozen from the 219 spatial OOF predictions before the fixed 74-view images or labels are opened. The one-shot fixed-validation audit determines whether the 100 unlabeled images are eligible for candidate soft pseudo-label generation. This is not a 3D semantic result and cannot start Scaffold training.

## OOF-only threshold freeze

Before opening any fixed-validation input, the 219 unique OOF probability maps were aggregated on a fixed 0.001 grid. The frozen artifact SHA-256 is `e96fbb36f91d6faf3dd1e2c9b57ec81acadd014aea008fe39d92a6f677bf1b62`; the freeze result SHA-256 is `f50244230dd1fc3f5aef48f23ef7ee521ddef4fd6850059e6de6a902a037ff5b`.

| Class | Threshold | OOF selective precision | OOF selective recall |
|---|---:|---:|---:|
| background | 0.000 | 0.997140 | 0.978408 |
| deck | 0.667 | 0.950093 | 0.966936 |
| stay_cable | 0.911 | 0.950004 | 0.903823 |
| tower | 0.942 | 0.950282 | 0.911645 |
| foundation | 0.999 | 0.957357 | 0.574783 |

Accepted stay_cable pixels occur in 218/219 OOF views (99.543%). Remote background false positives account for 36.030% of accepted stay_cable false positives, below the pre-registered majority limit.

The stricter fold-consistency gate is already negative under the single global threshold: fold 0 stay_cable precision is 0.935433, fold 1 is 0.967100, and fold 2 is 0.949922. No threshold, model, loss, or seed is changed in response. The one-shot fixed 74-view audit will still be reported, but this negative gate cannot be erased by validation performance.

Registered threshold command:

```bash
env PYTHONNOUSERSITE=1 "$SEM005_PYTHON" scripts/freeze_sem005_thresholds.py \
  --config experiments/runs/SEM-052-PL-AUDIT/config.yaml \
  --labels-root "$ICSHM_OUTPUT_ROOT/r2/sem_001/labels" \
  --fold-dir "$ICSHM_OUTPUT_ROOT/r2/sem_005/oof/fold-0" \
  --fold-dir "$ICSHM_OUTPUT_ROOT/r2/sem_005/oof/fold-1" \
  --fold-dir "$ICSHM_OUTPUT_ROOT/r2/sem_005/oof/fold-2" \
  --output-dir "$ICSHM_OUTPUT_ROOT/r2/sem_005/thresholds"
```

## One-shot fixed 74-view audit

The audit verified the committed threshold artifact before opening validation. It then loaded all three final fold checkpoints and evaluated the fixed 74 views once, without gradients, early stopping, or parameter selection.

| Metric | SegFormer-B2 2D ensemble | Frozen SEM-001 reference |
|---|---:|---:|
| mIoU with background | 0.804696 | 0.616103 |
| mIoU without background | 0.759031 | 0.563337 |
| stay_cable full precision / recall / IoU | 0.833876 / 0.882592 / 0.750613 | 0.077965 / 0.368036 / 0.068759 |
| background→stay_cable pixels | 512,859 | 13,166,331 |
| worst 10% foreground mIoU | 0.448957 | 0.423440 |
| 140–146 foreground mIoU | 0.798380 | 0.601213 |

At the OOF-frozen thresholds, fixed-validation selective precision / recall are:

- deck: 0.964659 / 0.949518;
- stay_cable: **0.911232 / 0.713761**;
- tower: 0.967167 / 0.793820;
- foundation: 0.995002 / 0.385642.

The stay_cable selective precision therefore fails the required 0.95 validation gate. The independently pre-registered fold-consistency gate also fails. Every other registered gate passes: OOF precision/recall/coverage, protected-class selective precision, validation mIoU, worst 10%, 140–146, remote false-positive share, licensing, threshold ordering, and leakage checks.

Audit result SHA-256 is `bc1ee3302905e588aecfbed1adedfec49557bfd5dbdcf5c635f5d6fd055eafee`; output manifest SHA-256 is `c0ac533abff2e8c6170c69858216c7e6bac38d22181bb5b7da7686f44c32aace`; per-view SHA-256 is `56dd888bd94c7889f34d13a896f149ead68985712e94d882d4dff786b8e5f62f`; audit log SHA-256 is `acc2b941cff7a7e665b5c06df5af47f20e268c35cd88067c4ed06c9264afb034`.

Conclusion: `not_eligible_for_pseudo_labels`. No output for the 100 unlabeled RGB images was generated, and no 3D/Scaffold semantic training was started. SEM-001 remains frozen. The correct fallback is to stop this pseudo-label route rather than lower the 95% gate or tune against the fixed validation set.
