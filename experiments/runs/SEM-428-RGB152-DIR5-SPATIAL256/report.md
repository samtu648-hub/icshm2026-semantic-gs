# SEM-428-RGB152-DIR5-SPATIAL256

Status: completed and independently audited; not promoted.

Protocol: [sem428_directional5_spatial_protocol.md](../../../docs/experiments/sem428_directional5_spatial_protocol.md)

This experiment is a single-variable comparison against SEM426 and a matched
quality comparison against the two-dimensional SEM396 anchor. It reused the
exact 256/30/17 roles, seed and 6000 SEM426 samples. Training took 427.119 s,
reserved 7,106 MiB, sealed the step-6000 checkpoint before confirmation access,
and preserved exact checkpoint replay and frozen RGB output.

## Results

- Training five-class/foreground mIoU: 71.171005% / 66.877154%.
- Spatial-confirmation five-class/foreground mIoU: 57.450753% / 50.296747%.
- Confirmation `stay_cable` IoU: 13.873347%; precision 15.573982%; recall
  55.956563%; 2 px boundary F1 16.602755%.
- Relative to static 3D SEM426: +7.536397 pp five-class mIoU, +8.350061 pp
  foreground mIoU, +4.478305 pp cable IoU and +3.852605 pp boundary F1.
- Relative to 2D SEM396: -31.753853 pp five-class mIoU, -36.370630 pp
  foreground mIoU, -75.482275 pp cable IoU and -62.916571 pp boundary F1.
- Region cable IoU was 3.858735% / 14.834066% / 23.605260%. The worst six
  confirmation views were 158, 160, 159, 153, 161 and 189, with mean
  foreground mIoU 31.946542%.

The 286 prediction hashes, confusion matrices, split roles, checkpoint,
resource bounds and RGB isolation passed a 20/20 independent audit. First-order
view conditioning therefore carries real signal and partially repairs the
static representation, but it remains far below the 2D anchor and still
over-predicts `stay_cable` (4,074,365 false-positive cable pixels on the
confirmation set). It is retained as a useful ablation, not a system candidate.
No seed, fold, higher-order basis or tuning rescue is authorized under this ID.
