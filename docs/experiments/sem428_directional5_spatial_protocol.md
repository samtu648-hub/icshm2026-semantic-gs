# SEM428 directional five-class spatial comparison

SEM428 is authorized only by the passed SEM427 engineering preflight. It keeps
RGB152 frozen and reuses SEM426's exact 256 training views, 30 spatial
confirmation views, 17 excluded buffer views, seed and 6000 crop/flip samples.
The sole main variable against SEM426 is replacing a fixed five-class Gaussian
distribution with five affine functions of registered viewing direction.

Training must finish and seal the step-6000 checkpoint before any confirmation
label is accessed. Evaluation uses the same five-class/foreground IoU,
per-class metrics, 2 px `stay_cable` boundary F1, worst-20% statistic and three
spatial regions as SEM426/SEM396. Step 6000 is the only selectable endpoint.

The result can show whether first-order view conditioning repairs the static 3D
failure on this development fold. It cannot establish blind-test superiority.
Any failure is preserved; no tuning or rescue occurs under this experiment ID.
