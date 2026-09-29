# SEM-423-COMPONENT-SEED2-CTRL

Completed2026-09-21. Fresh native control, seed20260919, RGB154,
spatial v2 255/30,6000 updates.
[Protocol](../../../docs/experiments/component_seed_confirmation_protocol.md).

Training1564.995 seconds,5 AMP retries, reserved8588MiB. Confirmation foreground
87.247512%, five-class89.637425%, cable IoU81.172680%, boundary F173.773098%.
All650 initial tensors and6000 actual samples match SEM424. Both checkpoints
sealed before inference;379 predictions/285 scored views independently checked,
and RGB154 three-camera replay remains exact before/after. Exit0.

Checkpoint SHA256 `ed666fafa03e31f4178c6652f0dcbe6c255ec8304f6dacb0b32b0e8a2539e611`;
inference SHA256 `e97226bfb41bcbc27494791928236a099f04ee73500e59173d0ace4bfa4b3b47`.
Commands/resources: `outputs/quality/QUAL-017/r1`; arm output:
`outputs/quality/SEM-423/r1`. This control does not replace the deployed system.
