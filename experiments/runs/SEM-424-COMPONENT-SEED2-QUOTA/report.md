# SEM-424-COMPONENT-SEED2-QUOTA

Completed2026-09-21; retention gate failed. Fixed original component quota at
seed20260919, paired with SEM423.
[Protocol](../../../docs/experiments/component_seed_confirmation_protocol.md).

Training1563.690 seconds,4 AMP retries, reserved8590MiB. Confirmation foreground
86.918042%, five-class89.373766%, cable IoU81.168815%, boundary F173.970527%.
Candidate-minus-control foreground-0.329470pp, cable IoU-0.003865pp,
boundary+0.197429pp. Cable precision+0.760821pp but recall-0.901140pp;
foundation IoU-1.127419pp. Both the primary gain and class-protection checks fail.
All three regional foreground differences are negative; region1 loses1.502803pp.

Small-cable TP still improves8236→9432/15257; small-foundation TP21585→22232/23992.
This does not offset whole-image/whole-class losses. Quota replacement affects
2798770 point draws across99250 mask calls, including auxiliary losses and
AMP retries. No change to quota size, losses, matching, endpoint or inference.

Engineering verification,379 prediction hashes,285 confusion/boundary recounts,
paired initialization/samples and RGB freeze all pass. All650 saved state
tensors finite. Checkpoint SHA256
`abf7ea0de380b94c46ca0920f4c9ca31df749ae8f8484afd920184fc2aa46f8a`;
inference SHA256 `d826ee9b1e3a8c304f11f6314d0f07d0f56b3451f1f0b4d6c71b710d192d0d69`.
Keep negative result and close the fixed quota route; no third seed, parameter
scan or production migration. See QUAL017 for both seeds.
