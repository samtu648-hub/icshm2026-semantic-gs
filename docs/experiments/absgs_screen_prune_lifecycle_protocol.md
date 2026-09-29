# AbsGS screen-radius lifecycle paired experiment

QUAL018 established that every active native screen-radius event had oversized
Gaussians before append, while append cleared the complete `max_radii2D` vector
and actual screen-size pruning saw zero candidates. It also showed that large
contributors can be useful in some views, so this experiment tests the native
20-pixel rule without changing the threshold or introducing semantic gates.

RGB161 is a fresh native control and RGB162 changes only append bookkeeping:
old radii survive and newly appended children receive zero. Both use the frozen
RGB152 spatial-v1 roles, 353 training cameras, 30 confirmation cameras, 17
excluded buffer cameras, the same 78,045-point training-only initialization,
seed 0, Standard+AbsGS thresholds and 30k schedule.

Before full training, both arms run to step 3200, crossing the first active
screen-size pruning events. The smoke verifies finite checkpoints, identical
registered inputs, native zero post-append candidates, and nonzero preserved
candidates in the intervention. Smoke output is discarded and full runs restart
from initialization.

Evaluation reports all train/confirmation metrics and fixed spatial regions,
plus bridge-focused RGB error on the released labels and visual review of
035--036, 051--052, 155--169, 208, 231, 235--245 and 385--392. Labels are used
only after training for evaluation. Full-image fit cannot override damage to
`deck`, `stay_cable`, tower or foundation. No threshold scan, semantic pruning,
LAS shape, appearance model, transient mask, camera optimization or matched
semantic training is included.

Execution outcome (2026-09-22): the causal smoke passed, but the candidate
removed 291,067 points at the first active event and degraded the 33-view bridge
review by 2.731111 dB PSNR and 0.037185 SSIM on average. The early recovery gate
therefore failed; the 30k pair and semantic follow-up were not run.
