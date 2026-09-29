# Fixed second-seed confirmation of the positive component quota

2026-09-21, under the user's explicit authorization to directly pursue worthwhile
follow-ups. QUAL015's fixed quota improves v2 foreground mIoU0.261825pp, cable
IoU0.585769pp and boundary F10.576574pp with all registered retention checks
passing. This supports one additional matched semantic seed, not a quota scan.

SEM423/424 change only semantic seed to20260919; use the same frozen RGB154,
v2 255/30 roles, 6000 steps, 768/512, 16 points/component, <=1024 full-image
area,5% cap, private quota seed418 and unchanged matcher/loss. Both start fresh
from the same650 tensor state checked against sealed SEM392. Preflight eight
updates uses the new registered paired sample plan and is discarded. Seal both
full fits before scoring. Reuse the unchanged component implementation and
independent PNG/boundary/small-component audit.

Require the original per-seed retention checks separately: foreground>0,
boundary>=0, cable precision loss<=0.1pp and each foreground class IoU
loss<=0.5pp. A positive two-seed mean cannot hide failure of either seed.
Keep region/small-component trade-offs and SEM416's failed prevalence gate.
No third seed, quota/area/weight tuning, combination or full-data migration
follows automatically. Single known development fold remains non-blind;
even success is not proof of stability across RGB runs or new scenes.

Run after the native RGB158 replay. Estimate70--100 minutes and4GiB. Source,
inputs, full commands, time, memory and failure records remain bound per stage.
