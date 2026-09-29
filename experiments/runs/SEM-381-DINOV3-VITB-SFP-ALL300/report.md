# SEM-381-DINOV3-VITB-SFP-ALL300

Status: completed; negative matched method-package result; not promoted.

- Parent/control: `SEM-378-RGB140-ABSGRAD-ALL300`
- Protocol: [SEM381 DINOv3 ViT-B/16 + SFP](../../../docs/experiments/sem381_dinov3_vitb_sfp_protocol.md)
- Config: [config.yaml](config.yaml)
- Output: `outputs/quality/SEM-381/r1`
- Scope: all300 training-view fit; unlabeled100 and trajectory144 qualitative only.
- Git: `49cdcdb1c9915a9544d568754af95ea20a85ec9c`
- Seed: `20260805`
- Split: `released_all400_rgb_all300_sem_v1`

## Execution

The registered treatment froze the hash-verified local DINOv3 ViT-B/16 and
trained only its SFP plus the existing ADE-initialized Mask2Former
decoder/classifier. RGB140, all 300 labels, the SEM378 6000-entry sampling
sequence, native 768 crops, 768/512 sliding inference, loss, optimizer schedule,
seed and checkpoint endpoints remained matched.

- The eight-update native-768 preflight completed with 8/8 successful updates.
  Peak memory was 2439.679 MiB allocated and 2604 MiB reserved. The first AMP
  scale backed off from 1024 to 256; two skipped optimizer attempts were
  recorded before eight successful updates.
- The independent formal run completed 6000 successful updates in 923.164 s
  (15.386 min), with two AMP skips. DINOv3 state SHA256 remained exactly
  `032a9ca331800355f69e7be93db33f564fff28a3f9f43736c89b422d3e3b7445`.
- The model has 109,753,222 parameters: 85,660,416 frozen DINOv3 parameters and
  24,092,806 trainable parameters. The 6000-step checkpoint is 439,229,999
  bytes, SHA256
  `248faf209079d8ea5d38028936febf59b914c33d6474783cd9c02cb1b79f5d29`.
- All 544 images were inferred in 200.413 s of measured forward time; median
  per-image time was 0.367510 s. Nine fresh probability replays were byte-exact.
- Both checkpoints were finite, all 544 RGB140 images were freshly rerendered
  byte-exact, trajectory metadata matched, and the 18 frozen artifacts, SEM378
  endpoint and RGB140 endpoint remained unchanged.

Evidence: `mask2former/result.json`, `inference/result.json`,
`verification.json`, `audit.json`, `visual-review.json` and
`comparison/index.html` under the output directory.

## All300 training fit

SEM381 mIoU with background was **94.796554%**, versus **95.297405%** for
SEM378: **-0.500851 pp**. Foreground-only mIoU was 93.650362%, versus
94.262248%: -0.611886 pp. This is training-view fit, not held-out accuracy.

- Background IoU: 99.381321%, versus 99.438031% (-0.056710 pp).
- Deck IoU: 96.867367%, versus 97.101676% (-0.234310 pp).
- `stay_cable` IoU: 94.572863%, versus 95.152749% (-0.579886 pp).
- Tower IoU: 93.652535%, versus 94.387299% (-0.734764 pp).
- Foundation IoU: 89.508684%, versus 90.407267% (-0.898584 pp).

Only 21/300 views improved; 279 degraded and none tied. Across 1,198,304
changed pixels, 467,311 were corrected, 709,350 were harmed and 21,643 changed
from one wrong class to another. Every GT class had more harmed than corrected
pixels.

For `stay_cable`, SEM381 fixed 136,437 SEM378 false negatives but introduced
225,493 new false negatives, a net increase of 89,056. It removed 151,680 old
false positives but introduced 229,389 new false positives, a net increase of
77,709. The 2 px cable boundary F1 fell from 0.880399 to 0.845960; precision
fell from 0.893135 to 0.874155 and recall from 0.868021 to 0.819527.

## Visual review

All fixed nine views, all 100 released unlabeled views and all 144 trajectory
frames were produced at matching RGB140 locations. The required 159/160/146/
057/064 views, trajectory frames 96--98 and unlabeled 350 were inspected;
311/390, the largest-change unlabeled views and additional trajectory ranges
were also checked.

- 057 and 064 degraded by 4.346309 pp and 4.096625 pp respectively; 064 has a
  visible hole in the cable envelope and 1,328 additional cable false negatives.
- 146 degraded by 1.100337 pp and showed no stable qualitative repair.
- 159/160 degraded by 1.867974/1.278259 pp. Their near cable boundaries are
  rougher or spill farther than SEM378, matching the increased cable FP/FN.
- On no-GT views 350 and 352--358, SEM381 creates tower-colored fragments near
  boats or water and rougher cable/deck transitions. These are qualitative
  concerns only, not accuracy measurements.
- The trajectory mean adjacent label-change proxy is similar (0.028895 versus
  0.029185), while the maximum increases from 0.117849 at transition 98 to
  0.121558 at transition 97. Frames 96--98 retain fragmented cable interiors.

The wide yellow `stay_cable` sectors reflect the released annotation style and
must not be described as reconstruction of individual physical wires.

## Preserved failures and recovery

The first verification launch reached the fresh RGB140 rerender stage and
failed because `$USER_HOME/nuscenes_env` lacked `plyfile`. A first dependency
reuse attempt stopped earlier because another environment's `scripts` package
shadowed the repository namespace. Verification then succeeded without source
changes by injecting only the existing single-file `plyfile` module; this did
not change predictions, checkpoints or bound code.

The original audit completed all 300 calculations but strict JSON writing
rejected an undefined foreground-only mIoU for a background-only ROI. The
partial `audit.failed-nan.json` and `error-changes.failed-nan/` were preserved;
the first ineffective recovery attempt was also preserved separately. The
successful runtime recovery represented only that undefined value as JSON
`null`, recorded the intervention in `audit.json.runtime_recovery`, and left
the bound audit script unchanged. Aggregate, per-view, boundary and region
counts were not changed.

## Decision

Do not promote SEM381. It is lower than SEM378 in aggregate mIoU, every class
IoU, paired-view count, corrected-versus-harmed pixels and cable boundary F1.
The lower memory and faster runtime of the frozen DINOv3 package do not offset
the quality regression. RGB140+SEM378 remains the system candidate and direct
rollback; RGB130+SEM371 remains the project default.

No layer, SFP, learning-rate, sampling, capacity, post-processing or endpoint
scan is authorized by this result. Any future DINOv3 work needs a separately
registered single-variable design plus independent held-out GT or multi-seed
evidence. One seed and all300 training fit cannot establish stability or
hidden-view generalization.
