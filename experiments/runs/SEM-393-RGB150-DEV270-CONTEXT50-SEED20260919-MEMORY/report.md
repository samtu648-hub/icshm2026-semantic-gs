# SEM-393-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY

- Status: completed confirmation failure; retained as a negative second-seed result and not promoted.
- Role: seed-thread-corrected seed-`20260919` 50% full-frame context candidate paired with SEM392.
- Scientific variable: odd updates retain the paired native `768x768` crop; even updates retain the same view/flip but use the complete frame resized to `1025x768`.
- Endpoint: exactly 6000 successful updates; checkpoint 3000 is diagnostic only.
- Initialization: exact all-650-tensor and load-report equality with SEM392 at the same seed.
- Holdout: inaccessible until both paired 6000-step checkpoints are sealed and hash-recorded.

Protocol: `docs/experiments/sem392_sem393_context_confirmation_protocol.md`.

## Preregistration Binding

- Config SHA256: `041c7d70cbc9b449aba2461a0421d44e93d607d305e2f906efad030044907019`.
- Parent SEM392 sampling plan SHA256: `0339682c6c0bf60512d774d784ce56936dca041390e8309890e9380bfb30380e`.

## Preparation

- Candidate sampling plan SHA256: `c27cf22ffe02c15511d2ffb2addff427e9cbf3e4c081607b49b3a50f187be4ad`.
- Pairing audit SHA256: `5ae4d828291cbdeaa1561bf5361cb9494884ef1592f1a48e7f2c0a0b224ae5bc`.
- All 6000 view/crop/flip records match SEM392. Odd updates are 3000 native `768x768` inputs and even updates are 3000 context `1025x768` inputs.
- RGB export hashes match SEM392; prepare read zero holdout RGB and zero holdout labels.

## Preflight

- Result SHA256: `ba4fa684a062238b918c362904ae9b74857ff6423701697ccc10fe3252e7bf67`.
- Initialization contract SHA256: `99109c02ca9c083265d80c2d96b4cd34c9304f3efe193dd54a433f197032eb8f`.
- At seed `20260919`, all 650 tensors, both classifier tensors, the full state SHA256, and the pretrained-load report matched SEM392 exactly.
- Four native and four context updates completed with finite losses and all-stage finite gradients. Peak allocated/reserved memory was `5669.15/6112.00 MiB`; no checkpoint was retained.
- No holdout RGB or label was accessed.

## Formal Training

- Result SHA256: `a8bc9466f0fdd89f44f2d3add0498517c96b0b9e8db6ec60206849f6c4698fef`.
- Checkpoint 3000/6000 SHA256: `1d8f1a47eab066884d927567f3ed1ce4e6d06c37b44f5b2c6e7964bcd6cbc6a7` / `858399f80306ccaefc6bcb585c22067d3763d696e709877a710e032d4323454c`.
- The complete trace matched all 6000 frozen entries and exactly 3000 native/3000 context updates. Fresh initialization matched SEM392 formal training across all 650 tensors and the load report.
- Training completed in `2071.03 s`; peak allocated/reserved memory was `5669.15/6112.00 MiB`. Each checkpoint is `866052203` bytes.
- Frozen RGB hashes remained exact; holdout label/RGB access during fitting was zero.

## Fixed Inference And Comparison

- Fixed inference completed for all 400 views after both paired checkpoints were sealed. Inference result SHA256: `ff465ddad80619200f78a1d84431bb8ec6ea1446f913f5361cc7ae7ed9d6c7bc`.
- Comparison result SHA256: `7e05d2be46986fa752889bf9b383447dca466ea14121d6df3b5bb5ae172bf069`; all 600 labeled prediction confusion matrices were independently reloaded and matched.
- Train five-class/foreground mIoU improved by `+0.053723 pp` / `+0.064044 pp`, but holdout five-class/foreground mIoU changed by `-0.536632 pp` / `-0.659322 pp`.
- Holdout class deltas were background `-0.045872 pp`, deck `-0.058053 pp`, `stay_cable -0.587087 pp`, tower `-0.224125 pp`, and foundation `-1.768021 pp`. Cable FP decreased by `13371`, but FN increased by `25163`; 2px cable boundary F1 increased by `+0.296566 pp`.
- Group five-class deltas were regular-local `-0.819150 pp`, oblique-local `-0.053232 pp`, and distant-local `-0.048976 pp`. Holdout had 9 improved and 21 worse views, with 28313 corrected and 44575 damaged pixels.
- The paired 10000-draw bootstrap five-class 95% interval was `[-0.861327, -0.186396] pp`, median `-0.526697 pp`, with `P(delta>0)=0.0013`.

## Visual And Confirmation Gate

- All 30 RGB/GT/control/candidate/change panels were reviewed across three fixed ten-view sheets. No new large-area collapse, remote isolated semantic block, or systematic cable/tower discontinuity was observed, so the frozen visual hard gate passed.
- Regular-local remains visibly adverse: changes around 110--114 include shortened `stay_cable` segments and tower/foundation boundary damage. Oblique view 148 is the clearest local regression; distant changes are mostly small and boundary-localized.
- The seed-`20260805` result passed its quantitative gate, but this seed failed five-class, foreground, class-protection, and group-protection checks. The two-seed mean five-class delta is `-0.246721 pp`, so both cross-seed checks fail.
- Final gate SHA256: `28b71c99ccd0bc9dc43554297648db3beb1a00e506c9c76ce0e1cad0ec233034`. Final status is `confirmation_failed`, `migration_authorized=false`. No RGB140 all-data migration, rescue tuning, endpoint selection, or further run is authorized by this result.
