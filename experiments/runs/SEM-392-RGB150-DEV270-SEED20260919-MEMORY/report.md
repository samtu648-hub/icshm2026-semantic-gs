# SEM-392-RGB150-DEV270-SEED20260919-MEMORY

- Status: completed control for a failed two-seed confirmation; no migration authorized.
- Role: seed-thread-corrected seed-`20260919` native-crop control paired with SEM393.
- Parent recipe/data: SEM384, frozen RGB150, fixed Dev30 270/30 split.
- Endpoint: exactly 6000 successful native `768x768` updates; checkpoint 3000 is diagnostic only.
- Initialization: fresh ADE Swin-L under the corrected per-tensor seed-aware contract.
- Sampling: model, training recipe, and private sampler must all receive seed `20260919`; a complete native trace replay must match before GPU training starts.
- Runtime: matched four-stage Swin activation recomputation.
- Holdout: inaccessible until both SEM392 and SEM393 6000-step checkpoints are sealed and hash-recorded.

Protocol: `docs/experiments/sem392_sem393_context_confirmation_protocol.md`.

## Preparation

- Config SHA256: `ae4bedcede10c8da99ef4a4f6a2949f4e72b2be24bb7bbc8aee3d9b660d26b92`.
- Sampling plan SHA256: `0339682c6c0bf60512d774d784ce56936dca041390e8309890e9380bfb30380e`.
- The 6000-step plan records seed `20260919`; its first entry matches the registered-seed regression fixture.
- 270 training-label hashes were verified; prepare read zero holdout labels and no holdout RGB.

## Preflight

- Result SHA256: `54dde181632ac2a06bb6e46ba5b22606903621b294684682ce85c1b55a4562b2`.
- Initialization contract SHA256: `c43c9fcc05ec4e7940c04f6b8cc7fe6df4ec8fd0f76c9d8c27a2cde33fd37fcb`.
- The old-seed state matched SEM384; 648 non-head tensors were exact across seeds and only the correctly shaped, finite classifier pair differed. Load reports matched.
- Eight native updates completed with finite losses and all-stage finite gradients. Peak allocated/reserved memory was `4940.37/5364.00 MiB`; no checkpoint was retained.
- No holdout RGB or label was accessed.

## Formal Training

- Result SHA256: `6e573ae021135c30910d05c7e736d44ff23c1531af5726755e14cc3f453dd67a`.
- Checkpoint 3000/6000 SHA256: `af5b5fd5e951eba47b8696b6b8e9338b4277b7cbf02912123f608e497cbf58f2` / `1763e203f02d9ac12c453abcd4f8d6521762af79232bb6bfa706b1f45a0e810a`.
- The complete 6000-step trace exactly matched the registered seed-`20260919` plan. Fresh initialization matched the preflight across all 650 tensors and the load report.
- Training completed in `1810.59 s`; peak allocated/reserved memory was `4940.37/5364.00 MiB`. Each checkpoint is `866052203` bytes.
- Frozen RGB hashes remained exact; holdout label/RGB access during fitting was zero.

## Fixed Inference And Pair Result

- Fixed inference completed for all 400 views after both paired checkpoints were sealed. Inference result SHA256: `cab0712b74912242a79418b8d347e902556874d7acab4099d54a8eee30e7424f`.
- Train five-class/foreground mIoU: `95.351882%` / `94.332759%`.
- Holdout five-class/foreground mIoU: `93.872085%` / `92.483224%`.
- SEM393 changed holdout five-class/foreground mIoU by `-0.536632 pp` / `-0.659322 pp` relative to this control. The paired comparison SHA256 is `7e05d2be46986fa752889bf9b383447dca466ea14121d6df3b5bb5ae172bf069`.
- This arm is a matched control, not an independent method promotion. The frozen two-seed confirmation failed and RGB140 migration is not authorized.
