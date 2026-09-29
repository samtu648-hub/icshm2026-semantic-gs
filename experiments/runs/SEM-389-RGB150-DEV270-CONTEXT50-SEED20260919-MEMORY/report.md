# SEM-389-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY

- Status: closed without candidate preflight after the paired SEM388 control preflight failed; no training, inference, holdout access, or quality result.
- Role: 50% full-frame context candidate paired to SEM388 at seed `20260919`.
- Parent sampling plan SHA256: `52ee0a5f1235a0818f8e814eb88742a9e0cdffbc3e10475fe1a1ccb04f4c997b`.
- Endpoint: exactly 6000 successful updates; odd steps use native `768x768`, even steps use complete `1025x768` frames.
- Runtime policy: all four Swin stages use the same native activation recomputation as SEM388.
- Holdout remains inaccessible until both SEM388 and SEM389 6000-step checkpoints are sealed.

Protocol: `docs/experiments/sem388_sem389_context_confirmation_protocol.md`.

## Preparation

- Config SHA256: `71a864d34164abf71a01aa9ffffe3dc956378622db922189191216e964f44e26`.
- Split SHA256: `b08062bd8ca9fe9dd91dea4a0f1cc784770caf323a05c7ca2c38289643905ad5`.
- Parent sampling plan SHA256: `52ee0a5f1235a0818f8e814eb88742a9e0cdffbc3e10475fe1a1ccb04f4c997b`.
- Candidate sampling plan SHA256: `f927e3e39f9f627c04c68a46ca9adb108d8836eaf1a8ccf170b1da5ea42cd803`.
- The candidate retains all 6000 parent views, crop-coordinate draws, and flips. Odd updates are native `768x768`; even updates are complete `1025x768` frames. Preparation read 270 training labels and zero holdout labels.

## Paired stop

- SEM388 stopped before its first preflight update because the implementation incorrectly required a fresh seed-`20260919` five-class head to have the complete state hash of seed-`20260805` SEM384.
- Per the frozen paired protocol, SEM389 did not run its candidate preflight and neither arm started 6000-step training. The prepared plan remains evidence only and is not a completed experiment.
- Any continuation requires new experiment IDs and a seed-aware initialization contract registered before execution. SEM389 will not be rescued or overwritten in place.
