# SEM-391-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY

- Status: closed without formal training because paired control SEM390 failed its trace contract; no inference, holdout access, or quality result.
- Role: corrected seed-`20260919` 50% full-frame context candidate paired with SEM390.
- Scientific variable: odd updates use the paired native `768x768` crop; even updates use the same view and flip as a complete `1320x989` frame resized to `1025x768`.
- Endpoint: exactly 6000 successful updates; checkpoint 3000 is diagnostic only.
- Initialization: fresh ADE Swin-L and exact all-650-tensor/load-report equality with SEM390 at the same seed.
- Holdout: inaccessible until both paired 6000-step checkpoints are sealed and hash-recorded.

Protocol: `docs/experiments/sem390_sem391_context_confirmation_protocol.md`.

## Preregistration Binding

- Config SHA256: `d031775cc623a90219ccdec890daea10346f996f8ac799fdc8878617d0ced728`.
- Parent SEM390 sampling plan SHA256: `03f8b19bf3f3ee1668a783dbe4086fe97c8e122ba027d16afc56b88355bdc489`.

## Preparation

- Candidate sampling plan SHA256: `28240eca557804d93da1de34b628681b53956ce4e343c0314fbcd7ce1de35a63`.
- Pairing audit SHA256: `1c510c0e994a84c8ca56badfec1c830db58d984a04ac8c92337576fa80dd38a5`.
- All 6000 view/crop/flip records match SEM390. Odd updates are 3000 native `768x768` inputs and even updates are 3000 context `1025x768` inputs.
- RGB export hashes match SEM390; prepare read zero holdout RGB and zero holdout labels.

## Preflight

- Result SHA256: `0f8aaecd3ed70cc1f28b9c63a2bb8e472d1874d05f274bf36d3c37b893a780d0`.
- Initialization contract SHA256: `aaf950a7102f8ec9fa0b7090b0e3c9f200b64449b9fbf4b6a53ade335a99cf9c`.
- The reconstructed old-seed reference passed the 648-non-head/two-fresh-head contract. At seed `20260919`, all 650 tensors, both classifier tensors, the full state SHA256, and the pretrained-load report matched SEM390 exactly.
- Four native and four context updates alternated at the fixed dimensions. Losses and all four Swin-stage gradients were finite. Peak allocated/reserved memory was `5669.15/6112.00 MiB`; no checkpoint was retained.
- No holdout labels or RGB were accessed.

## Closure

- SEM390 executed 6000 updates but failed sealing because its actual sampler trace used the inherited old recipe seed rather than the registered second seed.
- Per the fixed paired protocol, SEM391 formal training was not started. Its preparation and successful preflight remain preserved, but no checkpoint or metric is promoted.
- This ID will not be retried or repaired. Holdout remained closed for both arms.
