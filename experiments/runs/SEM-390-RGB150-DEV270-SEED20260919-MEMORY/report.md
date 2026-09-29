# SEM-390-RGB150-DEV270-SEED20260919-MEMORY

- Status: formal training failed the frozen trace audit and the experiment is closed; no inference, holdout access, or quality result.
- Role: corrected seed-`20260919` native-crop control paired with SEM391.
- Parent recipe/data: SEM384, frozen RGB150, fixed Dev30 270/30 split.
- Endpoint: exactly 6000 successful native `768x768` updates; checkpoint 3000 is diagnostic only.
- Initialization: fresh ADE Swin-L; the old-seed reference requires 648 non-head tensors to match and allows only the correctly shaped, finite classifier weight and bias to differ.
- Runtime: all four Swin stages use native activation recomputation, matched to SEM391.
- Holdout: inaccessible until both SEM390 and SEM391 6000-step checkpoints are sealed and hash-recorded.

Protocol: `docs/experiments/sem390_sem391_context_confirmation_protocol.md`.

## Preparation

- Config SHA256: `92cfd441f54a335e39a7c27e5b3a130d53f4427d597a261fd8761a4a2a150dc1`.
- Sampling plan SHA256: `03f8b19bf3f3ee1668a783dbe4086fe97c8e122ba027d16afc56b88355bdc489`.
- The independent plan contains exactly 6000 native updates from seed `20260919`.
- 270 training-label hashes were verified; prepare read zero holdout labels and no holdout RGB.

## Preflight

- Result SHA256: `8936880d926ab75589c1f112964df73f1a1e81cf58a520c271960c32b8e3e93b`.
- Initialization contract SHA256: `4bcfed859d1b357d93d5aad54492c6a8308756cb2cbbac63f96736a0a1a3038e`.
- The reconstructed seed-`20260805` full state matched SEM384. Across seeds, 648 non-head tensors were exact; only the two correctly shaped, finite classifier tensors differed, and pretrained-load reports were exact.
- The seed-`20260919` state SHA256 is `a9f66a9ada969f1bde050ac3cc184ae332d72877b25caeda56b7e9436de9f7d2`.
- Eight native updates completed with finite losses and finite participation from all four Swin stages. Peak allocated/reserved memory was `4940.37/5364.00 MiB`; no checkpoint was retained.
- No holdout labels or RGB were accessed.

## Formal Training Failure

- The fresh seed-`20260919` model initialization passed again and 6000 finite updates executed, but final sealing failed with `ValueError: Training trace differs from the frozen 6000-step sampling plan`.
- Failed result SHA256: `18873a3114be097b65a1c961ffb0970779525467deb2a7e74a295166a350c1eb`.
- Preserved trace SHA256: `d21fdeff4c1fee06695e65d8fe6d061f3e7bdc4d0ba99a3008f4f5ff421980c9`.
- Read-only diagnosis showed all 6000 actual samples exactly matched the old seed-`20260805` SEM384 plan. The shared runner passed the unchanged SEM370 recipe seed into its private sampler instead of the registered SEM390 seed.
- Checkpoints written before the failed sealing audit are invalid experiment endpoints and will not be used. Holdout remained closed.
- The output directory is immutable and will not be retried or repaired. The paired SEM391 formal training is cancelled. A corrected seed-threading run requires new IDs and preregistration.
