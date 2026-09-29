# SEM-388-RGB150-DEV270-SEED20260919-MEMORY

- Status: stopped and closed at the fresh-model preflight; no training, inference, holdout access, or quality result.
- Role: native-crop control for the `20260919` second-seed SEM386 confirmation pair.
- Parent recipe/data: SEM384, frozen RGB150, fixed Dev30 270/30 split.
- Endpoint: exactly 6000 successful updates; the 3000-step checkpoint is diagnostic only.
- Runtime policy: all four Swin stages use the same native activation recomputation as the paired context arm.
- Holdout remains inaccessible until both SEM388 and SEM389 6000-step checkpoints are sealed.

Protocol: `docs/experiments/sem388_sem389_context_confirmation_protocol.md`.

## Preparation

- Config SHA256: `0061aacad0fbef834dc9301665bf7dd952e6e5907c320f5913eee675728c01e4`.
- Split SHA256: `b08062bd8ca9fe9dd91dea4a0f1cc784770caf323a05c7ca2c38289643905ad5`.
- Sampling plan SHA256: `52ee0a5f1235a0818f8e814eb88742a9e0cdffbc3e10475fe1a1ccb04f4c997b`.
- The plan has 6000 native updates; every fitting view appears 6 to 39 times. Preparation read 270 training labels and zero holdout labels.

## Preflight failure

- An initial shell launch with the host `python3` exited before model import, data access, GPU use, or preflight artifact creation because that interpreter does not provide Torch. The preserved scientific preflight below used the registered CUDA environment interpreter.
- The preserved preflight stopped before update 1 with `ValueError: Fresh preflight ADE state differs from reference`; zero updates ran and no checkpoint was retained.
- Preflight result SHA256: `c65a93e551cf12806a7c34e31a275161d0e11bc6690d8efad4930dbd5df5826c`.
- Memory-policy result SHA256: `f4b4e01c608229bf65de9312139014db17035c2c1a279527f00232c29e337969`.
- An independent CPU-state diagnosis reconstructed the model once with seed `20260805` and once with seed `20260919`. Of 650 state tensors, 648 were identical; only `sem_seg_head.predictor.class_embed.weight` and `sem_seg_head.predictor.class_embed.bias` differed. Both loads had the same pretrained-load report.
- The old SEM384 reference used seed `20260805`. Requiring the complete second-seed state to equal that old-seed state was therefore incompatible with the planned fresh random five-class head. This is a preflight-contract design failure, not evidence about model quality, context mixing, numerical stability, or memory use.
- The frozen protocol requires the pair to stop after either preflight fails. SEM389 preflight and both 6000-step fits were not started. Any corrected seed-aware initialization contract requires new experiment IDs and a new preregistration; this failed result will not be overwritten or reclassified.
- Verification after closure: 252 JSON-compatible YAML configs validated; targeted lightweight tests reported 9 passed and 1 optional-Torch skip; the recomputation forward/RNG/gradient test passed separately in Torch 2.11.0+cu128; `bash scripts/repo_health.sh` passed 605 tests with 85 optional skips.
