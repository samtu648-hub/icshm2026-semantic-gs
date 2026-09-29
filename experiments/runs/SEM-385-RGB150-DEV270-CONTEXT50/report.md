# SEM-385-RGB150-DEV270-CONTEXT50

- Status: stopped at the preregistered GPU preflight on 2026-09-17; failure preserved and no full training started.
- Parent: `SEM-384-RGB150-DEV270`.
- Split: `dev30_local_composition_v1`, with the same 270 semantic training views and 30 sealed development holdout views.
- Seed and endpoint: `20260805`, 6000 successful updates.
- Single variable: even updates use an aspect-preserving `1025x768` full-frame input; odd updates retain the corresponding SEM384 native `768x768` crop.
- Fixed parent sampling-plan SHA256: `b17b813ae84d6523f4fc6b322643348e44663e47788cf02c2e9ffa521f0b7dc0`.
- Frozen SEM385 sampling-plan SHA256: `9341f59f4fe3f794223aea1dfa72dca09e4f9ae0ef7f6d033338e434431cb6d5`.

Protocol: `docs/experiments/sem385_context_mix_protocol.md`.

## Preparation

- The plan contains exactly 3000 `native_crop` and 3000 `full_frame_context` updates.
- Every candidate row retains the parent view, native crop coordinates, and flip; the wrapper calls the parent sampler on every update and preserves its Python RNG consumption.
- Only the 270 semantic training labels were hashed. Holdout label hashes read during preparation: `0`.
- The verified SEM384 370-view RGB export is reused through a local output symlink; no holdout RGB byte is opened during preparation or preflight.
- Five targeted sampling tests passed, including exact balance, four-corner retention, paired shape, parent trace, and RNG-state equality. A total of 242 JSON-compatible configs and Python compilation also passed before GPU execution.

## Preflight Failure

- The independent fresh-model preflight was required to complete four native and four context updates before any 6000-step run.
- Step 1, the registered `native_crop` for `293.png`, completed with finite loss `75.754219`, finite clipped gradient input, and one standard AMP backoff from scale 1024 to 512.
- Step 2 is the first registered `full_frame_context` input. Its forward pass reached the Mask2Former transformer decoder but failed with CUDA OOM while requesting another 40 MiB.
- At failure, PyTorch reported 9.87 GiB allocated and 415.05 MiB reserved but unallocated; only 50.69 MiB of the 11.46 GiB device capacity was free. No outside compute process was present before launch.
- Successful updates: `1/8`; retained preflight checkpoint: none; full 6000-step training: not started; inference and holdout evaluation: not started.
- The process exit released GPU memory; afterward the device returned to approximately 932 MiB idle use with no compute process.

The failure activates the preregistered stop condition. No long-edge resize, letterbox, activation checkpointing, model reduction, context-fraction change, or other rescue is allowed under SEM385. Therefore SEM385 provides an engineering resource result only and has no quality metrics or comparison against SEM384.

## Evidence

- Prepare result SHA256: `0d088356bec33cb73b4e648c0677efe2c80f2f2d052a835cc8b1b6201dde1b17`.
- Preflight result SHA256: `ef6c3971187b498ccbd8cd7621ee4757ec6835237b7c4ed97c2c49dd2348b720`.
- One-step training journal SHA256: `665069007833318895bbff0d7237e02eb58f2accd69a29b2cfe0c0dda22a3bdf`.
- Failure artifact: `outputs/quality/SEM-385/r1/preflight/result.json`.

Conclusion: retain SEM384 as the fixed Dev30 semantic baseline. A smaller full-frame representation or activation-checkpoint variant would be a new resource-and-input protocol and requires a new experiment ID and explicit authorization; the previously proposed RGB detail arm also remains unauthorized.

Final verification: 5 targeted SEM384/SEM385 tests passed, 242 JSON-compatible configs validated, Python compilation and `git diff --check` passed, and `repo_health` passed 575 tests with 80 optional skips.
