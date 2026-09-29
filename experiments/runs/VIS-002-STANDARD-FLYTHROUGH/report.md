# VIS-002-STANDARD-FLYTHROUGH — frozen Standard-3DGS novel-view video

- Status: completed; visualization-only.
- Parent: `RGB-011-R2-ALLOC`; reuses the camera-only trajectory from `VIS-001-FLYTHROUGH` for directly comparable visualization.
- Run HEAD: `49cdcdb1c9915a9544d568754af95ea20a85ec9c`; pre-existing user worktree changes were preserved.
- Run-completion (before final record update) tracked-diff / porcelain SHA-256: `0ab2f4c96b698ffe1adf3a24aac2ce3edab8df8217731161a371154724796c71` / `5e5a23233d801423152fbdcaffe6144c1c74b46058b32ba768e5606c80993a5f`.

## Frozen inputs and boundary

The video loads the completed 30k `RGB-011-R2-ALLOC` Standard-400 model once: 685,296 ordinary Gaussians and its unchanged final PLY SHA-256 `ebacf0b4fd8b5cec3cce7566e25e9c2d6b33406936c318df40b45c96560b413b`.
It reuses the 144 half-step interpolated poses already frozen by `VIS-001-FLYTHROUGH`, generated only from the seven official camera metadata keyframes `140→146→154→161→169→176→183`. The source pose file has SHA-256 `ebadf102be6e8d94255b38049721e8c7d08b849d16219158419ff6cf5dd744b9`; camera metadata has SHA-256 `b1dba05dd15f76f41dccbe6adaf56201bff817ad49c4d108f573ad7ca18e4e51`.

The renderer receives only official-reference `CAMERA_FROM_WORLD` poses and `SIMPLE_RADIAL` camera metadata, converts through identity Sim(3), and redistorts to the official 1320×989 grid. It accepts no RGB reference-image or label argument, does not train, and does not modify the Standard checkpoint, Scaffold, or SEM-001.

## Result and audit

The one-load batch rendered 144 frames in `25.9909 s` on RTX 5070; peak PyTorch allocated memory was `688 MiB`. Each native frame is RGB uint8 1320×989. FFmpeg encoded them as 24 fps H.264/yuv420p CRF18; a one-row pad produces the codec-required even height 1320×990.

| Artifact | Result | SHA-256 |
|---|---:|---|
| RGB flythrough | 144 frames, 6.0 s, 1320×990, 2,861,965 B | `501c60265de1b035122eae5d8d0e6265bf95733b2d110da9d13d744aa25c988e` |
| Render manifest | 144 per-frame hashes and render contract | `5c2ac9f66b046796efb9714de61546833d4f9fe9989d0d08c5d60ecad3434dd8` |
| Audit result | hashes, RGB dimensions, timeline, frozen PLY and no-reference-data contract passed | `b344be1e1cca91e91bc6c4058e79355e78d25dac44a6975142706dd445902e45` |

All generated frames, video, manifests and logs are Git-ignored under `$ICSHM_OUTPUT_ROOT/results/frozen_standard_rgb/flythrough_v1`; their total size is about 71 MiB. The video intentionally shows the Standard model's novel-view weaknesses: part of the middle trajectory becomes strongly washed out and exhibits floaters. No color correction, masking, frame removal, or other post-processing was applied.

Registered rendering and audit commands use `R1_ENV_ROOT`, `R1_CUDA_HOME`, `R1_THIRD_PARTY_ROOT` and `ICSHM_OUTPUT_ROOT`; the only model source is the frozen Standard-400 PLY. The video is presentation evidence, not a new PSNR/LPIPS experiment, and does not change the frozen RGB-backbone or semantic keep/drop decisions.
