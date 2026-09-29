# GEO-027-STD-CTRL-10K

Status: `completed_control`.

## Contract and execution

This is the fresh continuous Standard 3DGS control for the GEO-027 matched
pair. It used the original 72,292-point training-only DLT initialization,
Standard commit `54c035f7834b564019656c3e3fcc3646292f727d`, the fixed
`rgb_r1_spatial_v1` 303/74 split, seed 0, native 1320x989 training, and the
execution-time config SHA256
`0cfbec1241485b236dd98af2d4ff15619d6399d8e912f69d60c869b6f2a4360e`.
The repository commit recorded at preparation was
`49cdcdb1c9915a9544d568754af95ea20a85ec9c`; the bound runtime source used
working-tree diff / dataset reader / train.py / render.py SHA256 values
`843942f218b47a5b03a517881f4cc229d0eeb898f8fbdaa1f86aaf6ae69f8e21` /
`44ac9fe50b5a6a219676cceb8d334111f4bd21f1f83568e9378d82a4e7828b1a` /
`0fc5d65b0c789b250b6e770757925909f5ea94cdc7c5e977eaaec99ddc03e92c` /
`601b01c7caaf5f02fa0e11c508664ef156f2d9d8a6d5ad5f3bf7737ae773e305`.
The rasterizer/simple-knn commits were
`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` /
`86710c2d4b46680c02301765dd79e465819c8f19`. The runner SHA256 was
`a5f08c9c06c7f9386d4b3bc5dcb09963fff0a2eb83037e44c9bcbd7338e2763e`.

The shared lock, source hashes, no-overwrite, disk and GPU admission gates
passed. At launch the only compute process was the explicitly admitted
`ToDesk_Session` at 331 MiB, and 10,537 MiB GPU memory was free. Training ran
once from 0 to 10,000, from `2026-09-19T14:18:26+08:00` to
`2026-09-19T14:31:58+08:00`. The upstream progress timer reported 566 seconds
for the optimization loop; the complete arm, including four saved-model
renders and evaluations, took 812 seconds. The fixed runner did not sample
training peak VRAM, so an exact peak cannot be reconstructed and is not
claimed. The interactive terminal output reported no OOM, NaN, Inf, traceback
or runtime error, but it was not preserved as a log artifact; completion,
fresh checkpoint reload and independent finite-value scans are the durable
evidence.

## Frozen RGB results

All entries are mean PSNR dB / SSIM / LPIPS-Alex over the same 74 validation
views:

- 1k: `17.610473 / 0.811226 / 0.468915`
- 3k: `18.938533 / 0.826853 / 0.409985`
- 5k: `13.853408 / 0.786179 / 0.513674`
- 10k: `14.275202 / 0.793896 / 0.483561`

At 10k, the separately aggregated worst-10%-view PSNR was `11.676307` and the
mean PSNR over 140.png through 146.png was `15.460300`.

The control's best recorded mean endpoint was 3k (`18.938533` dB), followed by
a sharp drop at 5k and only partial recovery at 10k. The protocol fixes the
decision at 10k, so this trajectory is retained as interpretation evidence and
is not used to select an earlier checkpoint after seeing the result.

## Engineering evidence

- Each 1k/3k/5k/10k native checkpoint restored into a fresh Standard model.
  Restored counts were 92,276 / 283,062 / 391,438 / 554,024 Gaussians; a
  recursive scan found every floating tensor finite.
- The evaluated saved PLYs contained 85,872 / 273,477 / 382,496 / 549,022
  Gaussians, with all floating properties finite. Upstream saves the PLY before
  same-iteration densification and the optimizer checkpoint afterwards, which
  explains the expected count difference.
- Every endpoint produced exactly the frozen 74-view render and metric set.
- The 10k saved PLY rendered official pose `140.png` through the external
  `CAMERA_FROM_WORLD` + `SIMPLE_RADIAL` interface without reading reference RGB
  or labels. The output is a nonempty 1320x989 uint8 RGB image.
- 10k checkpoint SHA256:
  `30b4a1d1aec643612e890af0719290e73ea7750f5b096c55d050074d58989d9a`.
- 10k evaluated PLY SHA256:
  `538400fe65a73f5b72cd9585e33458c5efca08f19d35d6e9f6153a91e1a35e20`.
- 10k per-view CSV / summary SHA256:
  `4d65a26c4108b3c9c8c26c4477c1e1a2793eb2ec2634b04f98fe910d5db4f679` /
  `04b44cc397f09f303e1bf5b80b6d4c6f9ce1114d760759d088350afd08600160`.
- External 140 RGB / metadata SHA256:
  `6ae8837834c33921ebf3a5457c0025e298f911718c8a3e678dae8b0d41901559` /
  `b1ecdbf9d590a2de657bc6b2652ddfb6d2f9b35b593955f0e03a4e57d654cc3b`.

This arm is the completed control only. The matched candidate result and the
pre-registered promotion decision are recorded in `GEO-027-STD-VGGT-10K`.
