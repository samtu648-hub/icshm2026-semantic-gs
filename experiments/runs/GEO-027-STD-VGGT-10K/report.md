# GEO-027-STD-VGGT-10K

Status: `completed_not_promoted`.

## Contract and execution

The full GEO-026 audit passed, fixed operational-v2 fusion produced 100,000
points, and this candidate scene changed only `sparse/0/points3D.ply`. Its
initialization SHA256 is
`b1debba524689d1cacbb22d4379fa8402b054d9143f9f848e1f22b1bb7059cbd`.
The matched Standard commit, source tree, 303/74 split, seed, cameras, images,
loss, optimizer, densification and evaluator are identical to the control. The
execution-time config SHA256 is
`ffd75cec86a103578980f271e113252c3771d935895ea6af0a8c6922e622fd60`.
The repository preparation commit was
`49cdcdb1c9915a9544d568754af95ea20a85ec9c`; the exact working-tree diff,
dataset reader, train.py, render.py and submodule bindings are the same hashes
listed in the control report and in this execution config.

The shared lock and all launch gates passed under the same ToDesk-only
exception as the control. Training ran once from 0 to 10,000, from
`2026-09-19T14:31:59+08:00` to `2026-09-19T14:44:49+08:00`. The upstream
progress timer reported 562 seconds for the optimization loop; the complete
arm, including four saved-model renders and evaluations, took 770 seconds. The
fixed runner did not sample training peak VRAM, so an exact peak cannot be
reconstructed and is not claimed. The interactive terminal output reported no
OOM, NaN, Inf, traceback or runtime error, but it was not preserved as a log
artifact; completion, fresh checkpoint reload and independent finite-value
scans are the durable evidence.

## Frozen RGB results

All entries are mean PSNR dB / SSIM / LPIPS-Alex over the same 74 validation
views:

- 1k: `14.554529 / 0.782776 / 0.562157`
- 3k: `15.683631 / 0.792263 / 0.532522`
- 5k: `15.559680 / 0.794793 / 0.511185`
- 10k: `15.404514 / 0.796857 / 0.488015`

Candidate-minus-control mean trajectories were:

- 1k: PSNR `-3.055944 dB`, SSIM `-0.028449`, LPIPS `+0.093243`
- 3k: PSNR `-3.254902 dB`, SSIM `-0.034590`, LPIPS `+0.122537`
- 5k: PSNR `+1.706271 dB`, SSIM `+0.008614`, LPIPS `-0.002489`
- 10k: PSNR `+1.129312 dB`, SSIM `+0.002961`, LPIPS `+0.004454`

The 10k PSNR primary gate passed and the SSIM guard passed. The candidate's
worst-10%-view PSNR was `11.980698`, `+0.304391 dB` relative to the separately
aggregated control worst 10%, so that guard also passed. However, mean PSNR on
the fixed 140.png through 146.png subset was `13.230386`, versus `15.460300`
for control, a `-2.229914 dB` regression against the allowed `-0.1 dB` floor.
The LPIPS mean was also worse by `+0.004454`. Therefore the pre-registered gate
fails and this branch is not promoted.

The candidate was behind at 1k/3k and moved ahead in overall PSNR only after
the control's sharp 3k-to-5k degradation. Its own best recorded mean PSNR was
also at 3k (`15.683631` dB), below the control's 3k `18.938533` dB. The fixed
140--146 PSNR deltas at 1k/3k/5k/10k were `-6.568716`, `-5.877457`,
`-1.775230`, and `-2.229914` dB, so none of the saved endpoints offers a
post-hoc hard-view rescue; only 10k remains the registered decision endpoint.

The paired 10k PSNR bootstrap mean 95-percentile interval was
`[+0.702917, +1.542906] dB`; this is descriptive only and does not override the
fixed hard-view guard. These are single-seed results on a previously used
development split, not hidden-test or semantic evidence.

## Engineering evidence

- Each 1k/3k/5k/10k native checkpoint restored into a fresh Standard model.
  Restored counts were 114,430 / 253,499 / 370,008 / 562,319 Gaussians; every
  floating tensor was finite.
- The evaluated saved PLYs contained 109,955 / 245,678 / 361,702 / 557,033
  Gaussians, with all floating properties finite. The PLY/checkpoint count
  difference follows the same upstream save-before-densify ordering as control.
- Every endpoint produced exactly the frozen 74-view render and metric set.
- The 10k saved PLY passed the official external `140.png`
  `CAMERA_FROM_WORLD` + `SIMPLE_RADIAL` rendering gate with a nonempty 1320x989
  uint8 RGB image and no reference RGB or label access.
- 10k checkpoint SHA256:
  `2c69d962fb2c05f2ec431b55bfc3fb8fa7adc3459e875205fc5347517a65e54c`.
- 10k evaluated PLY SHA256:
  `9e7dbd1d8d600ceca394ebfef54374cc09dc6d237e44c888d156d5cbedcb6d13`.
- 10k per-view CSV / summary SHA256:
  `ffee0b5b4a7353acf60a90fab16a37b8c7f7174feab9ebab258c7db61bef1d21` /
  `e2e5fd5af8fed07863ffdaca3f55ada0fa0e170b3547d1ef43fb36b146ea181d`.
- Deterministic 10k comparison SHA256:
  `b06161baec90363017e9ea023c71de31115d6ef3a24e52753b894255d5f132b9`.
- External 140 RGB / metadata SHA256:
  `3e9c34651c5e0030e5dd305c3e1790513486af066b59b6818ba37f2fa6609bb7` /
  `1ed7ffa24566d3aa3a541ee0b4b683cc4598baa8ff467f30b8f933feff0d6d93`.

No matched 30k run or Mask2Former training is authorized. GEO-025-P0 remains a
separate preserved numeric-gate failure, and no semantic conclusion is drawn.
