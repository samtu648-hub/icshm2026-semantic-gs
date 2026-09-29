# RGB-151-STANDARD-ABSGRAD-LAS-DEV30

## Status

`completed_not_promoted` on 2026-09-17. The single fixed candidate completed 30k training and fresh 400-camera evaluation as `full-r2`; RGB150 remains the control and rollback. LAS improved all three mean metrics in the regular-local and oblique-local groups, but the distant-local group regressed in all three, including `-0.565968 dB` PSNR. The candidate is retained as a reproducible mechanism result and is not promoted.

## Problem diagnosis before implementation

The candidate addresses split footprint rather than candidate identity. Historical RGB141 finite removals showed that contributors around the difficult regions often affect 81--106 views and that removing one Gaussian can improve some views while damaging others. These are broad-footprint, mixed-responsibility observations; they do not justify deleting all elongated Gaussians. RGB144 found only one fixed cable-region proxy with a foreground contributor whose removal improved the local region, while 159 distant proposals were background and the water/most fine-cable evidence was sparse. It did not establish pure coverage absence or universal occlusion as the cause.

RGB146 then changed only some first-child XYZ locations for 1,922 events, 1.235% of 155,672 selected split parents. It reduced 159 distant-occlusion MAE by 20.404% and 159/160 `stay_cable`-region MAE by 12.413%/26.272%, but increased 159/160 near-field MAE by 14.614%/16.846%; overall PSNR changed by -0.004416 dB and the main streaks, water ghosts and trajectory occlusions remained. The evidence supports a uniformly applicable split-shape test, not a historical-ID fix or a claim that long Gaussians are erroneous. True `stay_cable` and bridge structure are explicit counterexamples: elongated support can be legitimate.

A new read-only shape check projects the eight RGB141 opacity-derivative-selected contributors into view 159 using the fixed PINHOLE camera and local covariance Jacobian. For the six shore/far/water artifact contributors versus the two true `stay_cable` contributors, median world max scale is 0.094042 versus 0.072577 (1.30x), projected major/minor sigma is 162.448/78.179 versus 69.605/44.547 pixels (2.33x/1.75x), and projected 3-sigma ellipse area is 436,387.9 versus 87,571.9 px2 (4.98x). Median world anisotropy is instead slightly lower, 3.922 versus 4.089. This supports testing excessive per-child scale/coverage, not using anisotropy or semantic identity as a selector. The sample has only eight deliberately selected contributors; the projection omits rasterizer low-pass/clipping and is a proxy, not a population distribution or causal result. Exact rows and input hashes are in `outputs/quality/RGB-151/shape-stats-r1/result.json`.

The fixed LAS geometry has a narrower per-child projected footprint. At `d/L0=0.45`, the child long/short scale ratios are 0.55 and 0.893029, the corresponding projected variance ratios are 0.3025 and 0.7975, and a single aligned child's projected ellipse area is 0.491166 of the parent. Two children at opacity multiplier 0.6 give a first-order opacity-weighted area ratio of 0.589399. Their combined outer long-axis 3-sigma extent remains exactly one parent extent because `1.35 + 3*0.55 = 3`. Thus the intervention reduces each child's central and transverse coverage without pretending to remove the entire legitimate long-axis support. These are analytical mechanism expectations, not rendered quality results.

`scripts/build_rgb151_split_diagnostic.py` creates a read-only compact sheet from the existing 159 near-field, distant-band and true bridge/`stay_cable` panels. The images are training-view historical evidence only; they are not RGB151 output or dev30 evidence and do not prove causality.

## Single candidate

For every large Gaussian already selected by RGB150's absolute-gradient and max-scale rule, choose local axis `a = argmax(scale)` and world direction `u = R e_a`. Replace the native random binary child shape by:

- centers `mu +/- 1.35 * scale[a] * u`;
- child longest-axis scale `0.55 * scale[a]`;
- child short-axis scales `sqrt(1 - 0.45^2) * parent_scale`;
- activated child opacity `0.6 * parent_opacity`;
- inherited rotation and SH features;
- the native two-child append, zero-initialized new Adam state and parent deletion.

The eligibility set, absolute-gradient threshold 0.0004, signed clone path 0.0002, losses, optimization schedule, pruning, opacity reset and SH schedule stay unchanged. The rule is uniform: no semantic class, historical Gaussian ID, anisotropy threshold, hand-selected region or proposal controls whether an eligible parent receives LAS.

This is an independent implementation of the Long-Axis Split shape equations from ImprovedGS, paper <https://arxiv.org/abs/2508.12313>, official repository <https://github.com/XiaoBin2001/Improved-GS>, commit `20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a`, under its non-commercial research license. It does not include ImprovedGS candidate sampling, importance budgets, clone replacement, RAP, MU or GC and must not be described as full ImprovedGS.

## Control and RNG contract

RGB150 is reused directly because `--split-shape native` is the worker default and does not import or install the candidate hook. The candidate uses the same manifest, fixed `dev30_local_composition_v1`, 370/30 access roles, training-only DLT initialization with 79,630 points, initial PLY SHA256 `30916591f71c5fb2af8b548c996bc1c036dec597cd81caba21cab2376ad8574e`, seed 0 and 30k recipe.

Camera sampling remains the upstream Python `random.shuffle`/`randint` sequence. The LAS hook does not call Python `random`. For every matching split topology it executes and discards the same-shaped native `torch.normal` draw before deterministic child construction. The CPU lifecycle test verifies the post-call Torch RNG state against that draw. Full 30k Torch RNG identity is not promised because changed child geometry can change later selected-parent counts and topology.

## Verification and commands

CPU checks cover multiple axis choices and rotations, finite positive scales, activated opacity, projected covariance/extent, explicit-off identity, parent deletion, two-child append, rotation/SH/radius inheritance, zero new optimizer moments and RNG consumption.

```bash
python3 -m unittest tests.test_standard_long_axis_split -v
"$R1_OUTPUT_ROOT/r1/envs/r1-cuda128/bin/python" scripts/diagnose_rgb151_shape_stats.py
python3 scripts/build_rgb151_split_diagnostic.py
python3 scripts/run_rgb151_split_shape.py --artifact-root "$R1_OUTPUT_ROOT" --stage verify
python3 scripts/run_rgb151_split_shape.py --artifact-root "$R1_OUTPUT_ROOT" --stage gpu-preflight
```

The actual Torch 2.11/CUDA 12.8 test run passed 4/4. The GPU preflight used the real Standard source and RTX 5070, passed the native signature/hook/rotation/finite-scale/CUDA-RNG checks in 0.924 seconds, and peaked at 8.134 MiB allocated/22 MiB reserved. The repository-default Python has no Torch, so the same four optional tests correctly skip there.

The registered training command and the successful preserved evaluation attempt were:

```bash
python3 scripts/run_rgb151_split_shape.py --artifact-root "$R1_OUTPUT_ROOT" --stage train
python3 scripts/run_rgb151_split_shape.py --artifact-root "$R1_OUTPUT_ROOT" --stage evaluate --evaluation-output evaluation-r3
```

The first `full-r1` launch safely stopped before the worker and before any optimization step because a just-finished SEM386 CUDA context was still visible after releasing the shared lock. Its only artifact is a `failed_preserved` result. The same scientific recipe was rebound as `full-r2`; CPU and GPU preflights were repeated before training. The first two evaluation directories also preserve zero-render engineering failures: system `python3` lacked Torch, then a resolved virtual-environment symlink was incorrectly treated as the same interpreter. `evaluation-r3` fixed only interpreter re-execution, rendered all 400 images, and left the endpoint hash unchanged. No failed attempt supplied a quality score or changed the model, data, seed, LAS parameters, or metric protocol.

## Results

- Training role, 370 views: PSNR `23.243955 dB`, SSIM `0.895333`, LPIPS-Alex `0.249396`; relative to RGB150: `-0.026544 dB`, `-0.000383`, `+0.001742`.
- Holdout, 30 views: PSNR `21.616533 dB`, SSIM `0.879414`, LPIPS-Alex `0.250775`; relative to RGB150: `-0.109522 dB`, `+0.001167`, `-0.001919`.
- Regular-local delta: PSNR `+0.143932 dB`, SSIM `+0.002593`, LPIPS-Alex `-0.003922`.
- Oblique-local delta: PSNR `+0.093470 dB`, SSIM `+0.001167`, LPIPS-Alex `-0.003102`.
- Distant-local delta: PSNR `-0.565968 dB`, SSIM `-0.000258`, LPIPS-Alex `+0.001269`.
- Of 30 holdout views, PSNR/SSIM/LPIPS improved on 19/27/24 respectively; 18 improved in all three and `182.png`, `183.png`, and `188.png` regressed in all three.

The fixed same-camera page at `outputs/quality/RGB-151/full-r2/review-r1/index.html` compares reference, RGB150, RGB151, and amplified candidate-control differences for nine post-hoc review views spanning historical and holdout contexts. It shows broad whitening/haze and lost bridge/water contrast in distant `188.png`, where PSNR/SSIM/LPIPS deltas are `-3.591201 dB`, `-0.021007`, and `+0.026941`. Improvements around some regular/oblique views do not offset this distant failure. This page is evidence review after the complete 30-view metrics, not a new score, parameter choice, or independent validation set.

## Execution evidence

- The native worker observed exactly 370 train and 30 evaluation cameras and loaded the registered 79,630-point training-only initialization.
- LAS ran on 144 native split calls and 233,108 selected parents. Camera sampling remained upstream Python `random`; the hook used no Python RNG.
- Final checkpoint: 30,000 iterations and 682,985 Gaussians, 58,972 fewer than RGB150 (`-7.948%`).
- Training wall time: `3025.220 s` (`50.420 min`), `+16.140%` versus RGB150. Worker peak allocated/reserved memory was `1884.202/5838.000 MiB`; reserved memory increased by 886 MiB.
- Evaluation rendered 400 images in `139.007 s`; median render time was `0.014385 s`. Peak allocated/reserved including LPIPS was `887.452/6906.000 MiB`.
- Candidate directory size after evaluation and review: `933 MiB`.
- Endpoint PLY SHA256: `99f9a7df8abefda996023dabcabcbb7a0be6abbccd4357d4f778d0f6d04887ee`; checkpoint SHA256: `1a03e70fa83e1ed2f78fec2869861dd375fc8621e11d66beb23396778ed5176b`.
- Training result SHA256: `044c50997876ba4c10a5291c154f42b0b63ebc235473d546f1fa6a9666c204d7`; completed evaluation result SHA256: `f0f81594eb380afd9f622dae51daa1cc9a68d5577f889ce1bd90ba5f6a990f96`; review result SHA256: `f4c3e13582c52def04ed9887421844ec61e1b1cbfc18722d94f79bbd221c74a3`.

Conclusion: the fixed LAS package trades fewer final Gaussians for slower training, higher reserved training memory, mixed overall holdout metrics, and a substantial distant-view regression. It does not replace RGB150, RGB140+SEM378, or RGB130+SEM371. A single development split and seed cannot establish statistical significance or hidden-test generalization; no parameter scan, second seed, all-data run, or matched semantic training is authorized from this result.

Final verification: the pinned Torch runtime passed 4/4 LAS tests, 243 JSON-compatible configs validated, Python compilation and `git diff --check` passed, the endpoint hash remained unchanged after evaluation/review, and `repo_health` passed 576 tests with 81 optional skips.
