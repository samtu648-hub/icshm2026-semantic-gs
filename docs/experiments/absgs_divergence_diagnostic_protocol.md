# AbsGS trajectory divergence and difficult-segment diagnostic

2026-09-21, authorized by the user to locate the cause of large reconstruction
divergence before proposing another reconstruction method. QUAL018 records the
diagnostic, RGB159/160 are identical seed0 native30k runs. Parent data/split are
RGB152 spatial v1:353 train,30 known confirmation,17 buffers excluded; the
78045-point training-only initialization is unchanged.

## Questions and scientific scope

1. Are actual loaded image tensors, camera sequence, initial parameters,
   optimizer configuration and RNG states equal?
2. At which observed update do numerical signatures first differ, and when do
   clone/split/prune choices or random split placements first diverge?
3. When does the known152--161 segment lose quality, and which Gaussian
   contributors/lineages produce the visible occlusion?
4. Does the screen-radius pruning signal survive the native densification
   lifecycle? The source resets max_radii2D in densification_postfix before
   final pruning, a concrete suspect to observe, not an authorized fix.

No optimization variable is changed and no semantic training is performed.
These are instrumented diagnostic runs on historically exposed development
data, not blind confirmations, new methods or checkpoint selection. Known
confirmation RGB may be scored at the fixed milestones solely to identify
failure onset; it never enters initialization, training loss or camera sampling.

## Fixed observations

The JSON config fixes state/checkpoint/probe steps. Record initialization and
early post-update XYZ/opacity signatures densely through1000 steps, sampled
gradient signatures, all30000 training camera names and loaded tensor hashes,
runtime flags and RNG. At each native clone/split/prune, record stable lineage
IDs, selected parents, signed/absolute gradient scores, split-sample hash/RNG,
point counts and pruning signals before and after append. Point lineage is a
sidecar only; it never reorders, filters or updates native parameters.

At fixed milestones save state/optimizer/RNG captures and native PLYs; render
the same9 probes and compute descriptive PSNR on saved8-bit images. Restore
Python/NumPy/Torch RNG after observation and require exact identity. Extra GPU
synchronization may affect floating-point atomic ordering; do not claim the
instrumented runs exactly replay historical ones. Both new runs use identical
observation schedules and neither is selected after seeing scores.

After fitting, load frozen milestone PLYs and measure native rasterized
contribution via unit-color gradients for156/159. Rank contributors without
labels, record their depths, projected radii, scales, opacity and ancestry. Test
temporary removal of top1/8/32 contributors without any optimization, restore
all parameters and require exact original RGB replay. This is a causal test of
their effect on those renders, not a deployable pruning policy or geometry GT.

From frozen state0 and599, repeat identical-view backward three times with
identical parameters and RNG. Compare forward outputs and gradient tensors,
without stepping an optimizer. Finite tiny differences demonstrate numerical
non-repeatability at that state; connecting them to trajectory failure requires
the independently recorded topology, RNG and contribution evidence.

## Interpretation and closure

Separate first observed numerical divergence, first discrete topology/RNG
amplification and first visible quality failure. If a causal link remains
unproven, state the boundary rather than attributing everything to CUDA or the
seed. Do not change pruning, gradients, opacity, initialization or camera poses
within this diagnostic. Keep every failed attempt, all historical outputs and
the RGB140/SEM378 production pair. Any future reconstruction fix follows a new
decision based on these observations.

Use the existing pinned CUDA12.8 runtime, shared GPU lock, original native
worker/data reader and source manifest. Seal source/config/data bindings before
running. Save full commands, stdout/stderr, runtime, VRAM and exit statuses.
Estimate2--3 GPU hours and15--30GiB; no host installation or source mutation.
