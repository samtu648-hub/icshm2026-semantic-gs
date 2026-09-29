# Component sampling and DropGaussian: two independent extensions

2026-09-21: explicitly authorized by the user to complete both proposed lines,
in agent-chosen order, and report their results together. ADR149 supersedes only
the previous no-new-experiment queue restriction; frozen production artifacts
remain immutable. Configuration inheritance is deep-merged from `base_config`;
resolved configs and all inherited files are sealed before execution.

## A: SEM416 diagnostic, SEM417/418 paired training

Frozen RGB154 and spatial v2 (255 semantic train/30 confirmation/21 buffer).
Use the first two existing seed-20260805 sample-plan occurrences per training
image (510 crops), and fresh/3000/6000 SEM412 states. Only training images and
labels are accessed. Record actual final and auxiliary matcher/loss point
coordinates, including full-image eight-connected components before cropping;
a clipped large object is never reclassified as small. Log low coverage (<4
points), zero coverage and matched-query classification confidence. These are
fixed snapshot diagnostics, not historical training replay.

Training is supported if either late snapshot has >=30 low-loss-point small
component observations across >=10 training views and >=10% of observed small
components have <4 loss points in the final decoder. Report matcher coverage
separately; it cannot alone prove matching correctness. If this gate fails,
close A after its diagnostic rather than manufacture a training rationale.

The sole candidate replaces the last random points in each matched foreground
mask's native 12544 points with up to 16 distinct pixel centres from each
visible full-image component of area <=1024 pixels, at most floor(5%*12544)
points total. Round-robin allocation with a private seeded generator handles
caps. Keep the uncertainty-selected points, point count, matcher, class labels,
loss formulas/weights, optimizer, fresh ADE state, 768 crop/512 stride and 6000
successful steps fixed. This is supervision redistribution, not an unbiased
estimate of the original sampling objective. A disabled hook must reproduce
native loss, gradients and RNG; both arms consume native sampling first.

Fresh paired fits use seed20260805 and the same view/crop/flip plan. Auxiliary
losses use the same rule. AMP retries restore the quota RNG per successful step.
Only after both 6000 endpoints are sealed evaluate all 255 train +30 confirmation
views, all classes/FP/FN/regions/boundaries and small-component recall. Retention:
foreground mIoU > control, boundary F1 >= control, cable precision drop <=0.1pp,
every foreground class IoU drop <=0.5pp. Report all losses regardless of gate.

## B: RGB156/157 paired reconstruction, conditional SEM419/420

Spatial v1, 353 training images, 30 confirmation and 17 excluded buffers;
the exact RGB152 78045-point initialization, seed0 and 30k recipe are retained.
Independent implementation of the DropGaussian (CVPR2025) idea, not a full
upstream reproduction: `r(t)=0.1*t/30000`, independent Bernoulli keep mask, and
temporary activated opacity `keep*opacity/(1-r)`. Do not change stored opacity
or permanently delete points. Dropout is disabled in all ordinary evaluation.
Native alpha clipping is retained; opacity expectation compensation is not an
unbiased RGB guarantee. Temporarily dropped points have zero output radii and
are excluded from visibility/densification denominators. Pruning still reads
stored, unmasked opacity. Original native optimizer behavior is retained.

Use an independent Torch generator for masks; preserve Python camera RNG and
record all training camera names. Topology can diverge so full native RNG or
capacity equality is not claimed. Control installs the exact r=0 branch.
Preflight tests zero-rate RGB/grad equivalence, private RNG, fixed-mask replay,
finite gradients and masked contribution; each arm then completes an unscored
701-step engineering probe through native clone/split/prune. Discard probes and
train both fresh 30k arms. Test iterations remain disabled; save only the fixed
endpoint and seal both endpoints before scoring either holdout.

Report train/holdout and all regions, PSNR/SSIM/LPIPS, camera traces, point count,
time and VRAM. Only proceed to matched semantic fits if overall PSNR >= control,
LPIPS <= control, at least one strictly improves and no region loses >0.3dB
PSNR. Otherwise B closes as a negative RGB result with no semantic claim.
If eligible, SEM419/420 independently fit the unchanged Mask2Former recipe on
the respective frozen renders, seed20260805, 256/30 roles, 6000 paired updates.
Do not combine this with component sampling. Evaluate after both semantic
endpoints are sealed, with the same semantic retention rule as A.

## Evidence and recovery

Both known spatial folds are repeatedly used development data. No hidden-test,
novel-fold or statistical-stability claim is allowed. No scan, second seed,
checkpoint selection or combined candidate follows automatically. Preserve
failed outputs; narrowly scoped engineering fixes may resume unaffected stages
under separate attempt directories and an additional source binding. Never
rewrite completed scientific results or frozen baseline files.

Every stage records command, config/source/input hashes, seed/split, wall time,
exit status, peak memory and output path. Retain before/after RGB hashes and
three-camera exact replay for frozen semantic parents. Independently audit
saved semantic PNG confusion and boundary counts; recount RGB CSV summaries.
Use the existing pinned CUDA12.8 environment and GPU lock; no host installation.
Estimated 3--5 GPU hours and 10--18 GiB including conditional semantic fits;
measure actual resources. New run reports and registry entries are updated at
stage transitions and after both lines close, followed by repository health.
