# Spatial v2 paired semantic-seed replication

Registered on 2026-09-21 following explicit user authorization to execute the
recommended single additional semantic validation pair.

## Identity and scope

- AbsGS arm: `SEM-414-RGB154-SPATIAL255-SEED20260919`, parent SEM412/RGB154.
- Standard arm: `SEM-415-RGB155-SPATIAL255-SEED20260919`, parent SEM413/RGB155.
- Pair/audit: `QUAL-013-SPATIAL-V2-SEED-REPLICATION`.
- Manifest: `data/manifests/contest_dataset_v1.json`.
- Split: `spatial_confirmation_v2`, 349 RGB training views, 255 semantic training
  labels, 30 confirmation views and 21 buffers. RGB seed stays 0.
- Only the semantic seed changes from 20260805 to 20260919. The new arms share
  all initial tensors and all 6,000 view/crop/flip samples.

The current full-data system and both RGB endpoints remain immutable. This is
development replication of known methods on historically exposed data, not a
new blind split, a new model family, or an RGB-seed experiment.

## Fixed execution

Use the exact existing SEM412/413 runtime and SEM396 training implementation:
fresh ADE Swin-L, 768-pixel crops, native 768/512 sliding inference, the same
losses, AdamW, learning-rate schedule, AMP and exactly 6,000 successful updates.
The 3,000-step checkpoint is provenance only; 6,000 is the sole scored endpoint.
The existing exact-zero-mass FP32 tile recovery applies equally to both arms.

Prepare both arms using training RGB and training labels only. Validate the
registered split and source/config hashes, then execute eight discarded training
updates per arm. Before update one, compare all 650 initial tensors and the
pretrained-load report against the sealed SEM392 same-seed initialization
contract. Across seeds, only the two fresh classifier tensors may differ; the
old-seed full-state equality check is explicitly forbidden.

After preflight, train each arm from a fresh process and the original fresh
initialization. A pair seal must verify both completed 6,000-step results,
checkpoint hashes, identical initial states and identical actual sample traces
before either new confirmation inference is permitted. GPU stages serialize
through the shared lock and retain stdout, stderr, duration and exit status.

Verify the frozen RGB model files and all cached input hashes. Freshly render
001/120/300, all training cameras, with each original renderer before and after
the pair and require exact RGB equality against its saved output. No buffer or
confirmation images are used for these engineering probes.

## Fixed evaluation and interpretation

The primary semantic metric is pooled four-class foreground mIoU. Evaluate both
arms on the same 255 training and 30 confirmation labels after sealing both
endpoints. Report five-class mIoU, every class IoU/precision/recall and FP/FN,
stay_cable two-pixel boundary precision/recall/F1, each region, each view, cost,
and both semantic seeds separately and as a mean. No pixel-level significance
claim is made.

The replication direction passes only when **each** of seeds 20260805 and
20260919 has AbsGS-minus-Standard foreground mIoU > 0, stay_cable IoU >= 0 and
boundary F1 > 0, with all engineering checks passing. A positive two-seed mean
cannot conceal a negative second-seed direction. Other classes and regions are
descriptive and must retain negative results. This new foreground-first rule
does not retroactively change QUAL012's historical four gates.

Recompute all new per-view confusion matrices from saved PNGs with an independent
counting implementation and reconstruct pooled/region summaries. Independently
rebuild the inner four-neighbor cable boundaries and their two-pixel counts.
Verify all 379 predictions per arm, the 255 training-label identities, and the
sealed checkpoints. The first-seed predictions/results remain read-only.

## Failure handling and stopping

Preserve every failed attempt. Do not change seed, losses, capacity, crop,
thresholds, data roles, endpoint, or select the better seed after seeing results.
Engineering failures are recorded separately from quality outcomes. An
engineering recovery must preserve the scientific protocol and use a separate
attempt directory. A nonpassing quality result ends this validation; it does not
authorize a third seed, RGB retraining, Mip, ensemble or method search.

## Execution environment and estimate

Reuse the existing pinned CPython 3.12 / Torch 2.11 CUDA 12.8 environment used by
the parents, without installing packages or changing the host. The entrypoint
takes `--artifact-root`; source, weights and outputs are project/config-relative.
Estimate 70--100 GPU minutes and 4--6 GiB of new artifacts, based on the two
previous v2 fits; actual time, memory and exit statuses must be reported.
