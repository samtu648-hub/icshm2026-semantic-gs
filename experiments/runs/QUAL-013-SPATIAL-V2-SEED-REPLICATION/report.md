# QUAL-013-SPATIAL-V2-SEED-REPLICATION

Completed on 2026-09-21; status: `completed_replication_passed`.

This audit binds SEM412/413 (seed 20260805) and the newly authorized SEM414/415
(seed 20260919) on frozen RGB154/155 and `spatial_confirmation_v2`.
The primary metric is foreground mIoU. Each seed must also retain nonnegative
stay_cable IoU and positive 2px boundary F1 differences for the replication
direction to pass. Mixed class/region results will remain visible.

Config and frozen hashes: `config.yaml`. Pair execution logs and seal:
`outputs/quality/SEM-414-415/r1`. Independent audit:
`outputs/quality/QUAL-013/r1`.

Protocol: [paired seed replication](../../../docs/experiments/sem414_sem415_spatial_v2_seed_protocol.md).

The paired eight-update preflight completed in 14.252 seconds; both 650-tensor
initialization checks and train-only sampling checks passed. Both original RGB
renderers reproduced 001/120/300 exactly before fitting. Four targeted contract
tests passed in both lightweight and native Torch environments; 288 configs and
diff syntax passed. Full training/evaluation and the independent audit subsequently
completed, serialized by `icshm-sem414-sem415-v2-seed.service`.

## Primary result: the three overall directions repeat

| Confirmation delta, AbsGS minus Standard | Seed 20260805 | Seed 20260919 | Two-seed mean |
|---|---:|---:|---:|
| Foreground mIoU | +0.594970 pp | +0.246371 pp | +0.420671 pp |
| Five-class mIoU | +0.501025 pp | +0.205683 pp | +0.353354 pp |
| stay_cable IoU | +2.917695 pp | +0.940255 pp | +1.928975 pp |
| stay_cable precision | +3.369702 pp | +1.055615 pp | +2.212658 pp |
| stay_cable recall | -0.025657 pp | +0.005470 pp | -0.010093 pp |
| 2px boundary F1 | +5.405989 pp | +1.466619 pp | +3.436304 pp |

All six pre-registered per-seed direction conditions pass, together with the
engineering audit. The second seed's absolute foreground mIoU is
87.404296%/87.157925% (AbsGS/Standard); the two-seed means are
87.502486%/87.081815%. These are averages of each seed's pooled scores, not
scores of a prediction ensemble.

The second seed reduces cable FP by 17,139 and FN by 71; tower IoU declines
0.173044 pp. Its regional foreground differences are
-0.377754/-0.115981/+1.010405 pp, and cable IoU declines in regions 1/2.
Both seeds improve boundary F1 in all three v2 regions. Across seeds the
regional foreground means remain -0.882111/-0.053023/+1.395613 pp, so the
overall mIoU advantage still depends on regional composition.

Decision: retain the quality-oriented AbsGS choice and Standard efficiency
fallback. The previously missing second semantic seed on v2 is now complete;
v1 and v2 each have two semantic seeds, but RGB remains seed 0. This is evidence
for repeated overall direction and a particularly useful boundary trade-off,
not statistical significance, uniform regional superiority, blind performance
or independent RGB-seed stability. No third seed, new backbone, ensemble or
full-data retraining is authorized by this result.

## Verification and resources

- Both arms used fresh ADE initialization and the exact sealed SEM392 same-seed
  650-tensor contract; all 6,000 actual sample records match between arms.
- Both step-6000 checkpoints were hash-sealed before any new confirmation
  inference; hashes remain identical after scoring.
- 379 predictions per arm completed, 758 prediction files hash-verified.
- 285 per-view confusion matrices per arm independently rebuilt from PNGs,
  570 total; pooled/region metrics and four-neighbor 2px boundary counts exact.
- RGB154/155 model and cached-render hashes unchanged; each original renderer
  reproduced 001/120/300 before and after with pixel maximum difference 0.
- All 13 execution stages exited 0. No failed attempt or exact-zero-mass FP32
  inference retry occurred; each full fit had five standard AMP overflow retries.
- Training loops: 1,764.948/1,776.611 seconds. Training allocated/reserved
  peaks: 8,097.570/8,588 MiB for both arms.
- Full orchestration: 4,099.419 seconds (68.324 minutes), including preflight,
  both fits, inference, RGB checks and the independent audit. GPU is released.
- Approximately 3.9 GiB of new arm outputs plus 4.3 MiB execution/source evidence.
- Four targeted contract tests passed in lightweight and native Torch runtimes;
  final repository health passed 288 configs and 678 tests (87 optional runtime
  skips), with `git diff --check` passing after the experiment-record updates.

## Evidence and rollback

- Audit result SHA256: `617e45464f0fbc97cf3b28cdf246cacbb8a8e3c38c98f0f19c062c34ea6c6dab`.
- Pair execution result SHA256: `c3529b26d9a95316b3a0f2dc983ba637b29f008b2d61073a83846ca0a90ace66`.
- Pair seal SHA256: `82fe25b8fe16fd2740a117dcab88505d438eaf4532e741efad13af57142acf27`.
- Frozen execution code/config snapshot: `outputs/quality/SEM-414-415/r1/source`.
- Complete per-view seed comparison: `outputs/quality/QUAL-013/r1/per-view.csv`.

Each arm's report contains checkpoint and result identities. The original
RGB140/SEM378 full-data artifacts, RGB154/155 and first-seed SEM412/413 are
unchanged. Rollback requires no file restoration: continue using the retained
full-data system, ignoring these validation checkpoints. Official-grid/latest
submission reproduction and the English paper/video remain separate tasks.

Final closeout independently CPU-loaded both new checkpoints: all 650 tensors
per model are finite, step counts are 6000, registry metrics/hashes match the
machine records, and all retained RGB140/SEM378 parent model artifact hashes
are exact. Evidence: `outputs/quality/QUAL-013/r1/closeout.json`. The two arm
directories contain 4,056,131,048 logical bytes (approximately 3.778 GiB);
filesystem allocation rounded by `du` is approximately 3.9 GiB.
