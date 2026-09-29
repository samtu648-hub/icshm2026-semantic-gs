# QUAL-014-COMPONENT-DROP-EXTENSIONS

Registered 2026-09-21 following explicit user authorization for both independent
extensions. Status: both methods and finite follow-ups completed; neither promoted.
Git base49cdcdb1c9915a9544d568754af95ea20a85ec9c
plus sealed working-tree files. Dataset manifest `contest_dataset_v1.json`;
A uses spatial v2 and semantic seed20260805; B uses spatial v1 and RGB seed0.

Execution identity, configs, commands, exit statuses and audit outputs are
retained in `outputs/quality/QUAL-014/r1`. Both production systems stay frozen;
failures are retained, no parameter search or combination is planned.
[Protocol](../../../docs/experiments/component_sampling_dropgaussian_protocol.md).

## Completed work and method decisions

**Component sampling:** SEM416 completed1530 training-only snapshot forwards.
Late final-loss low coverage(<4 effective points) is83/1529=5.43%, insufficient
for the original widespread-deficit gate; original SEM417/418 stay not run.
The user's further direct-experiment authorization led to separately registered
SEM421/422 with the same fixed quota. First-seed foreground/cable/boundary gains
were+0.261825/+0.585769/+0.576574pp, with all retention checks passing.
The worthwhile follow-up SEM423/424 changed only semantic seed and completed.
It returned-0.329470/-0.003865/+0.197429pp and foundation IoU-1.127419pp;
two-seed foreground mean-0.033823pp. Small-component recall improved in both
seeds, but overall utility did not repeat. Close the fixed recipe without
choosing the favorable seed or scanning quotas. See [QUAL015](../QUAL-015-COMPONENT-EXPLORATORY/report.md)
and [QUAL017](../QUAL-017-COMPONENT-SEED-CONFIRMATION/report.md).

**DropGaussian:** two701-step engineering probes and fresh RGB156/15730k arms
completed; zero-rate RGB/gradient, private RNG, fixed-mask and ordinary-render
recovery checks passed. Dropout changes holdout PSNR/SSIM/LPIPS by
-0.107260dB/+0.003063/-0.001686 versus the new matched control. The RGB gate
failed; SEM419/420 stay not run. Instead of rejecting a perceptual trade-off by
PSNR alone, the authorized QUAL016 probe ran120 predictions with two frozen v1
semantic models. Foreground changes-2.814603/-1.697606pp gave no downstream
training rationale. Close this fixed dropout recipe, retaining its SSIM/LPIPS
benefits and class trade-offs. See [RGB157](../RGB-157-ABSGS-DROP010/report.md)
and [QUAL016](../QUAL-016-DROPOUT-FROZEN-SEMANTIC-PROBE/report.md).

**Unexpected control variability:** RGB156(r=0) has18.391991dB confirmation PSNR
versus historical RGB15219.383911dB. A separately registered fully unwrapped
native RGB158 repeat yielded18.536542dB, also failing strongly on part of region1.
All four source manifests, initial/input PLYs and saved camera metadata match;
upstream CLI settings except output are equal. This is a full-trajectory
repeatability concern, not a proven CUDA/seed/initialization cause. It weakens
unqualified cross-run robustness claims without changing historical scores or
fixed-checkpoint replay evidence. See [RGB158](../RGB-158-NATIVE-ABSGS-REPLAY/report.md).

## Verification and resource evidence

- Three complete30k RGB runs(90000 iterations), four6000 semantic fits(24000
  successful updates),1402 RGB engineering iterations and1530 diagnostic crops.
- Component pairs:1516 saved predictions and1140 independently reconstructed
  confusion/boundary matrices across the two seeds. Frozen downstream probe:
  120 additional predictions/confusions. Complete fixed roles and label hashes
  checked; no hidden data accessed.
- Within each semantic pair,650 initial state tensors and6000 actual samples
  match; both endpoints sealed before scoring. All four new650-tensor semantic
  states and all28 tensors in each new RGB capture are finite.
- Before/after RGB154 replay at001/120/300 is pixel-exact for both semantic
  pairs. Original RGB140's6 recorded artifacts and SEM378's2 model artifacts
  remain hash-exact. No production replacement or source mutation occurred.
- All recorded experiment execution stages exited0, with full logs retained.
  A final CPU inspection initially rejected NumPy metadata in the native RGB
  captures under `weights_only=True`; after verifying artifact hashes, the
  trusted local native captures were loaded in their ordinary format. This
  inspection-only compatibility issue required no retraining or result changes.
- Aggregate recorded experiment execution time16076.730 seconds, approximately
  4h28m, excluding code preparation and queue waiting. New model/run directories
  total approximately10.8GiB logical size, plus diagnostic/summary evidence.
- Primary closeout: `outputs/quality/QUAL-014/r1/closeout.json`; before/after source
  check: `independent-record-check.json`. Per-stage commands, exit statuses,
  runtime, memory and source/config snapshots live under each QUAL/RGB run.

## Limits, rollback and next priority

All quality results use reused spatial development data and the internal
PINHOLE grid. No hidden-set score, statistical stability, full method-family
rejection or matched-DropGaussian-semantic score is claimed. The two methods
were never combined. The existing full-data RGB140/SEM378 and Standard fallback
remain available unchanged; rollback simply keeps using them.

There is no strong result supporting another immediate loss/backbone/rate
search. The higher-value remaining work is training-repeatability diagnosis,
the latest official-camera submission closure and honest paper evidence. No
additional training is queued after these completed follow-ups.

Final checks passed:304 JSON-compatible YAML configs,684 repository tests
(88 optional-runtime skips),6 targeted tests in the native Torch environment,
and `git diff --check`. All five finite execution services are inactive with
exit status0, and no GPU compute process remains. Closeout SHA256:
`bb2f332b9f6bf17fe5d1773269b7e822af35ae42d2a90c078c24381d9b703658`.
