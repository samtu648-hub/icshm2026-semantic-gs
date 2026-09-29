# SEM-263-NESTED-POINT-OOF

Status: completed_calibration_gate_failed; not promoted, matched development evidence retained.
Date: 2026-09-07. Parent SEM-262 remains failed at its original recall gate.
Formal RGB-030-FULL + SEM-161 and all SUBMIT-020 artifacts stay read-only.

The nested candidate recovers 0.981pp deck recall relative to its matched control,
but loses 0.076pp deck IoU to that control. Against SEM-160 OOF it gains 0.242pp
IoU while losing 0.403pp recall. Thus the strategy improves the recall trade-off,
not the complete acceptance criteria. No check/developer quality audit or final
training was triggered; no claim of surpassing the frozen SEM-161 final model.

## Hypothesis and controlled intervention

The SEM-261/262 correction head fits features/predictions of parent networks on
their own training images, then faces parent errors on spatially held-out images.
Test cross-fitted training evidence using TWO spatial inner folds inside each
of the THREE unchanged outer training sets. All three learned 2D parent branches
(five classes, stay_cable, deck) are independently trained from the original
pre-contest pretrained initialization, with original 3000-step recipes. No
contest-fitted outer or final weights may initialize an inner teacher.

Control and candidate heads share architecture 75-64-64-1, initialization, 3000
steps, optimizer, weighted BCE, sample coordinates and targets. The only arm
intervention is teacher-derived input source; both deploy with the same original
outer deck features and SEM-160 OOF baseline. To avoid arm-dependent sample
selection, BOTH arms sample the inherited static deck boundary and background
frame (static class2 becomes background for sampling only). This common sampler
is not an exact SEM-262 control replay; no historical result is overwritten.
Positive BCE weight is the same shared training-target ratio, not recalibrated.

Nested teacher training has 38-86 images per inner fold rather than full outer
training sets. Thus any difference tests this cross-fitting strategy, not pure
overconfidence causality independent of training-set size or feature-basis drift.
Static SEM-001 saw all219; the entire system is still incremental OOF. Existing
calibration/check partitions are reused development data, not a fresh blind test.

## Fixed gates and access

Deck IoU >= baseline+0.2pp; recall >= baseline+0.5pp; precision >=94%; total mIoU
nondecreasing; candidate IoU better than matched control; protected class2/3/4,
foundation rejection and RGB exact. Both background/deck directions are allowed;
no blanket old-deck protection or mask dilation. Seal all219 predictions and
configuration before current108 calibration scoring; on failure no111 check,
74 developer or final fitting. No seed, threshold, weight or checkpoint search.
Save source snapshots, parent provenance, inner/outer access logs, checkpoints,
paired feature/sample hashes, five-class results, panels/closeups and resources.
Run trained-head official COLMAP/native/small-camera engineering tests.

Maximum six GPU wall-clock hours, minimum100GiB disk free. Smoke uses only an
engineering budget, never a quality selection. Any incomplete run remains logged.

## Engineering execution

Two separate smoke outputs are retained. Both run three freshly initialized
inner branches for two steps each, two point heads for eight steps, and one
outer prediction without target scoring. r2 adds original-parent split-hash
verification and reduces library progress noise. Both passed teacher/head reload,
zero residual, frozen parent tensors, protected pixels, RGB and18 frozen hashes.
No quality-based recipe change was made. Full18-teacher training completed:
54,000 teacher steps plus six 3,000-step heads, generating 438 OOF predictions
(219 per arm). The two smoke runs and the failed calibration result are retained.

The standard repository-health environment passed. An additional broad suite
in the frozen model Python exposed unrelated existing test-environment issues:
OpenCV lacks ximgproc, pytest is absent (three test modules), and an old test
compares float32 0.40000000596 to Python0.4 using exact equality. These are not
SEM-263 training failures; production dependencies and unrelated tests were not
changed during the experiment. Targeted nested-OOF/refinement tests passed 11/11
in the frozen model Python; do not conflate these contracts with a universal
pass of all historical tests in that different environment.

Final `bash scripts/repo_health.sh` passed all seven stages: upload safety,
secret scan, Markdown links, 117 configuration files, shell syntax and
183 discovered unit tests (155 passed, 28 optional-runtime skips). The optional
in-repository dataset check is skipped because the configured dataset is
external. `git diff --check` also passes. No Git commit, reset or push occurred.

## Calibration results

All values below use the same reused 108-view calibration partition of the
219-view original three-fold spatial OOF. Percentages are pixel-aggregated;
delta units are percentage points, not relative percent. Five-class order is
background, deck, stay_cable, tower, foundation. Full confusion matrices and
per-view metrics for all three arms are in `calibration_sealed.json` and
`result.json`; no additional target labels were read for this report.

Access scope clarification: at final closeout, after the calibration failure
was sealed and the decision recorded, a read of the frozen manifest displayed
its embedded historical SEM-161 developer summary. No new 111/74 labels,
predictions or quality evaluation were opened for an audit, and that historical
summary did not drive this experiment's recipe or decision. Runtime access
flags describe the experiment scripts, not absence of all historical metrics
in repository documents or later manifest inspection.

- SEM-160 OOF baseline: deck IoU 94.530132%, precision 95.628069%, recall
  98.800008%; mIoU including background 86.879012%.
- Matched in-sample control: deck IoU 94.848871%, precision 97.296165%, recall
  97.416603%; mIoU 86.925306%. Relative to baseline: IoU +0.318739pp,
  precision +1.668097pp, recall -1.383405pp, mIoU +0.046294pp.
- Nested candidate: deck IoU 94.772442%, precision 96.258212%, recall
  98.397435%; mIoU 86.916060%. Relative to baseline: IoU +0.242310pp,
  precision +0.630144pp, recall -0.402572pp, mIoU +0.037048pp.
- Candidate versus matched control: IoU -0.076429pp, precision -1.037953pp,
  recall +0.980832pp. The new common sampler means these are not an exact
  replay or causal comparison against historical SEM-262 numbers.

Candidate five-class IoU / precision / recall (percent):

- background: 98.246912 / 99.493443 / 98.740823.
- deck: 94.772442 / 96.258212 / 98.397435.
- stay_cable: 89.636149 / 94.142319 / 94.930720.
- tower: 86.328338 / 88.865847 / 96.798254.
- foundation: 65.596457 / 67.635725 / 95.605586.

The latter three classes are exactly identical in baseline, control and
candidate, including all predicted-class pixel masks. Candidate deck counts:
TP 7,387,890, FP 287,185, FN 120,324; baseline TP 7,418,116, FP 339,142,
FN 90,098. Candidate adds 79,348 deck pixels and removes 161,531; 82,163 changed
pixels become correct, 106,128 previously correct pixels become wrong, and
52,588 are wrong-to-wrong changes. Both correction directions are actually used;
this failure is not caused by protecting every old deck pixel.

The preregistered IoU, precision, total mIoU, protected-pixel and RGB gates pass.
The recall gate (at least baseline +0.5pp) and the candidate-over-control IoU
gate fail. No threshold/seed/loss-weight/checkpoint search follows this result.

Calibration by original outer fold (46 / 42 / 20 views), derived from the
already sealed per-view confusion matrices:

- Fold 0: control IoU +1.366054pp, recall -0.029171pp; candidate IoU
  +0.957041pp, recall +0.365960pp versus its same-fold baseline.
- Fold 1: control IoU +0.807309pp, recall -0.981868pp; candidate IoU
  -0.241560pp, recall -0.404032pp.
- Fold 2: control IoU -2.482470pp, recall -4.575233pp; candidate IoU
  -0.230877pp, recall -1.819681pp.

Nested inputs mitigate the severe control degradation in fold 2, but the IoU
effect is not spatially consistent. This is useful matched evidence, not a
successful replacement or proof that overconfidence is the only root cause.

## Independent verification

`outputs/sem263/audit-20260907-r1/result.json` passed a separate audit of all
18 teacher checkpoints, six heads, 419 paired training views, 6,864,896 shared
sample coordinates and 438 prediction files. It validates original pretrained
initialization hashes, outer/inner train-held-guard isolation, actual access
and optimizer manifests, class counts/weights against the original label
manifest, cache/source/seal hashes, checkpoint reloads, and six zero-output
heads. Duplicate access and spatial guard violations are covered by tests.

The auditor independently reconstructs the protected set as old predicted
class2/3/4 plus background pixels rejected from static foundation. It verifies
all 219 native zero-residual predictions, exact protected-class masks, and
657 parent RGB/static/baseline PNG hashes. Shared RGB feature channels and
static one-hot channels are identical across arms. The positive BCE weights
are the same sampled-training negative/positive ratios in both arms:
2.642181 / 2.653607 / 2.651208 for folds 0/1/2, with no calibration fitting.
SUBMIT-020 remains 18/18 present and SHA-256 exact before/after and in the audit.

Label-free paired feature inspection finds fine-feature mean cosine similarity
0.873441 / 0.888134 / 0.843780, RMS difference 0.134992 / 0.122680 / 0.151663,
and coarse probability MAE 0.026457 / 0.029919 / 0.014723. This measures actual
representation differences; it does not establish that feature-basis drift
caused the quality result. Independent teachers see only 38-86 training images.

`outputs/sem263/external-20260907-r1/result.json` tests the actual trained fold0
nested checkpoint on official COLMAP camera140 (`COLMAP_CAMERA_FROM_WORLD`,
`SIMPLE_RADIAL`) at 1320x989 and 320x240. Frozen RGB/base semantic replay,
zero-head equivalence and protected pixels are exact at both sizes. The trained
head changes 2,569 / 550 pixels, demonstrating that the new branch really ran.
No GT or reference RGB is an inference input. This is one camera at two sizes,
not 140 cameras or final/hidden quality evaluation.

## Views and local inspection

Six preregistered views 028,295,162,283,282,268 have same-view RGB / GT /
SEM-160 / control / candidate / change panels and deck crops in
`outputs/sem263/full-20260907-r1/panels/`.
Twelve additional native 224x128 crops, uniformly enlarged 2x with nearest
sampling, are in `outputs/sem263/closeups-20260907-r1/`: one maximum-corrected
and one maximum-harmed region per fixed view, with crop coordinates/counts in
`manifest.json`. A maximum-corrected crop can still contain substantial harm;
all twelve are retained, not only attractive examples.

Inspection of 028_harmed shows persistent thin deck-edge losses; 162_corrected
also shows larger interior holes/erosion despite some corrected pixels.
Nested predictions sometimes erode less than the control, but are not uniformly
better than the baseline. Closeup green means newly correct and red means newly
wrong from a previously correct baseline pixel. Full-panel red additionally
includes wrong-to-wrong changes and must not all be counted as newly harmed.

## Resources and reproducibility

Git HEAD: `49cdcdb1c9915a9544d568754af95ea20a85ec9c`, with existing uncommitted
work preserved. Seed: 20260805. Dataset manifest:
`data/manifests/contest_dataset_v1.json`. Source snapshots accompany the run;
Git HEAD alone does not describe the dirty experiment code.

Outer split SHA-256:
`ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`.
Semantic split SHA-256:
`1764d3281fedc94e03c85f2ffc272e324bb12c0d8dba1dc1955078f71d147854`.
Preregistered config (kept unchanged, including its historical status field):
`acce89e57599fe6cf3aeda7af9b03c54d5cd6d6d8d6f7c7e8e541a42c6f5a347`.
Final result SHA-256:
`bc090faa4d8b18c2263343677f7bd5f501df6eaa0bba02fe63ede71642f68b37`.
The companion `artifact_hashes.json` indexes checkpoints, caches, sources,
prediction/calibration seals, audits and panels. Individual 438 prediction
hashes are transitively indexed by `prediction_manifest.json`.

Hardware: RTX 5070 12GB; Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8,
Transformers 5.5.3. Full run: 7,864.688 seconds (2.1846 GPU wall-clock hours),
including teacher training 7,008.383s, feature extraction 701.667s, both-arm
head training 16.087s and OOF inference 94.939s. Other time includes setup,
hashing and calibration. Independent audit: 8.908s; external audit: 15.535s,
both excluded from full-run time. Smokes/closeups are also outside that budget.
Peak teacher CUDA allocated memory: 5,683.518 MiB; peak OOF inference allocated:
2,843.138 MiB; external audit allocated: 2,517.521 MiB. These are PyTorch
allocated peaks, not total device/reserved memory. Each point head has 9,089
parameters. No pretrained parent/RGB/geometry/appearance/camera was modified.

Re-run only in a new empty output directory, with the configured model Python
and environment-selected roots (the completed directory must not be reused):

```bash
"$SEMANTIC_PYTHON" scripts/run_sem263_nested_point_oof.py \
  --artifact-root "$ARTIFACT_ROOT" --output-dir outputs/sem263/new-independent-run
```

## Decision and next direction

Do not promote SEM-263 or fit a final head. Keep its reduced recall loss and
fold2 mitigation as positive ablation evidence, alongside its failed overall
gate. The frozen RGB-030-FULL + SEM-161 remains the deployment candidate; rollback
requires simply not loading SEM-263 heads, not modifying or deleting artifacts.

The next worthwhile independent quality experiment is a native-resolution
foundation-versus-background specialist using the same frozen rendered RGB,
with a separately preregistered class-pair protocol. Existing calibration has
foundation IoU 65.596%, precision 67.636%, recall 95.606%; 447,569 of 462,591
foundation false positives (96.753%) are actually background. This is a larger,
more concentrated error budget than repeatedly rebalancing a deck baseline
already at 94.53% IoU and 98.80% recall. The existing foundation verifier uses
a five-class teacher, not a dedicated binary specialist.

That new line should permit real background/foundation corrections in its
research copy, including reconsidering historical foundation rejections;
protecting the target class everywhere would defeat its purpose. Deck,
stay_cable, tower and RGB should remain exact, with foundation recall and
all-class quality gates fixed before training. It is a proposal, not a run
or guaranteed gain. SEM-263 protection/gates and frozen SEM-161 remain unchanged.
