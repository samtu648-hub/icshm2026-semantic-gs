# SEM-322 — Pose-only bounded source accessibility

## Outcome — 2026-09-08

`completed_access_screen_passed`, independently recomputed. All3 targets have
an accessible source within the fixed top3 budget; rank1 alone passes2/3 targets.
Six of the9 candidate pairs pass both original per-pair geometric gates. No
labels, semantic predictions, source unions, training or production changes.

All numbers belong to this experiment/config, base commit and seed/split listed
below. Coverage is the fraction of a512x512 grid, **not semantic-class coverage
or native-pixel accuracy**. Retention is epipolar inliers among cycle-kept points,
not precision against ground-truth correspondences. Every candidate is retained:

- 017→015, rank1:25677 reliable points,9.794998% coverage,67.425555% retention;
  fails retention. 017→014, rank2:19815 points,7.558823%,78.187271%; also fails
  retention. 017→013, rank3:18069 points,6.892776%,82.950007%; passes and is
  selected. This is the first passing rank, **not the highest-coverage source**.
- 001→103, rank1:16572 points,6.321716%,81.470921%; passes and is selected.
  001→102, rank2:14242 points,5.432892%,93.994192%; passes. 001→023, rank3:
  6416 points,2.447510%,94.159084%; passes.
- 032→029, rank1:8437 points,3.218460%,94.257625%; passes and is selected.
  032→095, rank2:2744 points,1.046753%,99.025623%; passes. 032→096, rank3:
  2377 points,0.906754%,99.539363%; fails coverage despite high retention.

Selected-source post-cycle median epipolar errors are0.466778/0.329030/0.435003
native pixels; same-support identity controls108.395733/171.109468/51.108714.
All3 selected pairs meet the fixed2x comparison. No threshold/angle/K changes.
Full ranked source lists have19/53/42 angle-eligible sources; only the sealed
first3 from each list were matched. No claim that all eligible sources were tested.

### What this changes, and what it does not

The previous SEM321 selector used target track IDs and yielded only7 reliable
points for001→046. A pose-only eligible first source001→103 now provides16572.
Thus the earlier result does **not** prove this target lacks usable source evidence;
source choice matters in this fixed small diagnostic. For017, the closest source
still fails, while the third meets the unchanged gates. For032, the first source
remains029. The overlapping017→015 and032→029 forward/reverse arrays are bit-exact
to SEM320, supporting unchanged matcher behavior rather than implementation drift.

This is positive source-accessibility evidence, not a production or semantic gain.
Targets are the same3 reused calibration views, not a new blind sample. Pose
ranking and the candidate budget are a combined policy, not a pure causal ablation.
Some passing retention rates are close to80%; reliable coverage remains partial.
Geometry can match the wrong point on an epipolar line or concentrate on background.
No class-conditional coverage, transferred-label accuracy, mIoU, occlusion correctness
or arbitrary unseen-camera generalization was measured.

### Engineering and isolation

- Pose parser consumes observation lines without parsing keypoints/track IDs;
  empty and arbitrary synthetic observation lines leave retrieval unchanged.
  Independent inverse4x4 camera transforms reproduce the full source ordering,
  angle eligibility and top3 exactly; numeric pose metrics agree within1e-6.
-15 project/config source files were sealed before inference, plus both pinned
  upstream trees and official complete weight/hash inherited from SEM320.
  All9 pairs were sealed before matching; no GT or model scores ranked candidates.
-18 forward/reverse calls completed; one fresh-process first-forward replay is
  bit-exact. Runtime stage8.118s (model construction2.404s), replay4.987s;
  CUDA peak allocated2508.446MiB, reserved2718MiB. These are allocator figures
  for fixed512, not total-device memory or native-resolution/end-to-end benchmarks.
- Primary geometry1.668s; independent CPU Torch sampling/radial fixed-point inverse/
  essential-matrix audit2.472s. All9 support counts,4x4 coverage cells, decisions
  and selected ranks match; medians/P90 agree within2e-5 native pixels.
- Source images013/103/023/029 have historical check IDs but are explicitly
  permitted same-fold training-source RGB, never check targets. No source/target
  labels, real target RGB, developer/guard/unlabeled RGB were decoded. No novel
  camera was rendered or semantic output created. Camera coordinates stayed official.
- Selected rendered/source RGB bytes and arrays remain exact; all18 frozen
  production artifacts pass before/after. Upstream readonly-array warning remains,
  with hashes verifying no mutation. Original runtime dependencies are unchanged;
  inherited isolated Pillow12 overlay only. Ordinary sandbox GPU probe could not
  access the driver; authorized host GPU execution succeeded, no driver edits.
- New SEM322 tests6/6 and combined SEM320/321/322 tests14/14 pass in the
  existing training environment. Repository health passes129 configs and239
  discovered tests (47 optional-runtime skips); the default-path full dataset
  scan is skipped, while actual selected inputs/cameras were hash-verified.
  `git diff --check`, registry/result hashes and28 SEM320/321/322 sealed project
  source checks pass. Original environment Pillow remains10.2.0. No SEM322
  failure files exist; every passed/failed candidate-pair measurement is retained.

### Decision / rollback

Provenance SHA256 (all result paths below are under `outputs/sem322/`):

- Config: `83cfa4c978ea325f4006342db852fdc33ed1851806f24d38a4943dd25e24a398`.
- `prepare-20260908-r1/result.json`: `cdaf157636e6f58c439ab228ffe729efb991e89bc24079898b263acc6c37e348`.
- `runtime-20260908-r1/result.json`: `5d628c7ac3759417323894133f1b6d97ea2ce05509a718364181251d1588cbf5`.
- `replay-20260908-r1/result.json`: `ac0225d36ead25b2832e636d0dc5cd2ca59344f3a063b4b746e76e1d522476a0`.
- `geometry-20260908-r1/result.json`: `a7226ec46b3c544175233a692b0d3d34a5402af7a7e78854623bdf1dbebd136a`.
- `audit-20260908-r1/result.json`: `2eb4e750cead6290624cd118dbcbb3ed43d8eae5140cf0bafc270bb438f1e246`.

Keep this fixed pose-top3 plus first-geometrically-passing retrieval policy as a
research candidate. Do not modify SEM161 or start label transfer. Next, separately
register a bounded expansion to more preselected training-side targets, with the
same retrieval and filters, to test spatial consistency; any class/label diagnostic
needs its own explicit data-access registration. Do not tune K/angle/filter limits
on these3 targets. Final all400 RGB/all300 label fitting and paper/slides remain
after method selection.

RGB-030-FULL+SEM161 and the existing bundle are untouched. Rollback is simply
not invoking this optional diagnostic. All outputs and any failures are retained
under Git-ignored `outputs/sem322`; no Git commit/push or official upload.

## Registration — 2026-09-08

Preregistered before source ranking, RGB decoding or matching. User authorized
the next bounded source-accessibility diagnostic, not final fitting or production
replacement. Parents SEM321/SEM320/SUBMIT020; base commit
`49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus sealed sources; seed20260805;
manifest `data/manifests/contest_dataset_v1.json`; spatial split
`sem_005_oof_spatial_v1`. Same three historically reused calibration targets,
not a fresh blind sample.

Only intervention: a fixed pose-only source retrieval policy with a three-candidate
budget. Same-fold training sources with optical axes within60 degrees are ranked
by center distance, then angle and filename. Target observations/track IDs are
not parsed or passed into retrieval. Report rank1 separately from the existence
of an accessible source in top3; this is not a pure ranking-versus-budget ablation.

Reuse fixed RoMa fast512 and SEM321 native-pixel geometry filters. Every candidate
is evaluated and retained; no union of matches, labels, semantic output or fitting.
An accessible pair needs>=1% reliable grid coverage and>=80% post-cycle epipolar
retention. All3 targets need such a source, selected as the first passing rank;
at least2 selected pairs must beat the same-support identity median by2x. Fewer
than3 eligible candidates fails the bounded source screen. No gate or ranking
sweeps after results. This is necessary geometric evidence, not semantic accuracy.

Before GPU inference: seal full ranked candidate lists, selected input hashes,
all project source dependencies, configuration, official source/weight hashes and
18 frozen artifacts. Independent pose selection and independent geometry audit,
plus one new-process forward replay, are required. Historical check IDs may be
training-source RGB within the correct fold; no check targets or any labels.

Budget30minutes and100GiB disk floor. Failure retained, no automatic rescue.
Formal RGB-030-FULL+SEM161 and its bundle remain untouched. Rollback consists of
omitting this optional diagnostic. No full-data final, paper/slides, upload or push.
