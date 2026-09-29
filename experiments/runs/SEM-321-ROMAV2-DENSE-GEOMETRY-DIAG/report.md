# SEM-321 — Read-only dense geometry diagnostic

## Outcome — 2026-09-08

`completed_geometry_screen_failed`. Independent geometry recomputation passes and
confirms the failure. No new model inference, RGB/label decoding, semantic export,
threshold sweep, source reselection or production change occurred in this ID.

All metrics are for these three fixed calibration target pairs under the registered
config, base49cdcdb1c9915a9544d568754af95ea20a85ec9c plus sealed sources,
seed20260805 and `sem_005_oof_spatial_v1`, with the released dataset manifest.
Counts are **points on a512×512 grid**, not native-image pixel counts; cycle and
epipolar distances are measured in native1320×989 pixel units. All pair selection
and filters were fixed before scoring; no best-pair selection in the summary.

- Fold0,017→015:60371 both-confident points →38082 pass2px cycle →25677 pass2px
  epipolar. Reliable grid coverage **9.794998%** passes1%, but epipolar retention
  after cycle is **67.425555%**, below80%. Native cycle median/P90 on confident
  points1.481128/4.856121px; epipolar median/P90 after cycle0.788169/5.101392px.
- Fold1,001→046:16339 both-confident →11 cycle-consistent →7 reliable points.
  Coverage **0.002670%** fails1%; retention **63.636364%** fails80%. The11-point
  denominator is very small and not stable accuracy evidence. Native cycle
  median/P90 on confident points **93.391717/175.255154px**; a high forward
  confidence fraction alone would have badly overstated usable coverage.
- Fold2,032→029:22664 both-confident →8951 cycle-consistent →8437 reliable.
  Coverage **3.218460%**, retention **94.257625%**: both pass. Native cycle
  median/P90 on confident points2.828108/11.902053px; epipolar median/P90 after
  cycle0.435003/1.524120px. This is positive local geometric evidence, not semantic
  correctness or proof of generalization across all target views.

On identical post-cycle supports, identity-control median epipolar errors are
79.225160/35.876878/51.108714px; RoMa medians are0.788169/1.596245/0.435003px.
All three beat the weak identity control by more than2×. That sanity result cannot
override coverage/retention failures; only fold2 satisfies both per-pair gates.

### Independent check

The primary implementation uses NumPy bilinear sampling, OpenCV source-coordinate
undistortion and a pixel-space fundamental matrix. The independent audit uses
CPU Torch float64 `grid_sample`,15 radial fixed-point inverse iterations and a
normalized-camera essential matrix from relative4×4 world-to-camera transforms.
It does not call the primary numeric helpers. All valid/confident/cycle/reliable
counts, fixed4×4 spatial coverage cells and gate booleans match exactly; median/P90
values agree within2e-5 native pixels. The independent decision remains failed.

Official source radial k1=0.008987863345268233 is applied only to real source
coordinates. Rendered target coordinates are pinhole. Pixel centers are edge-origin
.5; array interpolation accounts for the .5 shift separately. The audit verifies
camera/input hashes and all18 frozen production artifacts. No RGB files are
decoded by this experiment; source and target masks/labels are never opened.

Primary run2.054s; independent audit1.981s. New geometry tests4/4 and preflight
tests4/4 pass. Repository health passes128 configs and233 tests with46 optional
runtime skips; optional default-path dataset scan skipped. `git diff --check`
passes. Source snapshots and unchanged original SEM320 config/results are retained.

Git-ignored result provenance:

```text
config
  23672a35c19139ca6000598ac4cb29fad9140ea5e0af080a0962685a9b15d9cc
outputs/sem321/geometry-20260908-r1/result.json
  02a1dfc068a6ee3c9078486166bc6f9fbfdf57cce7dedfb4caae60b7ab96cb22
outputs/sem321/audit-20260908-r1/result.json
  e4fb432d8b38ca85f8567a49d6f4cdeabbbab561659c5feebcb37d5cf711e334
```

### Decision and limitations

Do not start dense label transfer or replace SEM161 from these results. Preserve
the working512 implementation and fold2's positive geometric evidence; stop this
fixed single-source screen without weakening thresholds or increasing resolution
to rescue it. The known-camera source selector used released target track IDs;
arbitrary novel cameras do not supply those IDs, so source retrieval also remains
an unsolved deployment prerequisite. A later source-accessibility diagnostic would
need a separate ID and explicit pose-only/rendered-input retrieval rules; it has
not been executed and is not claimed to solve the observed gap.

Geometry agreement can occur along the wrong point on an epipolar line and can be
concentrated on unhelpful background. There is no class-conditional coverage,
ground-truth matching accuracy, source-label precision or semantic mIoU evidence
in this ID. Three pairs are insufficient to reject all RoMa/dense methods or to
prove low co-visibility is the sole causal failure. It is sufficient to reject
automatic progression from this small screen to semantic transfer.

Keep RGB-030-FULL+SEM161 and its reproducible bundle intact. Rollback is simply
not using these optional diagnostic scripts; no model or bundle restoration is
needed. All outputs, including the failed screen, remain local and Git-ignored.

## Registration — 2026-09-08

Original pre-run registration before any cycle/epipolar scoring. Parents SEM320/SUBMIT020,
base49cdcdb1c9915a9544d568754af95ea20a85ec9c, seed20260805, released dataset
manifest and existing `sem_005_oof_spatial_v1`. No new matching, labels or RGB
decoding. Reuse only the six forward/reverse maps for017→015,001→046,032→029.

SEM320 passed model/source/runtime/replay engineering but its access-policy prose
overstated exclusion of111 check identifiers:046 and029 were read as same-fold
training-source RGB, never as held-out targets and never as labels. This is
explicitly documented, not retroactively erased from its immutable config.
SEM321 accepts these existing same-fold source observations under its own explicit
scope; it cannot claim all111 source images were untouched.

Freeze overlap>=0.5 both directions, native target cycle<=2px and ideal-pinhole
symmetric epipolar distance<=2px. RoMa's edge-origin half-pixel centers are kept;
source SIMPLE_RADIAL coordinates are undistorted, target rendered coordinates are
already pinhole. Original official COLMAP world-to-camera poses define geometry;
no fitting from matches or RANSAC. Evaluate a no-motion normalized-coordinate
identity map on the same support as a sanity control.

Prospective small-sample screen: every pair reliable coverage>=1% of the fixed512
grid and epipolar retention>=80% after overlap/cycle filtering; at least2 pairs
halve the identity-control median geometric error. No threshold scan. Passing only
supports considering a larger or source-label diagnostic; geometry consistency is
not actual correspondence truth, semantic precision, candidate promotion or
authorization for label transfer. Failure preserves evidence and stops this screen.
