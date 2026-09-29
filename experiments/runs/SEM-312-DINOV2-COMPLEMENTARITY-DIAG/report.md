# SEM-312 — DINOv2 complementarity diagnostic

## Registration — 2026-09-08

Status: completed_read_only_diagnostic. Registration below preceded execution.

The user authorized continuation after SEM-311. This is a read-only diagnosis
of its sealed predictions, not a retry or a revision of its failed quality gates.
Parents: SEM-311, SEM-160 and SUBMIT-020. Base commit
`49cdcdb1c9915a9544d568754af95ea20a85ec9c`; source/config hashes will accompany
the run. Manifest `data/manifests/contest_dataset_v1.json`, inherited seed20260805
and `sem_005_oof_spatial_v1`; exactly the same reused108 calibration images.
No111/74 scoring, fitting, new inference, candidate export or model modification.

Before inspecting the new breakdowns, fix additions/removals as the two cohorts,
confidence bins0.5/0.6/0.7/0.8/0.9/0.95/0.99/1 and boundary distances4/8/16/32px.
Report useful, harmful and neutral edits separately, including true structural
pixels incorrectly predicted in the mutable0/4 lane. GT distances are explanatory
only; prediction distances/confidence/agreement are inference-available but not
assumed to generalize. Exact AUROC/AP use all edited pixels with tie handling,
not sampled pixels or significance tests. Check every spatial fold separately.

A direction motivates at most one newly registered auxiliary-evidence paired
experiment if corrected exceeds harmed in all three folds and aggregate corrected
pixels>=1000. This research screen is not promotion or proof of learnability;
SEM-311 remains failed. No confidence threshold or best bin will be selected.
Any follow-up must retain baseline and matched controls, frozen RGB/protected
classes, prior quality gates and independent checks. If the screen fails, pause
probability-level fusion rather than fitting on the reused108 diagnostic targets.

Rollback: omit all diagnostic outputs; the production candidate is untouched.

## Result

The exhaustive CPU run completed in7.675s. All108 per-view and aggregate original
confusion matrices and correction counts match SEM-311. The432 parent PNG/probability
files (two arms ×108 ×two formats),108 baseline/RGB fingerprints and18 frozen
artifacts were verified. No new inference, training, candidate PNG or111/74 labels.

- DINOv2 additions:478,709 pixels;23,170 corrected,453,554 harmed,1,985 neutral.
  Only4.8401% are correct additions. True-GT foundation distance>16px accounts for
  75,310/453,554 background additions (16.60%); most are near foundation, but
  baseline-edge proximity does not distinguish useful additions (AUROC0.5060).
- DINOv2 removals:94,073 pixels;72,554 corrected,15,508 harmed,6,011 neutral.
  Corrected fraction77.1252%; folds0/1/2 have corrected/harmed54,393/6,553,
  16,558/7,913 and1,603/1,042. The last fold has little margin:54.49% corrected.
- MiT removals:106,938 pixels;57,787 corrected,43,784 harmed,5,367 neutral;
  fold2 corrects2,402 but harms8,339. Its prospective direction screen fails.
- DINO removal-confidence AUROC0.6286 and AP0.8592 versus prevalence0.7713;
  fold AUROC0.6094/0.5739/0.7513. Confidence alone is not a demonstrated safe
  selector. Addition-confidence AUROC0.6820 overall but0.4661 in fold1;
  the aggregate rank conceals a spatial reversal. No best bin/threshold chosen.
- Agreement with MiT is not automatically more reliable: DINO-only removals
  correct46,789/harm6,209, while agreed removals correct25,765/harm9,299.
  These are descriptive cohorts, not an agreement-based inference gate.

Only DINOv2 removal passes the pre-fixed research screen. Register SEM-313's
minimal review mechanism instead of training a calibration-fitted selector.
This is selected-on108 development evidence, not a generalization guarantee.

Evidence: `outputs/sem312/diagnostic-20260908-r1/result.json`
SHA256 `d4f2f27b03de753358d10e630f11264583caf740022ff5e4b0cb5e563c6ccedb`;
config SHA256 `e1bc3b7547852ade0bde6c067062ddad34eb359b0c02f6a7ec33f77a1587e4c4`.
Five new diagnostic unit tests passed in the actual runtime, including exact-rank
comparison with brute-force pairs, ties, neutral errors and protected/replay guards.
Final repository health passed125 configs and219 tests (45 optional-runtime
skips); all21 targeted SEM312/313 and parent tests passed in the model runtime.
