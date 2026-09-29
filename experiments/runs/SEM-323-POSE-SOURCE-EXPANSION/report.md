# SEM-323 — Fixed-policy expansion to105 calibration targets

## Outcome — 2026-09-08

`completed_access_screen_passed`, independent315-pair audit confirmed. Fixed
policy finds accessible sources for91/105 targets (86.667%); rank1 alone86/105
(81.905%).242/315 pairs pass the original1% grid coverage/80% retention pair
gates. No source reselection, K/angle/filter tuning or model changes.

- Fold0:43/45 accessible, rank1=42/45; selected coverage median including
  abstentions6.753922%.
- Fold1:39/41 accessible, rank1=38/41; coverage median25.635910%.
- Fold2:9/19 accessible, rank1=6/19; coverage median0% because10 targets abstain.

All-target selected coverage P10/median/P90:0%/6.753922%/28.059998%. Failed
targets retained:024,025,014,015,040,043,051,162,163,236,239,240,243,244.
These are105 previously untested-by-this-matcher calibration targets, not fresh
blind semantic validation.315 pairs use41 unique real-source RGB files;
145 source observations have historical check IDs, explicitly allowed as same-fold
training sources. Some names serve as rendered targets in other folds; real RGB
is only used in the source slot of eligible folds. SEM323 decoded no labels.

630 directional matches completed in124.292s including load/I/O/compression;
CUDA allocated peak2508.446MiB, reserved2718MiB. One fresh-process first-forward
replay exact (5.319s); primary geometry38.851s, independent audit49.487s. Full
pose rankings,315 support counts/4x4 maps/decisions and source selection agree.
All18 frozen production artifacts and selected RGB byte/array hashes unchanged.
Existing source/kernel/Pillow overlay reused; no environment edits or downloads.

Validation after SEM324:19 combined SEM320–324 tests pass in the existing runtime;
repository health passes131 configurations and244 discovered tests (48 optional
runtime skips). Default-path full dataset scan skipped; actual selected inputs
and cameras were hash-verified. No SEM323 runtime failure files; all failed pair
measurements and14 target abstentions remain in the sealed results.

This clears the prospective research-followup criterion (>=35 targets and>=5
in each fold). It supports an abstaining correspondence branch, not universal
coverage or semantic correctness. Third-fold weakness is explicit. SEM324 was
therefore separately registered and executed for class agreement/error coverage;
its label reads do not belong to this no-label experiment. No semantic prediction
export, final training or production replacement in SEM323.

Config SHA256: `cab3d077ec1c5e038643cef79420fc9df062b8225cac1b24df64a13910a19764`.
Result hashes under `outputs/sem323/`:

- `prepare-20260908-r1/result.json`: `245c037cf9242361e5a764ab0cec96b5427a25be599d4f45cdf9b8c951d721c6`.
- `runtime-20260908-r1/result.json`: `451c93da058670907444399387b0c10f76a2f1b10e0b013970d9b1080c0b2c65`.
- `replay-20260908-r1/result.json`: `940afd2a6d81a0d52e54e34cd1845ddc9e80a19d0e4cbe6adcf26753d070a65d`.
- `geometry-20260908-r1/result.json`: `c9121086e26568c92b896b30a1fb9ca7facc94eb65e9da885329dbfcdd792de3`.
- `audit-20260908-r1/result.json`: `b758d52f2440988755f8402ae35d64f5db2070de357b2242216ac483e3917162`.

## Registration — 2026-09-08

Preregistered before ranking or decoding new inputs. Base commit
`49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus sealed sources; seed20260805;
manifest `data/manifests/contest_dataset_v1.json`; split `sem_005_oof_spatial_v1`.
Parent SEM322/SUBMIT020; immutable SEM320 model and SEM321 geometry implementation.

Only change is evaluation scope: all remaining105 calibration targets (45/41/19
per fold), excluding the3 already tested in SEM322. Same pose-only top3 retrieval,
first passing source,512 model and1%/80% geometric pair gates. Short lists and
failed targets stay in the denominator. Report rank1/top3 and full distributions.
No universal-success requirement: this optional branch may abstain. At least35
accessible targets and5 in each fold justify a separately registered class/label
diagnostic. These are research-continuation criteria, not deployment gates.

Only rendered target and same-fold real source RGB. Check IDs or another fold's
target names are allowed in the training-source role, never as real target input.
No labels or target tracks,105 targets fixed before images or matching. Preserve
all failures, original models and18 production artifacts. One representative GPU
replay and existing independent geometry audit reused; no new dependency work.

No semantic exports or final fitting. Rollback: omit the optional matcher; SEM161
and RGB remain unchanged. Budget30minutes/100GiB disk floor.
