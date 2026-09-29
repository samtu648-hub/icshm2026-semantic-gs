# SEM-395 RGB150 cross-view reliability diagnostic

Status: `completed_geometry_insufficient_not_promoted`, 2026-09-20.
Parent experiments: RGB-150-STANDARD-ABSGRAD-DEV30 and
SEM-392-RGB150-DEV270-SEED20260919-MEMORY. Base commit:
`49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus sealed worktree source
hashes. Seed: 395. Training: none.

## Scope and access

This was a read-only frozen-model diagnostic over 24 preselected SEM392
training-side targets and 48 preselected training-side source pairs. It is not
a holdout result, blind test, training result, geometric ground-truth
evaluation, or deployable prediction change.

- Target/source universe: only `primary_fold.train_rgb_300` (270 views).
- Labels before evidence seal: 0 in prepare, render, and evidence.
- Holdout RGB/label reads: 0 in every stage and the independent audit.
- Score opened exactly the 69 sealed target/source labels after the evidence
  seal; each opened file matched the sealed SEM370 label manifest hash.
- The 100 unlabeled images were never used for supervision or evidence.

The frozen RGB150 renderer supplied `D_inv = sum(T * alpha / z)` and
`A = sum(T * alpha)`; the diagnostic uses only
`Z_proxy = A / D_inv`, an opacity-normalized reciprocal inverse-depth surface
proxy. It is not geometric truth, expected depth, or an independent surface.

## Executed attempts

`r1` prepare, render, and evidence completed. Its first score attempt was
preserved as `outputs/quality/SEM-395/r1/score/result.json`: after the
evidence seal and while all 69 allowed labels had been read, an engineering
lookup used the prepare plan instead of the sealed evidence target summary and
raised `KeyError: 'evidence_counts'`. This did not alter the filter,
selection, target/source identities, evidence artifacts, labels, or any
quality conclusion. The failure was not overwritten.

`r2` was a complete fresh attempt with only that data-reference fix and a
regression test. Its prepare plan, access contract, confirmation candidate,
pair summary, and target summary were byte-identical to `r1`; render replayed
all 69 RGB150 views pixel-exactly; evidence re-ran the same sealed filter;
score completed; and an independent arithmetic audit passed.

## Main result

Fixed outcome:

`current_rgb150_depth_or_geometry_is_insufficient`

The sealed single-source reliability stages accepted 3,528,236 pair-pixels,
but requiring both sources to agree left only 28,475 consensus pixels:

- reliable consensus fraction of 31,331,520 target pixels: 0.0009088291918170583;
- reliable foreground coverage: 12,726 / 4,807,117 = 0.0026473247894736075;
- diagnostic replacement would change 46 pixels;
- corrected pixels: 2;
- damaged pixels: 44;
- net correct pixels: -42;
- sources agree while target GT disagrees: 50;
- full-target five-class mIoU counterfactual delta: -0.0017707581300374642 pp;
- full-target foreground mIoU counterfactual delta: -0.00219104524059599 pp.

Dominant first rejections were geometric: 59,120,094 geometry rejects versus
15,559 semantic-confidence/two-source-disagreement rejects. The largest
specific stages were forward depth conflict (32,709,148), source outside or
behind (15,490,734), target depth discontinuity (5,094,860), cycle
reprojection (2,776,292), and source depth discontinuity (2,583,113).
Occlusion-direction counts among forward depth conflicts were 18,781,723
behind the source surface and 13,927,425 in front of it.

Of the 50 target disagreements, 48 had both source predictions matching their
own nearest source-view GT, while 34 showed nearest-source GT disagreement
between the two source samples. This is descriptive: nearest-pixel source GT
comparisons can mix geometry, visibility, and annotation-boundary effects.

## Preserved evidence and verification

- Prepare: 24 targets, 48 pairs, 69 unique touched training views, 17 covered
  trajectory segments, 6 fixed visualization targets; 0.796 seconds.
- Render: 69/69 selected RGB150 views replayed pixel-exactly; peak GPU
  reserved 5580 MiB; 37.183 seconds.
- Evidence: 48/48 pairs and 24/24 targets completed; every target pixel has
  exactly one first-rejection or reliable stage; 31.282 seconds.
- Score: 69 labels read only after the evidence seal; 6 fixed panels written;
  3.010 seconds.
- Independent audit: passed all 11 checks, including exact confusion,
  counts, coverage denominators, rejection counts, label hashes, and outcome;
  1.147 seconds.
- Unit/static checks before sealing: 14 SEM395 tests, `py_compile`, ruff,
  258 JSON-compatible configurations, and `git diff --check` passed.
- The six fixed panels (`006`, `051`, `132`, `216`, `250`, `294`) were
  manually inspected; sparse evidence is visually consistent with the small
  quantitative coverage and does not support deployment.

Final attempt `r2` output is under `outputs/quality/SEM-395/r2`; the
preserved failed attempt is under `outputs/quality/SEM-395/r1`. Each attempt
uses about 535 MiB, and the `r2` budget remained below 4 GiB output,
30 GPU minutes, 120 CPU minutes, 48 pairs, and 11,200 MiB reserved GPU
memory.

Key `r2` result hashes:

- prepare: `6ed923510fc0fce4a99d2e132b4b338133e678db82399bda7eef584e160ee795`
- render: `43b1e5f2c850a049abacb3c8f24fbb649ebca7e8911f580e5466a5b355e42263`
- evidence: `91eb557e3985da5f19d6007af9472d716b92773ddab35532f41ea8a89c00ba0a`
- score: `20377b16251cd1f6f46dd09bc96f5b9067d6193f4439185059aaf93738c2822d`
- independent audit: `3bfb780903c0ef49eea48526745f7349f781e2b75744a11e0fcdc5414ee85345`

## Future spatial confirmation candidate

`prepare` also sealed a candidate-only spatial confirmation partition from
camera metadata only. It selects three ten-view labeled regions
(`seg-008` 152--161, `seg-015` 255--264, `seg-010` 181--190), excludes 17
buffer cameras, and retains 353 RGB views and 256 labeled semantic views. The
released camera graph already has two components; the retained graph also has
two components, 28,151 retained edges (0.762321273830156), 581,177 retained
point-observation proxies (0.8748686591339481), and 24 retained trajectory
segments. The candidate has not been executed and is not historically unseen;
its historical-use audit estimates one matched RGB+semantic chain at about
77.043 GPU minutes and three seeds at about 231.130 GPU minutes, excluding
fresh scene preparation and once-only confirmation work.

## Interpretation and decision

The conservative two-source geometric filter is too sparse and slightly
damaging in the only allowed training-side score. The dominant bottleneck is
the current RGB150 native inverse-depth surface proxy and its cross-view
projection consistency, not SEM392 source confidence. No training or fusion
candidate is authorized from this diagnostic. RGB150, SEM392, RGB140+SEM378,
and RGB130+SEM371 remain unchanged.

Any continuation must use a new experiment ID and a new hypothesis. A useful
next diagnostic would test a different, explicitly registered depth or
geometry representation before considering semantic fusion; it must not reuse
SEM395 to search thresholds, replace sources after scoring, or rescue the
failed counterfactual.
