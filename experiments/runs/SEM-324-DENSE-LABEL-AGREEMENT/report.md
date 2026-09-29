# SEM-324 — Dense source-label agreement and error coverage

## Outcome — 2026-09-08

`completed_read_only_label_diagnostic`. Class/transition counts produced by
independent nearest-sampling implementations agree exactly; all18 production artifacts unchanged.
The sealed105 targets and91 selected-source observations used114 unique label
PNGs plus105 baseline OOF PNGs. Runtime13.563s. No RGB decoding, new matching,
fitting, semantic prediction exports or production changes.

All counts below refer to the fixed512x512 evaluation grid, not native1320x989
pixel counts. Baseline is SEM160 OOF on exactly these105 calibration targets,
not SEM161's74 developer set. RGB/source support was selected before label access.
Both source and target labels are PINHOLE; distorted RoMa source coordinates were
undistorted before source-label sampling. All91 support reconstructions reproduce
the parent reliable counts and4x4 coverage; labels never alter source choice.

### Evidence usefulness

-27,525,120 total target grid points;3,152,172 have usable semantic support.
  Source-target label agreement is94.085824%, but this alone is insufficient.
- SEM160 has444,621 errors on this grid; support reaches70,409 (15.835734%).
  Source evidence would correct48,069 (10.811230% of all baseline errors).
- Unrestricted replacement would also harm164,085 previously correct points:
  net **−116,016 correct points**.22,340 supported points remain wrong under both
  sources;2,917,678 are correct under both. Reject unrestricted direct transfer.
- The direct-replacement net is negative in every fold:−33,670/−78,158/−4,188.
  Baseline-error coverage varies15.145%/28.628%/1.207%; third-fold access is weak.

Source-class precision / target-class grid coverage:

- background:96.764082% /11.477384%.
- deck:98.933090% /12.457934%.
- stay_cable:53.810163% /8.682540%.
- tower:92.950783% /13.334375%.
- foundation:88.174600% /25.962815%.

These are conditional class evidence, not full-frame IoU. High deck precision
does not imply useful additions where the baseline is wrong. Background↔stay_cable
is the dominant failure:0→2 corrects2,840 but harms80,550;2→0 corrects3,916 but
harms73,140. Geometry agreement does not guarantee view-consistent thin-structure
labels; the present data do not uniquely identify the causal source of this gap.

### Post-hoc direction decomposition and next experiment

All20 directed old-class→source-class transitions were decomposed from the sealed
125-bin per-target tensors, with no new label reads. The complete20-direction
summary is retained in `outputs/sem324/transition-summary-20260908-r1.json`.
This decomposition chooses a development hypothesis, not a blind validation result.

The strongest net-positive direction is **old foundation4→source background0**:
27,359 affected points,23,474 corrections,3,707 harms,178 still wrong; net+19,767.
Corrections/harms by fold are10,056/1,929;13,174/1,707;244/71: all net positive.
The3,707 harmed true-foundation points imply about1.92pp recall loss on this grid;
the benefit must not be reported without that cost. Tower→background and
deck→background also have net+8,143/+4,828, but are not bundled into one intervention.

Decision: retain the dense source infrastructure and prioritize one **native-resolution
foundation-only review** experiment, with abstention when no source qualifies;
keep other classes/RGB unchanged. First implement the actual inference rule and
evaluate full native predictions under a separate registration, then use the
appropriate frozen evaluation protocol. Do not call these sampled-grid counts a
completed native candidate or use target labels at inference. No further generic
matching expansion or same-target threshold sweeps are needed before that test.

Provenance:

- Config SHA256: `e2ae6b8a7c400e260d679b860065a46b4249e6eba9d17dde362fa4043daba44f`.
- `outputs/sem324/diagnostic-20260908-r1/result.json`: `e149a0ab66459cded131930161e119ac99eb09e2d1ff9b530de53c5c624f2d55`.
- Pre-decode `access_seal.json`: `3b6d77505385ccee3af59b967f2f46ae6d0a49210c16a9d2feae0f708821e068`.

Rollback remains the untouched RGB-030-FULL+SEM161 bundle. No official upload,
Git commit/push or final all-data fitting was performed.

Validation:19 combined SEM320–324 tests pass, including3 new label-audit tests.
Repository health passes131 configs and244 discovered tests (48 optional-runtime
skips); default-path full dataset scan skipped. Selected label/baseline hashes,
source snapshots, registry/result hashes and `git diff --check` pass. No geometric
support points fell outside the PINHOLE label grid. No SEM324 failure files.

## Registration — 2026-09-08

After SEM323 sealed105 targets and91 first-passing sources, preregister this
read-only diagnostic before decoding any label PNG. Base commit
`49cdcdb1c9915a9544d568754af95ea20a85ec9c` plus sealed source; seed20260805;
same `sem_005_oof_spatial_v1` and released dataset manifest. SEM323's independent
audit must pass before label access. User authorized continuation toward useful
semantic evidence, without repeatedly expanding engineering preflights.

Read all105 calibration target labels for unbiased denominators, only the sealed
same-fold selected source labels, and105 SEM160 OOF baseline PNGs. Do not use
target labels to select a source or matching support. Both label sets are already
undistorted PINHOLE: undistort the real-source RoMa positions before indexing
source labels; target grid remains pinhole. Nearest-center sampling only.

Report5-class label confusion, class coverage, and counts of baseline errors that
source evidence would correct versus correct pixels it would harm under direct
replacement. No prediction images, candidate export, fitting, threshold/source
selection or mIoU promotion. This checks evidence usefulness, not a deployed method.
Abstentions stay in target-class/error denominators. All positive/negative cases
are retained. Independent sampling/counting and frozen artifacts checked once.

Budget10minutes/100GiB floor. Rollback is unchanged RGB-030-FULL+SEM161; no upload
or Git push. Any resulting learning/fusion mechanism must be separately registered.
