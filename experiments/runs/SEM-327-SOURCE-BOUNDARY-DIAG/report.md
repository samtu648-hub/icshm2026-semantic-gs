# SEM-327 — Source boundary harmful-deletion diagnosis

Completed2026-09-08 (`completed_boundary_diagnostic`). Parents SEM325/326; seed20260805; base commit49cdcdb1c9915a9544d568754af95ea20a85ec9c plus source snapshot. Same105 training-side calibration and manifests inherited from SEM325. Every parent and frozenSEM161 preserved. Immutable config retains preregistered status; report/registry record completion.

Measure nearest-foundation source-label pixel-center distance separately for correctly deleted, harmed and still-wrong native pixels. Six fixed distance bin boundaries are descriptive. One fixed2-source-pixel protection counterfactual is set before reading distributions, not selected by radius sweeping. No candidate export, training, new matching,111 target or74 evaluation. The2px heuristic is not a proven source-coordinate uncertainty bound.

Seal inputs/code, reproduce old confusions, independently check native coordinates and EDT distances against iterative inverse/Torch and KD-tree. Keep RGB bytes and18freeze exact. If recall improves without losing mIoU/foundationIoU relative to the old dense reviewer, register an actual-prediction experiment separately. Otherwise report the tradeoff; do not use old internal pass/fail labels to erase real improvements.

## Measured result

5.405799s, no new matching/training or PNG export. All136,987 existing4-to0 pixels across105 calibration targets were independently checked. Corrected/harmed/still-wrong totals117,614/18,502/871 exactly reproduce SEM325. Both source native geometry and EDT versus KD-tree distances agree, as do all105 original baseline/dense confusion matrices and independent counterfactual matrices.

Disjoint source-label pixel-center distance bins (correctly deleted / harmed / still wrong):

- <=1px:24,177 /10,944 /194.
- (1,2]px:24,566 /4,106 /155.
- (2,4]px:32,313 /1,479 /258.
- (4,8]px:19,908 /488 /201.
- (8,16]px:8,773 /140 /20.
- (16,32]px:4,985 /603 /9.
- >32px, including infinity for sources without foundation:2,892 /742 /34.

Thus81.342558% of harmful deletions, but also41.443196% of correct deletions, are within2px of a source foundation pixel center. This supports association with source-label boundary proximity, not proof of correspondence-error causation or a clean harm-only separator. Some far-field harms remain.

The single preregistered2px in-memory protection restores15,050 true-foundation pixels, but loses48,743 correct background deletions and returns349 other-class pixels from wrong0 to wrong4. Compared with SEM325 dense:

- Foundation recall95.150025%→96.710519%, **+1.560494pp**.
- Foundation IoU70.568617%→69.116679%, **−1.451937pp**.
- Foundation precision73.201692%→70.780671%, **−2.421021pp**.
- Five-class mIoU87.879673%→87.583512%, **−0.296162pp**.
- Fold0/1/2 foundationIoU changes−1.070972/−3.043352/−0.047545pp; recall+2.811148/+2.005914/+0.077393pp. Every fold trades IoU for recall.

It remains better than pre-dense SEM160 by foundationIoU+3.137015pp and mIoU+0.638570pp, with recall only0.357929pp below SEM160. This is a meaningful lower-deletion tradeoff, **not a useless method**, but not the sought simultaneous improvement over the dense reviewer. No alternate radius was selected or tested as a quality candidate; bins are descriptive, not seven evaluated prediction recipes.

## Decision, verification and provenance

Retain all diagnostic data and the low-deletion tradeoff. Do not export/promote the2px boundary recipe or tune its radius in this ID. The separately registered SEM328 tests a second-source foundation conflict veto without the boundary margin; it is a different single variable, not a posthoc SEM327 rescue.

- Config SHA256 `6c0f06d9c268609f70f14f48697f9cb513f4d099b99ef0184daaa7f82ed196ec`.
- Result `outputs/sem327/diagnostic-20260908-r1/result.json`, SHA256 `2c966032b0ef9b664a7a6bab14aecbf169a5e6744c758ae5ec9508b044df1c13`. Its `access_seal.json` links config/source snapshot,105 target labels, selected-source labels, original baseline and RGB hashes before decoding.
- Dataset manifest SHA256 `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e`; OOF split SHA256 `ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`; parent config/105 prediction/audit/score hashes fixed in config.
- New tests cover Euclidean3–4–5 distance, independent KD-tree equality, absent foundation/infinity and invalid coordinates, exact bin boundaries,2px inclusivity, protected classes/bypass and illegal-transition rejection.5/5 pass in existing experiment runtime.
- All105 RGB file hashes,18 frozen artifacts and source/baseline files remain unchanged. No RGB decoding/rendering,111 target or74 label access, new external-camera execution or production change. Source roles may include historical check IDs when eligible same-fold train; this is not blind validation.

Rollback is simply to continue using existing SEM325/326 or formal SEM161: no predictions or models were overwritten. No failed run, commit/push or upload occurred in SEM327. All runtime data/source snapshots stay gitignored; source paths are derived from project/config or CLI roots.
