# SEM-325 — Native dense foundation-only review

- Status: **completed_oof_quality_passed**,2026-09-08. Both105 calibration and fixed111 check pass. Qualified research candidate only; SEM161 production and all18 frozen files unchanged.
- Execution-before-results registration: implementation and native quality were pending when config/ADR074 were recorded; the following results do not alter those gates.
- Parents: SEM324 direction diagnostic, SEM323 fixed matching/access, SEM160 OOF baseline, SUBMIT020 freeze.
- Seed20260805; commit49cdcdb1c9915a9544d568754af95ea20a85ec9c plus sealed worktree sources; dataset and split hashes inherited and checked from SEM320→324.
- Only major variable: execute foundation4→background0 correction at actual1320×989 pixel centers. Matching remains512 and all source/geometry thresholds remain frozen. This is not native-resolution matching.
- Source GT is same-fold train only; target GT is scoring-only after predictions are sealed. SEM324 used calibration to choose this direction, so this is reused incremental developer OOF, not blind evidence.
- Gates and staged105→111 evaluation are fixed in config before native predictions. Formal SEM161, RGB, class1/2/3 and18 artifacts remain untouched.

- r1:105 native predictions completed in41.224s. Audit startup failed before reconstruction/target GT scoring: Torch exposes a virtual `_ops.py` module; source snapshot attempted to hash it as a local file. r1 outputs and failure retained. r2 fixes only source-snapshot discovery with `is_file()` and regenerates predictions under the corrected source seal; no inference/quality/config change.

## Calibration completed

- r2 prediction41.041s, independent full-native reconstruction25.430s, scoring3.808s. All105 candidate PNG hashes exactly match r1. Main config SHA256 `836b45510e33a0c63040592074f4e6f3af293272f9c11602571403683bac6770`.
- Native support15,703,663/137,077,380 pixels;136,987 changed,117,614 corrected,18,502 harmed and871 still wrong. Outside source-label coordinates:0. Foundation FP reduction118,485 is NOT identical to117,614 five-class corrections.
- Relative to same105 SEM160 OOF: mIoU+0.934732pp, foundation IoU+4.588953pp (65.979664→70.568617%), precision+5.880558pp, recall−1.918423pp (97.068448→95.150025%). All seven predeclared quality gates pass, plus exact class1/2/3 masks/RGB bytes/18 artifacts.
- Fold0/1/2 foundation IoU deltas+4.751428/+8.039789/+0.138665pp; mIoU+0.966480/+1.633000/+0.028619pp. Recall deltas−3.334438/−2.576802/−0.093336pp: aggregate gate pass does not mean uniform recall safety. Third fold benefit remains small.
- Calibration result `outputs/sem325/calibration-score-20260908-r2/result.json`, SHA256 `6c9c6bdc70db1f9809c5d992a39f21c4c760acfec150da9625d79894c27ee1be`.
- Check111 launched only after this result qualified. `matching_config.json` binds the same main config and passed calibration;333 pairs, no label access in matching. Native prediction/scoring functions are AST-exact copies of calibration, sharing the same core and quality gates. Check geometry completion is descriptive (abstentions allowed), not an additional35/min5 access threshold or semantic-quality claim.

## Fixed111 check completed

- Actual full-frame mIoU with background:86.831544→87.643590%, **+0.812045pp**. Foundation IoU65.492006→69.461184%, **+3.969178pp**; precision66.104538→71.094488%,+4.989950pp; recall98.604892→96.798472%,**−1.806420pp**. All seven fixed quality gates pass.
- Of152,970 modified pixels:131,201 five-class corrections,21,239 true-foundation deletions,530 changes that remain wrong; net correct pixels+109,962. Foundation FP594,462→462,731 (−131,731); TP1,159,348→1,138,109. Lower FP alone must not be confused with correct background recovery.
- Fold0/1/2 foundation IoU deltas+4.410876/+5.837501/+0.259974pp; mIoU+0.903512/+1.190152/+0.053606pp; recall−2.892720/−1.982006/−0.223623pp. All folds improve IoU, but fold0 recall cost is larger than aggregate and fold2 gains are small.
- Pose-top3 access95/111 versus rank1-only90/111;249/333 pairs pass fixed pair gates. Fold access45/49,37/42,13/20. Native reliable support12,457,189 pixels, out-of-label-frame support0. All16 inaccessible targets remain in the quality denominator and retain baseline predictions; some accessible views also have zero changed pixels.
- GPU matching666 directions took130.595s wall time (67.464s summed synchronized matching, excluding loading/serialization); new-process first-forward replay5.194s exact. RTX5070 peak allocated2508.446MiB/reserved2718MiB, not whole-board use. Primary/independent333-pair geometry40.793/51.986s. Native111 prediction43.790s, independent reconstruction26.552s, scoring4.027s. Outputs occupy about1.8GiB, all gitignored. Existing environment and upstream weights/code untouched; inherited readonly-array warning remains, with exact input hashes/arrays.

## Verification and reproducibility

- All216 completed candidate PNGs have exact independent full-native reconstructions using CPU Torch interpolation, iterative radial inversion and essential-matrix distances. Full-image class1/deck, class2/stay_cable, class3/tower masks remain exact. Baseline/source-label and rendered-RGB hashes remain unchanged;18 frozen production artifacts verified. This is byte-invariance evidence, **not** fresh RGB rendering or a new external-camera rendering run.
- Target GT is not an argument of `native_prediction`; only same-fold training-source labels are read during inference. Per-stage access lists and source/config snapshots precede decoding; candidate PNGs precede target-role GT scoring. Calibration prediction uses17 unique eligible source labels. Independently computed confusion matrices match on all216 images; separate arithmetic/seal audits agree on both partitions without additional label decoding.
- Both calibration/check currently sealed13 project/config source files per prediction stage. Direct score linkage records dataset SHA256 `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e` and OOF split SHA256 `ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`; full parent chain checks semantic split, original label/rendered-RGB/baseline manifests and camera hashes.
-28 combined SEM320–325 tests pass in the pinned experiment Python, including9 new native tests: strict field bounds, cycle/epipolar rejection, source distortion, protection/abstention, official-world-translation and chunking invariance, fixed recall gate, and unchanged calibration/check inference/scoring. One added test initially had a misplaced local variable; test-only placement corrected, no inference/config edit. Repository health final result recorded below.
- Calibration prediction/audit result hashes: `f6c50cf9ac1db51e8ac6f9fe54f4f3af22535d428e1abbbd80f247c3bcb720c5` / `b040fda7499f7400f1b145f0ebd6cc274187186bbba8f96e42e59b52a3e5c0eb`.
- Check score `outputs/sem325/check-score-20260908-r1/result.json`: `a811c2edefb2ea960088aac9b08a2aa5c92bc362a651e982f7a17e63bcfeb3c6`.
- Check prediction/audit results: `5179c6107b0359de41188e807a6ca5a7cbf8cccf97d788b86b9b52e3d714e45d` / `b2b1aea669d14f26bf1c7e9204cb6c743b6515539bed4c85438e48b287d4317d`.
- Check independent quality audit `outputs/sem325/check-quality-audit-20260908-r1.json`: `d7f432c09c9d6618eeb212d4e167e22794a9a220fb4cb2912039c24b391e6e6c`.
- Check matching prepare/runtime/geometry/audit results are under `outputs/sem325/check-matching-20260908-r1/{prepare,runtime,geometry,audit}/result.json`. Linked hashes respectively `30e0871de1f971277ca0a0f3cf8da3e620a8aa9c93250eada57c664c14321aeb`, `2bc7880d88b6d2fe3a101139a8af65c273f5381cbfe002262476b27698fe2509`, `0622a0e956b846af0f6c4dfcb29199063c10d39c797494bb84083143f2a243b1`, `afe544ada6b7323da71de89f3566aadf87788eee1ffe0e5aecacc29c4f1ad4f8`.

Entry points: `scripts/run_sem325_native_review.py` (calibration predict/audit/score), `scripts/run_sem325_check_matching.py` (prepare/run/replay/geometry/audit), `scripts/run_sem325_check_review.py` (same native core for check), and `scripts/audit_sem325_results.py` (independent arithmetic/provenance). All data/artifact roots are explicit CLI arguments; select new empty output directories for replays. Check inference additionally requires `--matching-root`; GPU matching uses the existing scoped `outputs/sem320/runtime` dependency overlay, never a global environment update.

## Interpretation, next step and rollback

This is the first actual dense-source foundation-only candidate in this line to retain the diagnostic signal on both partitions. It is not a complete semantic replacement, not proof of high-resolution matching, and not hidden-test evidence. Direction was chosen using SEM324 calibration;111 is historically reused incremental OOF, and some names are legally used in source roles across folds. Other-class corrections and foundation additions were not tested here.

The gain is close to SEM313's DINO review on the same check, with a larger recall cost; do not claim general superiority to DINO. It establishes usefulness of a different correspondence-based evidence source. SEM314 already shows why good OOF results cannot authorize final deployment by themselves.

Next: separately register SEM326, use the full219 training reference bank with the frozen SEM161 baseline and unchanged source/rule thresholds, then one sealed74 developer evaluation plus actual official-COLMAP new-camera interface checks. Focus on source-bank migration, recall, spatial regressions and runtime dependencies; no new DINO fusion or OOF threshold sweep. Not started in this experiment.

Rollback: bypass this optional review and return the original SEM160 OOF PNGs; formal deployment continues to use the untouched RGB-030-FULL+SEM161/SUBMIT020/030 bundle. No production/model/data overwrites, Git commit, push, upload or all-data fit occurred. All failed artifacts remain available.

Final repository verification: `bash scripts/repo_health.sh` passed132 JSON-compatible YAML configurations and253 discovered tests (49 optional-runtime skips); the separately pinned-runtime SEM320–325 suite passed28/28. `matching_config.json` was parsed and bound during every matching/native-check stage. `git diff --check`, registry/result hashes, and a final independent18-artifact freeze verification passed. Frozen manifest SHA256 remains `53ca7d469eed2666cd1a5d3bfd8c4374275daa6e173cddd62d88a4d410794815`. One retained experiment failure is `outputs/sem325/calibration-audit-20260908-r1/failure.json`; no failed output was deleted.
