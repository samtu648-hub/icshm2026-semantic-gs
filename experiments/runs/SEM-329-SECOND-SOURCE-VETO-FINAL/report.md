# SEM-329 — Fixed second-source final comparison

Documentation update2026-09-08: the [stage research summary](../../../docs/experiments/research_summary_20260908.md) consolidates the completed evidence and links this run with earlier work. README/METRICS/status/roadmap now distinguish the formal SEM161 bundle from the SEM329 research candidate. This is documentation only; experiment configuration, source snapshots, scores and outputs are unchanged.

Status: `completed_three_arm_final_comparison`, 2026-09-08. Positive developer candidate retained; preferred to SEM326 for the measured aggregate IoU/recall balance, not an overwrite of formal SEM161. Parents SEM328/326/161; seed20260805; basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c plus sealed worktree sources. Full219 reference bank and same74 developer targets, dataset/split and parent hashes fixed in config. The immutable config retains its preregistration status; report and registry record completion.

Only the unchanged SEM328 second-source foundation veto is added to SEM326. No boundary margin, new threshold or training. Source selection and all512 RoMa geometry are unchanged; native cached matching reused. Compare SEM161, SEM326 and SEM329 on the same labels; overall mIoU benefit and recall/precision/local tradeoffs reported separately from historical internal gates. The74 are reused, not a blind test.

Seal74 outputs and source access, independently reconstruct every output, then run four actual fresh official external-camera probes (native140 twice,small140,novel140). Only after engineering completion score74 once and independently verify three-arm metrics/fixed six panels. Preserve RGB/class1/2/3/18freeze, all parent artifacts and failed runs. No final production/bundle overwrite or upload.

External r1: both native probes completed; small render/matching also completed, then wrapper incorrectly used the training-label reader (fixed989×1320) on a240×320 output and failed before applying the new veto. Parent/child failure records retained. r2 only replaces output reading with a camera-size-independent0–4 single-channel ID validator; source training-label reader and every inference/selection/coordinate rule are unchanged.74 native prediction/audit already sealed and unaffected, no targetGT scoring occurred before this fix.

## Result and decision

There is measured positive score benefit against both parents. SEM329 gives the highest aggregate mIoU and foundation IoU of these three fixed arms, while recovering part of SEM326's lost recall. This is a small incremental improvement over SEM326, not a Pareto improvement or evidence of hidden-set significance.

- SEM161: five-class mIoU **83.387017%**, foundation IoU **72.770345%**, precision **78.275843%**, recall **91.186564%**.
- SEM326: mIoU **83.682675%**, foundation IoU **74.242876%**, precision **81.475684%**, recall **89.319965%**.
- SEM329: mIoU **83.705003%**, foundation IoU **74.354428%**, precision **81.244656%**, recall **89.761802%**.
- SEM329 minus SEM161, in percentage points: mIoU **+0.317985**, foundation IoU **+1.584083**, precision **+2.968813**, recall **−1.424762**.
- SEM329 minus SEM326: mIoU **+0.022328**, foundation IoU **+0.111552**, precision **−0.231028**, recall **+0.441838**.

Of the1,431 vetoed deletions,739 restore real foundation,672 undo correct background recovery, and20 remain incorrect other-class pixels (14 deck/1 stay_cable/5 tower). Thus23.670724% of SEM326's3,122 harmful deletions are prevented, while91.866376% of its8,262 correct deletions are retained. Net five-class correct pixels increase by67 relative to SEM326; the20 incorrect-to-incorrect changes are not counted as corrections.

Relative to SEM161, SEM329 changes10,053 pixels:7,590 corrected background,2,383 harmed real foundation and80 still wrong (20 deck/9 stay_cable/51 tower), net correct pixels **+5,207**. Foundation TP/FP/FN are150,132/34,658/17,124, versus SEM161152,515/42,328/14,741. The7,670 fewer foundation false positives include80 cross-class mistakes and must not all be called fully corrected pixels. Class1/2/3 predictions and their metrics are identical across all three arms.

Keep SEM329 as the preferred measured IoU/recall research candidate among SEM326/329; preserve SEM326 as the slightly higher-foundation-precision alternative and SEM161 as the unchanged formal fallback. SEM329 still loses recall versus SEM161 and does not fix existing baseline0→4 omissions: it only restores selected SEM326 deletions. No production promotion, new experiment, all-data fit or parameter sweep was performed in this ID.

## Spatial and fixed-view comparison

All four inherited trajectory groups retain positive mIoU/foundation IoU versus SEM161; these are segment intersections, not new OOF folds. Delta pairs below are mIoU/foundation IoU in pp, with SEM329 foundation recall separately stated:

- seg-007,29 views: versus SEM161 **+0.038004/+0.188954**; versus SEM326 **−0.000331/−0.001639**. Recall91.179060%, +0.016969pp versus SEM326. The small aggregate regression is retained.
- seg-008,30 views: versus SEM161 **+0.414943/+2.068611**; versus SEM326 **+0.014120/+0.070713**. Recall88.698493%, +0.541494pp versus SEM326.
- seg-009,7 views: versus SEM161 **+0.910227/+4.545326**; versus SEM326 **+0.038216/+0.190962**. Recall88.758426%, +0.326158pp versus SEM326.
- seg-010,8 views: versus SEM161 **+1.406358/+7.009904**; versus SEM326 **+0.273467/+1.366160**. Recall83.634565%, +3.578815pp versus SEM326 but still **−10.403741pp** versus SEM161. It supplies much of the recovery and remains a meaningful local recall weakness.

Fixed140–146 mean per-view mIoU is83.863630%/84.814621%/84.816663% for SEM161/326/329: SEM329 **+0.953033pp** versus SEM161, only **+0.002042pp** versus SEM326. SEM161-ranked worst8 (061,060,059,062,065,063,057,058) remain58.733667%/58.751901%/58.751901%; the new veto adds nothing to that subset. Neither subset is reranked after seeing SEM329.

Per-view mIoU versus SEM161:57 improve,8 regress,9 tie; largest regression088 −0.976516pp. Versus SEM326:22 improve,17 regress,35 tie; largest regression145 −0.487149pp and largest gain180 +0.718368pp. Group aggregate gains do not imply every image improves.

The unchanged historical SEM326 internal cutoffs are recorded only as context:13/14 now pass; mIoU+0.30pp and foundation IoU+1.50pp are exceeded, while the90% absolute foundation recall floor remains short by0.238198pp. These are neither official competition rules nor statistical-significance thresholds and do not determine whether the above real gains exist. Old SEM326 records are not reclassified.

## Execution and independent evidence

- The full219 training source bank is hash-validated and excludes74 developer/7 buffer/100 unlabeled source roles.57/74 targets have a second passing source;11 unique second-source labels actually decode.6,233 old deletion pixels have reliable second-source evidence and1,431 have label4 vetoes. No labels affect pose-only source ranking. No second source or no reliable evidence keeps SEM326, not SEM161.
- The74 actual1320×989 PINHOLE PNGs are sealed before GT access. Prediction takes2.915916s and independent full-output reconstruction2.834991s. The unchanged SEM328 core and four copied native prediction/audit functions are hash/AST-verified. Cached SEM326 matching is reused: these times exclude its original matching cost and are not end-to-end inference latency.
- RoMa remains512 with unchanged geometry, source-coordinate undistortion and native-pixel field sampling. Official external outputs are distorted; raw target pixel centers are undistorted before the same veto. Source label coordinates remain PINHOLE. No frozen renderer/matcher/model changes, target real RGB input, extra training, installation or downloads.
- External r2 runs four actual fresh renderer+matcher+current-veto subprocesses, not saved-output stand-ins. Native140 twice: first source137/second092,238 old deletions,180 reliable,40 vetoes,14.273118s/14.417977s; repeated current output is exact. Small140320×240: source137/second136,0 old deletions/vetoes,12.471969s. Novel official140 world-to-camera t+[0.05,0,0]: source137/second092,203 old deletions,163 reliable,34 vetoes,13.805329s. Total59.753257s, peak CUDA allocated2508.446MiB. Small is an interface/no-op check, not quality evidence; novel is one local pose perturbation, not arbitrary-camera generalization.
- Each external probe independently reconstructs the veto and verifies disabled bypass, class1/2/3 and RGB. All four raw RGB/SEM161/SEM326/PINHOLE RGB exactly match their sealed parent probe references, including novel. No target GT is read. The inherited read-only NumPy warning remains nonfatal; exact checks pass.
- Only after these engineering checks, the same74 GT are scored once in3.842300s. Both old arms' per-view and aggregate confusion matrices exactly reproduce SEM326's sealed score. An independent indexed-accumulation auditor in4.142536s recomputes all three arms, all74 views, groups, fixed subsets, restoration arithmetic, benefit signs and descriptive gates. It reopens the same74, no new partitions. Independent audit status is `passed_independent_three_arm_quality_audit`.
- RGB bytes, class1/2/3 masks and all18 SUBMIT020 frozen artifacts remain exact. Config/source/output seals distinguish dirty-worktree code from the base Git commit.

Six preregistered native-pixel panels054/081/140/159/160/183 were created and visually inspected. Labels and headings have separate padding; image tiles are checked byte-exact, with no resampling in the saved panels.054/081/159/160 have no new veto pixels, so SEM329 equals SEM326 there; their original structural errors persist.140 restores20 true-foundation pixels and20 background pixels.183 restores101 true-foundation pixels,52 background and4 other-class pixels, visibly combining recovered thin parts and some returned fringe. These examples do not prove a universal occlusion/boundary cause. In particular, the fixed six are not substituted for the full74 comparison or used to tune a rule.

## Provenance and retained outputs

All paths below are project-relative. Runtime data, matching weights, predictions, logs and snapshots remain gitignored. Artifact/data roots and Python executable are supplied by CLI/configuration, not embedded personal paths. Config fixes inherited source/label/trajectory manifests and all parent hashes.

- Config `experiments/runs/SEM-329-SECOND-SOURCE-VETO-FINAL/config.yaml`: `f1b41fbbfd814ef611ac7cc362acb7e157c7d85396ed7988f12f27db98c43584`.
- Dataset manifest `data/manifests/contest_dataset_v1.json`: `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e`.
- Split `experiments/splits/sem_001_scaffold_static_v1.json`: `1764d3281fedc94e03c85f2ffc272e324bb12c0d8dba1dc1955078f71d147854`,219 train/74 developer/7 buffer/100 unlabeled.
- Unchanged veto core `src/semantic_second_source_veto.py`: `f92b7bb6352db3dc8df630fc266a789bcd297523848b5e63eccef284d6e42731`.
- `outputs/sem329/developer-predict-20260908-r1/result.json`: `73c281ef5bac85faf311f81c55dbf9b0b4fccdfc7d4d72618de2154a1a35229c`; access seal and source snapshot in the same directory.
- `outputs/sem329/developer-audit-20260908-r1/result.json`: `9f52ee9d5fef39dd83e8f49a4736a33fa4bb8c2e9e65cfcce4a64bc67e2ae4c7`.
- `outputs/sem329/external-20260908-r2/result.json`: `16a48792bb9e4e0cb22074c7f803ccc503ed316e059f35629308937b7f74cc05`; links four fresh probe results, access/code/output seals and preserved dense stages.
- `outputs/sem329/developer-score-20260908-r1/result.json`: `1fa850cce879345bd9c323be96ae840431d775cb8070418772608053c88792c1`; linked score access seal/source snapshot.
- `outputs/sem329/quality-audit-20260908-r1/result.json`: `8a320276936abd0b61ebdc68df233f426b4e7a2f0da27724da6d123e3b69de84`; six images in `panels/` with image hashes and crop coordinates.
- Retained failure `outputs/sem329/external-20260908-r1/failure.json` and `small/failure.json`; completed native probes and small dense stage are also retained. The camera-size reader fix above has a regression test and does not change the74 sealed inference sources.

## Validation, risk and rollback

Four new SEM329 tests cover unchanged native-function ASTs, core/external contract, no GT access on engineering failure, and small-camera ID reading with invalid-ID rejection. Combined SEM314 and SEM320–329 targeted tests pass54/54 in the existing runtime. After documentation/registry updates, `bash scripts/repo_health.sh` passes136 JSON-compatible YAML configs and273 tests (53 optional-runtime skips in system Python); `git diff --check` passes. Final read-only closeout rechecks all74 prediction files, all18 frozen artifacts, four external semantic/RGB/baseline/dense outputs and correspondence files, all six panels, parent/current source snapshots and every registered result hash: all exact, with zero additional target-GT decodes. The four new runtime tests also pass on rerun.

This is a single-seed, historically reused developer comparison, not a fresh blind test or a significance estimate. Reference sources can share errors, and passing geometry does not establish label correctness. Compared with SEM326 the new increment is small and precision decreases; compared with SEM161 recall remains lower, especially seg-010. The source RGB/PINHOLE GT and full matching weights remain additional deployment dependencies, and this reviewer has not been integrated into the clean submission bundle.

Rollback requires no restoration: formal RGB-030-FULL+SEM161, the18 frozen artifacts, old SEM326 and all parent predictions remain unchanged. Omit the separate reviewer to use SEM161; disable the second-source veto to recover SEM326 exactly. No commit/push, official upload, production overwrite, final all-data fit or paper/presentation work.

Next, if further optimization is requested, use SEM329 as the frozen research starting point and separately diagnose remaining harmful deletions on the training-side OOF outputs: distinguish missing second-source coverage from conflicting/incorrect matched labels before choosing one inference-available variable. This is a proposed next experiment, not executed or shown beneficial here. Do not tune source/threshold/region policies on these74 final comparisons. All-data fit and materials remain deferred until method selection.
