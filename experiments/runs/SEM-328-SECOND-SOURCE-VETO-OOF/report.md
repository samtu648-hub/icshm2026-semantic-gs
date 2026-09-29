# SEM-328 — Second-source foundation veto OOF

Completed2026-09-08: `completed_oof_paired_improvement`. Parents SEM325/327, seed20260805, basecommit49cdcdb1c9915a9544d568754af95ea20a85ec9c plus sealed source snapshots. Both105 calibration and subsequently fixed111 check improve mIoU/foundationIoU over SEM325 while raising recall. Report all fold and SEM160 comparisons; no arbitrary minimum gain or reinterpretation of SEM326 old gates. Immutable config keeps its preregistered status.

SEM327's fixed2px source-boundary protection restored15050 true pixels but lost48743 correct background deletions; recall improved1.560494pp but foundationIoU/mIoU fell1.451937/0.296162pp versus dense. Retain that low-deletion tradeoff, do not combine it here.

Single new variable: second source in preexisting pose-ordered accessible_sources may veto an old4-to0 deletion only with native reliable source label4. No second source or unreliable/conflicting other class retains the original dense result. Source/geometry/K/matcher unchanged; no background-consensus requirement, margin or training. Source labels same-fold training, target GT only after105 PNG sealing and independent full output reconstruction. No74 or new external-camera render claim. Source access, RGB and18freeze verified. FormalSEM161 untouched.

## Actual calibration105

Compared with the original dense SEM325 reviewer:

- Five-class mIoU87.879673%→87.882717%, **+0.003044pp**.
- Foundation IoU70.568617%→70.585333%, **+0.016716pp**.
- Foundation recall95.150025%→95.669291%, **+0.519266pp**.
- Foundation precision73.201692%→72.915132%, **−0.286560pp**.

11,795 former deletions are vetoed:5,008 true-foundation pixels restored,6,727 correct background deletions lost,60 other-class pixels still wrong (15 deck/22 stay_cable/23 tower). Remaining corrections110,887, harms13,494, still-wrong811. Thus the mechanism retains94.280% of correct dense deletions and removes27.067% of dense harmful deletions; it is not a Pareto improvement across every conceivable metric (precision and total correct pixel count fall).

FoundationIoU / mIoU / recall deltas relative to SEM325 by fold:

- Fold0: **+0.185863 /+0.037183 /+0.764782pp**.
- Fold1: **−0.130115 /−0.026792 /+0.834446pp**.
- Fold2: **+0.000818 /+0.000153 /+0.012464pp**.

Relative to pre-dense SEM160, aggregate foundationIoU/mIoU are+4.605669/+0.937776pp, recall−1.399157pp; all folds still improve foundationIoU versus SEM160. Small positive aggregate gains suffice for the preregistered research progression; they are not statistical-significance claims. The fixed rule proceeded directly to111 without tuning source choice, thresholds or training.

## Actual fixed check111

Compared with SEM325 dense:

- Five-class mIoU87.643590%→87.652811%, **+0.009221pp**.
- Foundation IoU69.461184%→69.508201%, **+0.047017pp**.
- Foundation recall96.798472%→97.221776%, **+0.423304pp**.
- Foundation precision71.094488%→70.916805%, **−0.177683pp**.

11,029 former deletions are vetoed:4,977 true-foundation pixels restored,6,016 correct background deletions lost,36 other-class pixels still wrong (18 deck/13 stay_cable/5 tower). Remaining corrections125,185, harms16,262, still-wrong494. This preserves95.415% of dense correct deletions and reduces its harmful deletions23.433%. Foundation TP1,138,109→1,143,086; FN37,642→32,665; FP462,731→468,783. Net five-class correct pixels fall1,039 versus dense despite better mean-class IoU; do not replace the actual objective with raw pixel count or claim every metric improves.

FoundationIoU / mIoU / recall deltas versus SEM325:

- Fold0: **+0.171218 /+0.034317 /+0.636222pp**.
- Fold1: **−0.057060 /−0.011970 /+0.528826pp**.
- Fold2: **+0.004133 /+0.000812 /+0.032979pp**.

Compared with pre-dense SEM160, check foundationIoU/mIoU are+4.016195/+0.821267pp, precision+4.812267pp, recall−1.383116pp. All three folds retain positive foundationIoU gain against SEM160. The same slight fold1 tradeoff appears on both partitions; it must remain visible. These are historically reused incremental OOF partitions, not new blind tests or new final74 numbers.

## Engineering and provenance

- Actual105+111 full1320×989 PNGs exported in separate gitignored directories. Independent second-source native correspondence uses Torch bilinear sampling, iterative radial inversion and essential-matrix checks; primary uses the unchanged SEM325 NumPy/OpenCV path. Independent flattened source-label gathering/full-map reconstruction gives exact216 output maps and native counts. First-source decisions remain frozen parent artifacts; this is a delta reconstruction on parent deletions, not new RoMa inference.
- Second-source access exists for84/105 calibration and83/111 check targets;20 and19 unique second-source labels actually decoded. Reliable second-source evidence covers91,595 and91,886 original deletion pixels. Source eligibility and pose order derive from the already sealed top3 pair lists; no GT-selected source. No second source/evidence leaves the original dense decision intact.
- Prediction/audit/score wall times: calibration **3.500761/3.277791/4.737680s**; check **3.658645/3.534997/5.065242s**. These are cached-map incremental CPU costs, excluding the original matching and renderer costs. No model loading, new GPU work, environment change or training.
- Config/source/access hashes precede second-source label reads. Target-role GT is decoded only after all PNGs and full independent output audit are complete. Every old baseline/dense per-view and aggregate confusion matrix exactly reproduces SEM325; new candidate matrices agree under bincount versus independent indexed counting and independent restoration arithmetic.
- Disabled branch returns dense exactly. Source/direct-output class1/deck, class2/stay_cable, class3/tower masks and RGB byte hashes remain exact;18 frozen artifacts remain unchanged. Unit tests exercise distorted official-frame coordinates and independent equivalence, but this experiment has **not** run fresh external-camera renders; that belongs with final consolidation.
- Six new tests cover deterministic second-passing source order, absent/disabled source fallback, source4-only veto and known geometric support, unreliable rejection, official distorted-coordinate independent/protected output and invalid transitions. Together with5 SEM327 tests, SEM320–328 plus SEM314 targeted suite passes50/50 in the existing experiment runtime. `bash scripts/repo_health.sh`:135 JSON-compatible YAML configs,269 tests with51 optional-runtime skips in system Python. `git diff --check` passes.

Config SHA256 `965a8256464a74efd743feaedb2de840bc6e784e9d05ef25a79c797e36e865ba`; dataset SHA256 `0a188d4b4cfa5ffc2a7f151018f1cb012782e8f9860a3338aed084478d675b3e`; OOF split SHA256 `ce8a24e11d93df7634331a1ce33ae59385297cae90a0bd688b61769cf27f222c`. All parents/results/matching links are fixed in config. Output root `outputs/sem328/`:

- `calibration-predict-20260908-r1/result.json`: `b93af7528db4575333bb5dfb0df1634ded2b6850ce162d44643e2269eddebbb6`.
- `calibration-audit-20260908-r1/result.json`: `8544357a9b1c2b2baf73f113e94ca66a85d69bd58894abf0dda079abea9abe1c`.
- `calibration-score-20260908-r1/result.json`: `4e5d35464359390d2c1603e4f4c8cd425eaeebd910330edc56a6d531f88bf4c9`.
- `check-predict-20260908-r1/result.json`: `00de285f7483bd42d3e5109a892ece6bf20fb8172a461a9e5f91a0e7a5bb8584`.
- `check-audit-20260908-r1/result.json`: `9d392577763cb1c70b3bbf80c7cb4d692dc5b0840100c15f2b73b5ab880d4b00`.
- `check-score-20260908-r1/result.json`: `17ba2fcccaf060d89ae27a554db34ab53cb81193af936ae8d5043c7ad30429cf`.

Each prediction links `access_seal.json` and `source_snapshot/`; each score links `score_access_seal.json`, config, prediction and independent audit. Paths come from project/config and CLI data/artifact roots. No failed run occurred; no data/weights/logs were staged, no Git commit/push or upload was performed.

## Decision and next step

Keep SEM328 as a modest OOF improvement focused on recall recovery while retaining nearly all dense IoU benefit. It is neither a large new scoring jump nor universal superiority: precision and fold1 IoU fall slightly, different sources can make correlated errors, and improvements on105/111 do not determine what will happen on the full219 source bank and SEM161 final baseline.

Next separately register fixed final consolidation of this unchanged veto into SEM326's full219+SEM161 combination; compare SEM161, SEM326 and the new candidate on the same74, preserving all versions. Reuse sealed matching where valid, verify current rule at official external-camera native/small/new-pose coordinates and RGB equality. Report continuous mIoU/IoU/precision/recall and local tradeoffs, not just the old internal gate labels. No current-turn74 scoring, final candidate or production replacement is claimed.

Rollback: bypass the new veto to recover existing SEM325/326 dense output exactly, or use unchanged formal RGB-030-FULL+SEM161. No frozen module or baseline artifact was modified. Paper/presentation and all-data fit remain deferred.
