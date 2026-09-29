# SEM-261-POINT-REFINEMENT-OOF

Status: completed calibration gate failed; not promoted. The positive fine-feature
ablation signal is retained. No111 check or74 developer target metrics were
computed in this run, and no final model was trained.

SEM-260 found 95.46% of calibration deck FN and 71.09% of deck FP within
four pixels of a GT boundary. This motivates native point refinement, not
another threshold sweep. RGB gradients and example images are suggestive, not
proof of recoverable semantic information.

Only the new point residual heads are fitted. The candidate receives frozen
fine features/native RGB and the matched equal-size control receives zeros in
those slots. Both use identical coarse inputs, samples, optimizer, final-only
budget and bidirectional background/deck fusion. No blanket old-deck protection
or protection-mask dilation is allowed. The frozen class2/3/4 and rejected
foundation pixels stay exact. Parent networks and RGB artifacts are read-only.

Same-fold parent checkpoints generate training inputs. Other-fold OOF teachers
must not supply current-fold training features, because their training sets
can contain current-fold held-out labels. SEM-001 nevertheless inherits all219
labels, so the resulting study is incremental OOF, not full-chain OOF.

The fixed 108 calibration and 111 check partitions have been reused, with all219
also evaluated by SEM-250. This experiment still sequences its own metric reads
after prediction sealing, but does not claim a fresh independent test.

The implementation is a small independent PointRend-inspired point MLP with
native dense evaluation, not the paper's adaptive subdivision algorithm.
Source: [PointRend](https://arxiv.org/abs/1912.08193).

## Executed results

The full matched three folds and18,000 total head optimisation steps completed.
Each head has9,089 parameters. On108 reused calibration targets, fine candidate
versus SEM-160 yields deck IoU+0.186668pp, precision+2.367207pp, recall−2.211658pp,
and total mIoU+0.012806pp. It fails the unchanged+0.2pp IoU and+0.5pp recall gates.
Against the same-size coarse control, its deck IoU improves0.955738pp. Thus fine
features help the matched model, but the current recipe is not a safe upgrade.

Baseline deck IoU/precision/recall are94.530132/95.628069/98.800008%; coarse
control93.761062/97.571627/96.001286%; fine candidate94.716800/97.995275/96.588350%.
The candidate adds26,370 deck pixels and removes383,210. It corrects127,734
previously wrong pixels but harms191,099 previously correct pixels; a change
from a wrong deck label to a wrong background label counts as neither correction
nor harm. The five-class confusion matrix, all IoU/precision/recall, per-view
metrics and both-arm same-view panels/native deck crops are retained in result.

Full runtime448.838s (0.124677h); feature extraction336.195s, point-head fitting
11.860s for both arms combined, OOF inference90.326s. Peak allocated CUDA memory
2,846.886MiB. Runtime: Python3.12.3, Torch2.11.0+cu128, Transformers5.5.3; parent
training reports recorded4.57.1. Seed20260805, frozen split hashes/config and
worktree source digests are in the prediction seal. The original executed runner
and helper source snapshots have been retained before subsequent refactoring.

Every fold's checkpoint reload is exact, parent tensors are unchanged, and
enabling hidden-state extraction changes parent probabilities by exactly0.
All219 zero-residual class maps and RGB arrays replay exactly; SUBMIT-020 is18/18
before and after. Native989×1320 and240×320 official SIMPLE_RADIAL/COLMAP cameras
exercise the actual trained fold0 candidate head, with zero-residual, protected
pixels and frozen RGB replay all exact. This is an engineering probe, not a final
or hidden quality result. External probe runtime14.586s.

Full result SHA256: `b5ca773926c97920732e7d40de582809ea6660390c595d1a2e296fe302bfe654`.
Fold0 candidate checkpoint SHA256:
`b64887b63f9afd247439850dd9b9312ef30be1a811cf76b07275854af9dd01ca`.
Outputs: `outputs/sem261/full-20260907-r1/` and
`outputs/sem261/external-20260907-r1/` (Git ignored).

SEM-262 independently tests a training-class-balance objective with two identical
fine-feature heads. It does not alter this result, threshold, gate or checkpoint.

Calibration spatial-fold breakdown (46/42/20 views, derived from the already
sealed per-view confusions): candidate deck IoU gains are+1.358592,+0.731624,
−2.939024pp versus SEM-160. The aggregate gain hides substantial fold2 regression;
these are descriptive reused-calibration statistics, not new independent tests.
