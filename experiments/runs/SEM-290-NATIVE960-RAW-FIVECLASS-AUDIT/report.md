# SEM-290-NATIVE960-RAW-FIVECLASS-AUDIT

## Status

Completed on 2026-09-07 as a read-only diagnostic. It cannot replace
`RGB-030-FULL + SEM-161`, train a final model, or open the 111-view check and
74-view developer labels.

Two initial monolithic replay attempts (`r1`, `r2`) were interrupted by the
interactive host execution ceiling after respectively 145 and 39 raw PNGs.
The first chunked attempt (`r4/chunk0`) then exposed a result-serialization
spelling error after producing raw PNGs but before any result file. The corrected
12-piece replay completed all 219 raw PNGs with the intended 108-label access,
but its first merger (`r5`) stopped on a path-type error before emitting a
combined result. None of the interrupted attempts wrote a complete metric
result. Their partial ignored outputs are retained as engineering interruptions.
The corrected merger (`r6`) accepted all 12 completed chunks after verifying
every raw PNG hash.

## Question

`SEM-270` seals one scalar foundation probability per view and then protects
deck, `stay_cable`, and tower during fusion. Its reported fused result therefore
cannot answer how its five-class control behaves before that protection. This
audit will measure the fixed raw five-class argmax of the already trained,
three-fold OOF control without changing any model or decision threshold.

## Protocol

The audit reloads the three sealed `five_control` checkpoints, re-renders their
held-out native 1320×989 RGBs with the registered 960/480 sliding window, and
writes raw class PNGs for the 219 inherited OOF views. It verifies checkpoint,
RGB, source-config, split and freeze hashes. Target labels are decoded only for
the already reused 108-view calibration half. The 111-view check, 74-view
developer set, unlabeled images, hidden inputs and external cameras remain
outside the audit.

The report will compare raw argmax with the sealed SEM-270 fused five-class
control and the SEM-160 baseline. It is a mechanism diagnostic, not evidence
that a raw model should replace the existing cable, foundation or deck experts.

## Completed evidence

All three sealed `five_control` checkpoints were reloaded and generated one
native 1320×989 raw-argmax PNG for each of the 219 inherited OOF views. The
combined audit decoded exactly the already reused 108-view calibration labels;
111 check labels and 74-view developer labels were accessed zero times. It did
not train, select a threshold, write float probability maps, modify frozen
artifacts, or produce a candidate checkpoint.

The raw five-class argmax is materially worse than the sealed protected fusion
on this same reused calibration evidence. mIoU including background is
0.860263 versus 0.878520 (-1.826 pp). Its deck, `stay_cable`, tower and
foundation IoU are respectively 0.922544, 0.861195, 0.852872 and 0.684479,
which are -2.276, -3.517, -1.041 and -1.928 pp against the protected output.

Raw `stay_cable` recall is only +0.048 pp higher, but background→`stay_cable`
false positives increase from 609,942 to 1,006,971. Raw foundation recall is
+0.587 pp higher, but its IoU and precision both decrease. Thus the apparent
extra raw recall is not safe class evidence that can replace the existing
specialists or their protected fusion.

The final result is
`outputs/sem290/full-20260907-r6/result.json` with SHA-256
`11ae172b84e0f0ed9ae76b0ee76a4597658a96b6e49a1f68ddda1e2ab2d723f0`.

## Disposition

Do not start a learned expert-fusion head from these raw SEM-270 probabilities:
this audit provides negative evidence for direct raw five-class replacement,
and no independent holdout remains in this result. Retain multi-expert
cross-fitted fusion only as a later conditional research option if a different
source demonstrates complementary OOF evidence. The next technical hypothesis,
if the team authorizes one, remains a separately registered five-class
foundation signed-distance or contour mechanism that constrains both sides of
the boundary. `RGB-030-FULL + SEM-161` and its rollback ladder are unchanged.

## Registered stop condition

Any artifact, checkpoint, RGB, split, probability-mass or protected-fusion
reference mismatch stops the audit. There is no threshold, checkpoint, crop,
seed or training retry. A successful diagnostic records evidence only; a
subsequent learned fusion would require a separately registered cross-fitted
protocol and an unused independent check.
