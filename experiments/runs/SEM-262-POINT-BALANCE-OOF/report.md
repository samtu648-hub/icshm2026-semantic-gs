# SEM-262-POINT-BALANCE-OOF

Status: completed calibration gate failed; no promotion. Parent SEM-261 remains
failed against its original gate; no threshold or historical result is changed.

SEM-261 fine features beat the matched coarse control by0.9557pp deck IoU,
but relative to SEM-160 the gain is only0.1867pp and recall falls2.2117pp.
Precision rises2.3672pp. This supports testing the head's asymmetric training
objective, not declaring that fine features are ineffective or relaxing recall.

Both new arms use the SAME fine features and identical native points, parent
models, initialization and budget. The sole variable is unweighted BCE versus
positive-weighted BCE, with weight computed once as negative/positive sampled
point counts within each training fold. There is no validation-derived weight
or threshold search. A fresh matched unweighted control is retained.

Fixed gates remain deck IoU+0.2pp, recall+0.5pp, precision>=94%, mIoU nondecrease,
exact protected pixels/RGB, and superiority over matched control. Both directions
of bg/deck correction are allowed. No check/developer metrics before the current
calibration gate. Reused partitions and inherited SEM-001 mean incremental OOF,
not a new independent test. Failed results and artifacts are retained.

## Results

All three matched folds/18,000 head optimisation steps completed. Training-derived
positive weights were2.125033,2.226308,2.092735, not chosen from calibration.
The unweighted control exactly replays all219 SEM-261 fine-candidate prediction
hashes, providing a particularly strong matched repeatability check.

Relative to SEM-160 on108 calibration views, balanced candidate deck IoU improves
0.275321pp, precision1.845733pp, and total mIoU0.035418pp. Recall falls1.606467pp.
It passes the IoU, precision, mIoU and matched-control superiority gates, but fails
the unchanged recall gate. Compared with the unweighted fine control, it recovers
0.605191pp recall and adds0.088653pp IoU, but does not eliminate recall loss.

Five-class IoU/precision/recall (%) of the balanced candidate:

- background:98.205753 /99.393014 /98.798281;
- deck:94.805453 /97.473802 /97.193540;
- stay_cable:89.636149 /94.142319 /94.930720;
- tower:86.328338 /88.865847 /96.798254;
- foundation:65.596457 /67.635725 /95.605586.

Total mIoU86.914430%. This is reused calibration, not the74-view developer result.
The candidate adds49,828 deck pixels and removes320,460, correcting121,647 old
errors while harming167,510 previously correct predictions. Protected class
masks and foundation-rejection pixels stay exact. No111 check/74 developer
target metrics or final training were run.

Full runtime445.586s (0.123774h), including333.757s feature extraction,12.561s
two-arm point fitting,89.104s OOF inference. Peak CUDA allocated2,846.886MiB;
each head has9,089 parameters. Seed20260805, fixed splits, config, source snapshot,
six checkpoints and all219×2 predictions are hashed in the output seals. All
checkpoint reloads, parent tensor invariance, zero-residual and RGB checks pass.

The trained fold0 balanced head also passes native989×1320 and240×320 external
COLMAP/SIMPLE_RADIAL probes: frozen RGB/base semantic replay, zero residual and
protected masks are exact. Probe runtime14.363s. Candidate head SHA256:
`f8233834b6330138073de11dda473f4a2eb4403f1dbc7021b06a7ab963e4ab3f`.

Full result SHA256:
`68469e5b8d82a3dec91efaec616949d6e78cf34ce34893e078bfbabb165922ae`.
Output: `outputs/sem262/full-20260907-r1/`. Both improvements and harmed regions
are shown in the twelve2×nearest native crops in
`outputs/sem262/closeups-20260907-r1/`, with explicit ROI coordinates and hashes.

Interpretation: fine point features and class balancing each have measurable
matched benefits; they are not sufficient for the agreed precision/recall
tradeoff. Do not erase that research progress, declare success against SEM-161,
or keep scanning class weights. The next meaningful training-evidence test is
nested OOF input generation: current point heads see in-sample frozen-parent
features during fitting but out-of-sample parent features at evaluation. This
distribution mismatch is a hypothesis, not a proven cause. A nested experiment
must train inner teachers wholly within each outer training fold; simply reusing
another existing fold's predictions would leak outer held-out supervision.

Access terminology: "no check target metrics" does not mean those image files
were forbidden as training labels in other allowed folds. Every fold's fitting
role accesses are explicitly listed in its checkpoint/result; only its own
train names are permitted. Target-role scoring is limited to108 calibration
names after all predictions are sealed. This distinction also applies to SEM-261.

Calibration fold0/1/2 deck IoU gains are+1.456596,+0.710744,−2.679489pp on46/42/20
views respectively, computed from already sealed per-view confusions. This
spatial inconsistency is additional evidence against promotion and motivates
testing training-input distribution mismatch, without claiming it proves that
mechanism. Both matched-control and candidate predictions remain available.
