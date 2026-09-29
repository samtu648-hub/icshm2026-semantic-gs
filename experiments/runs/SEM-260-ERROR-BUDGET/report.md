# SEM-260-ERROR-BUDGET

Status: completed diagnostic on the reused 108 calibration views, seed0,
Git49cdcdb1c9915a9544d568754af95ea20a85ec9c plus recorded worktree source hash.
No new check/developer targets were read. Frozen inputs are content-addressed.

SEM-160 baseline has 90,098 deck FN and 339,142 deck FP. Of those, 95.4583%
and 71.0929% respectively lie within four pixels of a GT boundary. 54,054 FN
(59.9947%) and all deck FP lie in the allowed bidirectional background/deck lane;
36,044 FN are protected. No protection-mask dilation was applied.

The GT-assisted mutable-lane oracle raises deck recall from98.8000% to99.5199%
and IoU from94.5301% to99.5199%. This is an unattainable oracle, not a candidate.
The recall gate of+0.5pp therefore requires recovering about69.5% of mutable FN
without net losses. It is demanding but not mathematically impossible. We keep
the agreed promotion gate, while allowing both removal and addition of deck.

Five-class IoUs on this diagnostic partition: background98.3040%, deck94.5301%,
stay_cable89.6361%, tower86.3283%, foundation65.5965%. These are NOT the74-view
developer values. Protected prediction masks also lock errors in other classes;
a later class-specific experiment may revisit them under a new protocol.

Runtime9.397s; CPU peakRSS173.438MiB; no CUDA model loaded. Native comparison
panels and deck crops are in the Git-ignored output directory. RGB derivatives
are descriptive only, not a classifier of recoverable structure. The selected
028 panel visibly retains an edge near the deck; it cannot establish accuracy
for all examples. Result/source/config/input/panel hashes accompany the output.

Output: `outputs/sem260/full-20260907-r1/`. Formal fallback unchanged.
