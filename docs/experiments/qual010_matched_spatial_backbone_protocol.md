# QUAL-010 matched spatial backbone comparison

Experiment ID: `QUAL-010-MATCHED-SPATIAL-BACKBONE`.

This is a read-only comparison of the already sealed RGB152/SEM396 AbsGS chain
and RGB153/SEM407 original-Standard chain. Both chains use
`spatial_confirmation_v1`, the exact 78,045-point training-only
initialization, RGB seed 0, semantic seed 20260805, 30k/6k endpoints, identical
semantic sample traces, and the same evaluation implementation.

The audit verifies all four parent result hashes, joins the 30 paired views,
reports train/confirmation and three-region RGB and semantic differences, and
records point count, training time, rendering time, and model size. It does not
train, tune, select a checkpoint, alter a prediction, or claim blind testing.

The previously stated decision rule is applied descriptively: prefer AbsGS
only if it materially improves the spatial confirmation result and the benefit
is not a broad regression across the other regions; otherwise prefer original
Standard for its lower cost. Because the initial evidence may be dominated by
one difficult region, promotion remains provisional until an additional
semantic seed or disjoint camera-only spatial folds test stability.

