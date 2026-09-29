# SEM-416-COMPONENT-POINT-DIAG

Completed 2026-09-21; prevalence gate did not support the conditional training. Parent SEM412;
seed20260805, `contest_dataset_v1.json`, `spatial_confirmation_v2` training side
only. Fixed 510 crops at fresh/3000/6000 states, actual matcher/mask-loss point
coverage. Output: `outputs/quality/SEM-416/r1`. Full binding and conditions:
[protocol](../../../docs/experiments/component_sampling_dropgaussian_protocol.md).

All 510 fixed crops were evaluated at fresh/3000/6000 states, with zero
confirmation/buffer image or label access and no parameter updates. Each
snapshot contains1529 visible small-component observations in the final branch.
Final loss sampling has100/83/83 observations with <4 effective points and
33/21/21 with zero points. Both late snapshots have only5.43% low coverage,
below the registered10% prevalence threshold. Final-branch matching has334/1529
low-coverage and75/1529 zero-coverage observations, unchanged across the fixed
snapshot RNG schedules. Matching coverage alone does not establish wrong
assignments, and low loss coverage is not a widespread cause on these crops.

All final/auxiliary branch coordinates and matched-class probabilities are
recorded in `coverage.jsonl`. A point counts when its bilinear contribution
from the particular full-image component is at least0.5. These observations
are crop/component incidences, not independent physical bridge instances.
Models were exactly unchanged after each snapshot. Runtime195.448 seconds.

SEM417/418 remain not run under their original support gate. The user's later
explicit instruction to test promising directions without excessive stopping
is recorded separately as SEM421/422: an unchanged-quota exploratory pair,
not a rewrite of this diagnostic or proof of widespread point starvation.
