# QUAL-015-COMPONENT-EXPLORATORY

Completed2026-09-21; first-seed exploratory result positive. User authorized direct limited
experimentation even where prevalence evidence is weak. SEM416/417/418 retain
their original negative/not-run states. No settings are changed to rescue them.
This is an exploratory paired test, not validation of a broad sampling-deficit
hypothesis. Output: `outputs/quality/QUAL-015/r1`.
[Protocol](../../../docs/experiments/component_exploratory_protocol.md).

## Matched seed20260805 result

- Foreground mIoU87.414385%→87.676210%: +0.261825pp.
- Five-class mIoU89.793529%→90.009365%: +0.215836pp.
- Cable IoU83.457819%→84.043589%: +0.585769pp.
- Cable precision/recall:+0.317626/+0.377915pp.
- Boundary F174.848952%→75.425527%: +0.576574pp.
- Foreground region deltas:+1.029626/-0.033385/-0.002378pp.
- All first-seed retention checks pass, including the four-class IoU protection.

The execution record contains11 stages, all exit0. Full pair took
3640.482 seconds (60.675 minutes), including discarded preflight, two6000 fits,
sealed evaluation, RGB checks and570 independent PNG confusion/boundary recounts.
All758 prediction hashes and all scored label identities matched. RGB154 and
production RGB140/SEM378 artifact hashes remain unchanged. Native criterion
tests verified disabled loss/gradient/RNG exactness and candidate global RNG
preservation using real Mask2Former point sampling; the hook adds no inference
computation.

Source/config snapshots, commands and exit statuses:
`outputs/quality/QUAL-015/r1`. Audit SHA256
`9fadb52256ac9c9c1b70fe867325a06a30807d7ff8f9b518c5210aab6a398a73`.
The second seed under [QUAL017](../QUAL-017-COMPONENT-SEED-CONFIRMATION/report.md)
failed; final method decision is not promoted. Both seeds' small-component
recall improvements are preserved as a mechanism trade-off, not a stable
overall-system improvement. Rollback uses the unchanged RGB140/SEM378 system.
