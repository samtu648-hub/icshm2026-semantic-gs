# QUAL-017-COMPONENT-SEED-CONFIRMATION

Completed2026-09-21; replication failed. Fixed second-seed confirmation of
the positive exploratory quota result; no search or combined method.
[Protocol](../../../docs/experiments/component_seed_confirmation_protocol.md).

## Two-seed result, candidate minus matched control

| Metric | Seed20260805 | Seed20260919 | Mean delta |
|---|---:|---:|---:|
| Foreground mIoU | +0.261825pp | -0.329470pp | -0.033823pp |
| Five-class mIoU | +0.215836pp | -0.263659pp | -0.023912pp |
| stay_cable IoU | +0.585769pp | -0.003865pp | +0.290952pp |
| stay_cable precision | +0.317626pp | +0.760821pp | +0.539223pp |
| stay_cable recall | +0.377915pp | -0.901140pp | -0.261613pp |
| 2px boundary F1 | +0.576574pp | +0.197429pp | +0.387001pp |

Second seed fails foreground improvement and the <=0.5pp class-IoU loss guard
(foundation-1.127419pp); foreground decreases in all three regions. A positive
boundary mean cannot override those failures. Whole-class/whole-image utility
does not reliably follow the repeated small-component recall improvements:
small-cable TP gains1802/1196 pixels and small-foundation TP gains611/647,
using the same15257/23992 GT pixel denominators. These are component observations
across images, not counts of independent physical bridge instances.

Decision: close the fixed quota method, preserve its useful scale trade-off
evidence, and keep the original system. No third seed, quota reduction, alternate
area threshold, fusion, new loss or all-data migration follows this result.
The original SEM416 prevalence diagnosis is not rewritten by the first seed's
positive outcome. This remains a reused development fold, not a blind result.

## Execution and verification

The second pair completed11 stages, all exit0, in3629.011 seconds(60.484 minutes).
Initial650 tensors and6000 actual samples are paired, with a sealed same-seed
initialization reference. Both endpoints sealed before evaluation. Independent
recount verified570 confusion matrices/boundary counts and758 prediction hashes;
before/after frozen RGB154 three-camera replay is exact. All checkpoint tensors
finite. The four full semantic fits across both seeds total24000 successful
updates; there were no failed training attempts.

Audit: `outputs/quality/QUAL-017/r1/audit-result.json`, SHA256
`b1ce4c28b0a8c5ea2868488f8acff1ef02cc654addcfcf243f5a3c1befc007a1`.
Two-seed result: `outputs/quality/QUAL-017/r1/two-seed-result.json`, SHA256
`c6b05145808c3588c459f210933891d4cbb1e3a9ecbc468aedf6c0b87e22948c`.
Commands, source/config bindings, timing and exit statuses are in the same run
directory. Rollback requires no restoration: retained RGB140/SEM378 and all
historical checkpoints remain hash-exact.
