# QUAL-011-SEMANTIC-SEED-REPLICATION-AUDIT

Registered on 2026-09-20 as a read-only audit of the two sealed semantic-seed
pairs. It performs no training, inference, threshold search, or label access.

Status: completed on 2026-09-20; all pre-registered direction gates passed.

## Two-seed result

| Confirmation metric | Seed 20260805 delta | Seed 20260919 delta | Two-seed mean delta |
|---|---:|---:|---:|
| Five-class mIoU | +17.224317 pp | +15.669094 pp | +16.446706 pp |
| Foreground mIoU | +21.093956 pp | +19.184885 pp | +20.139420 pp |
| `stay_cable` IoU | +15.688421 pp | +7.918670 pp | +11.803546 pp |
| `stay_cable` recall | +8.886116 pp | +8.130220 pp | +8.508168 pp |
| 2px boundary F1 | +17.554241 pp | +9.884617 pp | +13.719429 pp |

The pre-registered gate requires positive AbsGS-minus-Standard differences in
five-class mIoU, `stay_cable` IoU, and boundary F1 for both overall confirmation
and region 1 in each seed. All 12 conditions pass. Region-1 five-class/cable/
boundary gains are +39.044441/+50.416176/+46.133311 pp in the first seed and
+37.101256/+38.895006/+39.058851 pp in the second.

This supports keeping AbsGS as the final reconstruction-backbone candidate.
It does not establish universal regional dominance: second-seed region-2
boundary F1 is 6.626270 pp lower. It also does not remove dependence on one
historically exposed spatial split and one RGB seed; a disjoint spatial fold is
the next high-value generalization experiment.

Machine result SHA256:
`815e573a92fd345512fb73afcb95191d4c5acaefbb9f6dbcc20b0879069cf1f6`.
