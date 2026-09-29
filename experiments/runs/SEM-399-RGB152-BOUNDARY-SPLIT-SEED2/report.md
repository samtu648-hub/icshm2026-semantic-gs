# SEM-399-RGB152-BOUNDARY-SPLIT-SEED2

- Status: completed; first-seed boundary result not replicated; current recipe closed without promotion.
- Parent: SEM-398 positive mechanism candidate.
- Sole replication variable: global seed `20260921` instead of `20260920`.
- All method settings, data roles, budgets, confirmation-access rules, and promotion gates remained frozen.
- Both endpoints were sealed before the 30 confirmation RGB images and labels were opened.

## Mechanism and matching

Native preflight passed RGB152 PNG replay, five-class probability normalization (`1.43e-6` maximum error), finite nonzero Class ID 2 gradient, semantic-to-geometry gradient blocking, and the 4,096 split capacity gate. The common stage completed 2,560 semantic updates plus independent 256-view semantic and 353-view RGB evidence passes.

The second seed again produced 6,067 jointly eligible parents, 2,740 control/candidate overlaps and 1,356 different selections. Across seeds, the RGB control selected exactly the same 4,096 IDs; semantic candidates shared 4,093/4,096 IDs, with only three IDs replaced. Thus the semantic-conflict ranking itself was highly stable. Both arms ended with 793,377 Gaussians and used byte-identical restoration sampling files within seed2.

## Fixed results

| Role | Control five-class mIoU | Candidate | Delta | Control cable boundary F1 | Candidate | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 256 training fit | 77.225347% | 77.340368% | +0.115021 pp | 33.796149% | 33.920007% | +0.123858 pp |
| 30 spatial confirmation | 50.670760% | 51.379519% | **+0.708760 pp** | 19.283154% | 19.388328% | **+0.105174 pp** |

On confirmation, candidate minus control foreground mIoU was `+0.878413 pp`, RGB PSNR/SSIM was `+0.160151 dB/+0.001159`, but `stay_cable` IoU was `-0.085479 pp` and precision was `-1.274886 pp`; recall was essentially unchanged (`-0.004244 pp`). Region mIoU changed by `+0.717102 pp`, `-0.078549 pp`, and `+1.875873 pp` for regions 1/2/3. Region boundary F1 changed by `-0.179959 pp`, `-0.057727 pp`, and `+0.732986 pp`.

The RGB, SSIM and overall five-class mIoU gates passed. The required `+0.5 pp` boundary gate, cable-IoU retention, and cable-precision retention failed. Therefore `promotion_authorized=false` and the frozen decision is `replication_failed_closed`.

## Cross-seed interpretation

The two-seed mean candidate-minus-control deltas remain positive: five-class mIoU `+1.761408 pp`, cable boundary F1 `+0.909186 pp`, cable IoU `+0.213327 pp`, cable precision `+0.115269 pp`, and PSNR `+0.133382 dB`. Those means do not override the per-seed all-gates-required rule. With nearly identical semantic parent IDs but substantially different boundary/precision outcomes, the current native random child placement and short restoration path are a material source of variance.

The result narrows the claim: semantic conflict consistently provides some overall semantic/RGB value over RGB-only ranking in these two seeds, but the specific thin-boundary and false-positive benefit is not stable enough to promote. No third seed, split-budget/threshold/prior/restoration scan, stronger-teacher experiment, or all-data migration is authorized from this result. RGB152/SEM396, RGB140+SEM378, and RGB130+SEM371 retain their prior roles.

Sealed endpoint SHA256 values are control `760c5373d46d9f80bfeb0e6f8696d4656a955ea5ccb0f698d51694870df35e6c` and candidate `155f847b68dd9b5f5d04f0a94938c22e084a15897bc0b46d0cc46c4f0a32bffe`. The complete audit is `outputs/quality/SEM-399/evaluate-r1/result.json`.
