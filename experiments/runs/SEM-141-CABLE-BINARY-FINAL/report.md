# SEM-141-CABLE-BINARY-FINAL — final cable-binary developer candidate

## Status

Completed on 2026-09-04 after SEM-140 passed its training-only OOF gate. The
all-219 final checkpoint and every engineering gate passed. On the explicitly
reused 74-view development set it exceeded the registered cable IoU, recall,
overall mIoU and difficult-subset targets, so it is recommended for official
hidden evaluation. SEM-111 remains the historical formal candidate and SEM-001
remains the independent fallback until hidden evaluation confirms generality.

## Frozen protocol

- Train one final 3,000-step SegFormer-B2 checkpoint on all 219 SEM-101 frozen
  rendered RGBs with the exact qualified SEM-140 seed, binary target, 960
  native crop, weighted CE, optimiser and augmentation.
- Freeze and reload the checkpoint before registering its audit hashes or
  opening the explicitly reused 74-view development labels.
- Use the OOF-frozen threshold 0.70. Copy SEM-001 deck/tower/foundation pixels
  exactly; the binary branch may decide only background versus `stay_cable`.
- Materialise all 74 predictions before labels are read. Relative to SEM-111,
  target at least +2 pp cable IoU, +3 pp recall, no mIoU loss, no more than
  1.5x background false positives, at most -0.5 pp on worst-10% and views
  140–146, exact protected pixels, checkpoint reload, RGB identity and the
  official external-camera output contract.

## Claim boundary

The 74-view set is already an explicit development set. Passing the target
only recommends the frozen candidate for official hidden evaluation; it does
not establish fresh generalisation or overwrite SEM-111/SEM-001.

## Final training evidence

The one authorised all-219 checkpoint completed 3,000 steps in 486.89 seconds
(0.13525 GPU-hours), with 5,570 MiB peak CUDA allocation. Loss was 0.721788 at
step 1, 0.008913 at step 3,000 and 0.021980 over the last 100 steps. The binary
training counts were 262,037,215 non-cable and 23,862,905 `stay_cable` pixels;
the frozen inverse-frequency weights were 0.545533 and 5.990472. Strict reload
passed before any development validation path opened.

The checkpoint SHA-256 is
`6858766c26599496b0021ee20da11552ad8c960547c4444aadd8c854fef432e0`;
the training result SHA-256 is
`4f6e7f8af515c42e8698fc299892970bf0239868e1ef80e3b0a9783a7096cd4b`.
The subsequently executed audit configuration, before its status field was
updated after completion, was frozen with SHA-256
`602ef66d78745dce681e6045cb03bcaf5c9672f545dd1cefb4bea4511e9cedc8`
before validation inference.

## 74-view developer result

All 74 cable probability maps and protected five-class predictions were
materialised before any development label was opened. With the OOF-frozen
threshold 0.70, the aggregate comparison is:

| Metric | SEM-111 | SEM-141 | Change |
|---|---:|---:|---:|
| mIoU with background | 0.760635 | 0.774899 | +1.426 pp |
| mIoU without background | 0.705681 | 0.722893 | +1.721 pp |
| `stay_cable` IoU | 0.638137 | 0.706983 | +6.885 pp |
| `stay_cable` precision | 0.860977 | 0.842841 | -1.814 pp |
| `stay_cable` recall | 0.711445 | 0.814334 | +10.289 pp |
| background→`stay_cable` | 347,412 | 447,859 | +100,447 (+28.91%) |
| worst-10% mIoU without background | 0.428629 | 0.445049 | +1.642 pp |
| views 140–146 mIoU without background | 0.758237 | 0.774075 | +1.584 pp |

The registered target was cable IoU at least +2 pp and recall at least +3 pp,
with no mIoU loss; all three passed by wide margins. The allowed background
false-positive ratio was at most 1.5x and the observed ratio was 1.289x.
Deck/tower/foundation IoUs remain exactly `0.900999/0.834870/0.448718` because
their SEM-001 prediction masks are byte-identical. The candidate improved
cable IoU, recall and non-background mIoU on 66, 70 and 66 of 74 views;
median per-view gains were +11.175, +12.828 and +2.794 pp respectively.

Visual/quantitative review of the fixed panels agrees with the aggregate: in
054, 140 and 183 the new branch adds respectively 3,903/4,790/4,097 true cable
pixels for only 55/542/229 added background pixels. The main recorded local
regression is 174, where cable IoU falls 3.59 pp because added background and
removed true-cable pixels offset 4,899 true additions. This remains a hidden
generalisation risk and is not used to retune the frozen recipe.

## Engineering result and decision

Checkpoint reload, exact protected predictions, external RGB byte identity and
the official `CAMERA_FROM_WORLD + SIMPLE_RADIAL` 1320x989 RGB / five-logit /
uint8 class-ID contract all passed. The audit result SHA-256 is
`4d003d02fbdf4245e97977e35ce96b9edc973b27f810232ca0532602ae6bef1c`.
The 74-file prediction manifest SHA-256 is
`eae2885c225e77f72248be228b27f2754a4efc6027472d13416bcc79181d624a`.

SEM-141 is therefore the strongest current **developer candidate** and should
be submitted to official hidden evaluation without further 74-view tuning.
It is not renamed a fresh blind result. SEM-111 and SEM-001 remain intact as
the two rollback levels.
