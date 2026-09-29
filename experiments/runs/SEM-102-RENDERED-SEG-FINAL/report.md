# SEM-102-RENDERED-SEG-FINAL — final rendered-RGB SegFormer and one 74-view audit

## Status

Completed; formal retention gate failed. `SEM-101-RENDERED-SEG-OOF` passed its
five training-only OOF qualification gates, so this was the single separately
registered final model allowed by ADR-021. It is a view-conditioned 2D semantic
layer on frozen Scaffold RGB, not a modification of Scaffold geometry or the
SEM-001 sidecar. Three protection gates failed on the once-only audit, so the
fallback remains `RGB-030-FULL Scaffold-GS + SEM-001`.

## Frozen protocol

- Reuse the exact, hash-addressed 219-image rendered RGB dataset qualified by
  SEM-101. Train a single SegFormer-B2 checkpoint for final 3,000 iterations
  with the identical seed, optimiser, loss, class-weight rule, crop, flip and
  sliding inference. The only difference from SEM-101 is all-219 final fitting
  instead of three OOF folds.
- The final trainer has no 74-view, unlabelled, real-RGB or external-camera
  input. It writes and reloads the final checkpoint before validation can be
  rendered.
- Only after reload, render exactly the frozen 74 validation cameras from
  RGB-030-FULL without labels or reference RGB. Generate every prediction and
  write its manifest before opening validation labels for one fixed audit.
- Run the official `CAMERA_FROM_WORLD + SIMPLE_RADIAL` interface separately:
  output official-grid RGB, five-channel logits and a uint8 class-ID PNG. It
  never reads a reference image or label.

## Formal retention gate

The one audit compares its fixed 74-view metrics against the recorded SEM-001
baseline. mIoU, `stay_cable` IoU and recall must not decline; deck, tower and
foundation may each fall by at most 0.5 pp; background→`stay_cable` must fall
at least 1%; and worst-10% and views 140–146 mIoU may each fall by at most
0.5 pp. Frozen RGB artifact hashes and repeated pixels must be identical,
checkpoint reload must pass, and the external camera output must meet the
five-class 1320×989 contract. A failed gate records a candidate failure and
leaves SEM-001 as the fallback; OOF/74-view results cannot select another
hyperparameter or retrain this experiment.

## Executed result

The fixed all-219 final model completed 3,000 iterations (seed `20260805`) in
300.30 seconds (0.0834 GPU-hours), with 3,203 MiB peak allocated memory. Its
final checkpoint reloaded through the registered trainer. The checkpoint
SHA-256 is `88e3084db7b6394ab8253fea2ac129d68d81053af50fc3b38556f1288972763c`;
the training result SHA-256 is
`2ccae7090ee1a9a980d7d3762cb946556bb6ea59cca7c35e52027b1f492dbacd`.

Validation RGB was rendered in three label-free shards after checkpoint freeze.
Exactly 74 predictions were written before labels were opened. The validation
render manifest SHA-256 is
`a805ab84e367090b501b8dc813469dc7f609d0fd3b9ef80a2521abd236e87ec4`; its
three repeated frozen-RGB samples (`054.png`, `140.png`, `183.png`) have equal
artifact and PNG hashes and maximum pixel difference zero. The audit result
SHA-256 is `7ee7d21b36b84c8a16ff72153d64ddf254994dd0c16f918b6b550821acca03a7`.

| Metric (74 fixed views) | SEM-001 baseline | SEM-102 final | Gate |
|---|---:|---:|---|
| mIoU with background | 0.616103 | 0.750301 | pass |
| `stay_cable` IoU / recall | 0.068759 / 0.368036 | 0.685329 / 0.810826 | pass |
| background→`stay_cable` pixels | 13,166,331 | 520,271 (−96.05%) | pass |
| deck IoU | 0.900999 | 0.876788 (−2.42 pp) | **fail**; limit −0.5 pp |
| tower IoU | 0.834870 | 0.741701 (−9.32 pp) | **fail**; limit −0.5 pp |
| foundation IoU | 0.448718 | 0.463487 (+1.48 pp) | pass |
| worst 10% mIoU without background | 0.423440 | 0.351778 (−7.17 pp) | **fail**; limit −0.5 pp |
| views 140–146 mIoU without background | 0.601213 | 0.750786 | pass |

The high global and `stay_cable` gains therefore do not qualify the candidate:
it loses too much deck/tower accuracy and is unstable on its hardest views.
Representative four-panel evidence is stored for `054.png`, `140.png`, and
`183.png` under the gitignored validation audit root. Those panels show a
largely correct bridge silhouette but thin-cable, deck-boundary and local-tower
errors, consistent with the failed protection gates.

The official external camera contract also passed after the final audit:
`CAMERA_FROM_WORLD + SIMPLE_RADIAL`, 1320×989 RGB, float32 `[5,989,1320]`
logits and uint8 `0–4` semantic PNG with valid simplex probabilities. Its
evidence SHA-256 is
`a63003346b5dd75b39bb6a8dc805bd7745f2fa65d15d3965462b2d06d7367f76`.

One attempted validation-render shard under the semantic training environment
stopped before rendering because its Scaffold dependency set lacked `colorama`.
The registered renderer was then run in the existing frozen RGB runtime, where
all three label-free shards succeeded. This operational dependency correction
did not alter model, data, split, frozen RGB or the audit sequence.

## Decision

`SEM-102` is **drop / show-only**. Per the preregistered protocol, no threshold,
architecture, loss, seed, checkpoint, or retraining choice may be made from
this audit, and the 74-view set will not be reopened for this experiment. The
only retained submission fallback is `RGB-030-FULL Scaffold-GS + SEM-001`.
