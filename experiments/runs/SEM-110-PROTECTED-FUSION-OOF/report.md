# SEM-110-PROTECTED-FUSION-OOF — protected two-stream fusion diagnostic

## Status

Completed; training-only OOF gate passed. This is a qualification experiment,
not itself a replacement model and not a use of the fixed 74-view set. It responds to
SEM-102's specific failure mode: its rendered-RGB 2D output improved
`stay_cable` and background false positives but degraded deck, tower and hard
views. SEM-110 therefore preserves every SEM-001 prediction of deck, tower and
foundation bit-for-bit, and permits a fixed 2D teacher to decide only between
background and `stay_cable` elsewhere.

## Frozen protocol

- Load the hash-addressed, frozen SEM-001 checkpoint only to render exactly its
  219 authorised training-camera class maps. This extractor has no labels,
  validation, unlabelled, real-RGB or external-camera argument.
- Reuse the completed SEM-101 three-fold spatial OOF class probabilities: the
  2D teacher did not train on the image it predicts. Validate all 219 file
  hashes and the frozen OOF view assignment before labels are opened.
- Partition the training filenames into deterministic, label-independent
  calibration/check halves. Evaluate the fixed class-2 threshold grid
  `0.20/0.35/0.50/0.65/0.80` on calibration only. Choose the candidate with
  greatest background→`stay_cable` reduction among those passing every gate;
  ties select the higher threshold. Check labels are read once only after this
  selection, and never choose a threshold.
- A fused pixel is exactly its SEM-001 class when that class is deck, tower or
  foundation. All remaining pixels are emitted as only background or
  `stay_cable`, using the SEM-101 OOF class-2 probability. RGB, Scaffold
  geometry/color/opacity, SEM-001 logits and the SegFormer checkpoints are not
  modified.

## Qualification gate

On both the selected calibration candidate and the once-only check result:

- protected SEM-001 prediction pixels must be exactly unchanged;
- mIoU including background must not decline against the rendered SEM-001
  training-side map;
- `stay_cable` IoU must be at least 0.65 and recall at least 0.90;
- background→`stay_cable` must decrease by at least 80%.

If calibration has no passing threshold, check labels remain closed. If check
fails, SEM-110 is recorded as a training-only failure and does not access
74-view. If check passes, it can justify—not automatically execute—a separately
registered final candidate with a new access protocol. In every case the
submission fallback remains `RGB-030-FULL Scaffold-GS + SEM-001`.

## Executed result

The frozen SEM-001 checkpoint was rendered over all 219 authorised training
cameras in six inference-only shards. Its completed render manifest SHA-256 is
`fc2b3008b38c6c2196cdd7cdeab6d369d771770552b01c7c7d6c51fba042ae9a`.
It records 219 class maps and proves that labels, fixed validation, unlabelled
images, real RGB and external-camera modes were all closed during rendering.

The deterministic name hash partition produced 108 calibration and 111 check
views. Every fixed threshold passed calibration; threshold 0.80 had the largest
background→`stay_cable` reduction (95.68%) and was selected without inspecting
check labels. On the independent check half, it retained `stay_cable` IoU
0.871190 and recall 0.945650, raised mIoU from 0.680865 to 0.807461, and reduced
background→`stay_cable` from 16,573,599 to 1,113,641 pixels (−93.28%).
Deck, tower and foundation prediction masks were exactly unchanged for every
pixel, not merely within a metric tolerance.

The result SHA-256 is
`e7d3d22ffd7a0f10db3a0a0abdddbe33467770fa2957a56b5c12be1fa90746ff`.
The GT / frozen SEM-001 / fused / error panels for `001.png`, `017.png`, and
`031.png` visually show removal of the large background yellow envelopes while
the normal main cables, suspenders, deck, towers and foundation predictions are
preserved.

## Decision

SEM-110 passes its training-only qualification and permits one separately
registered final application of its fixed threshold. It does not authorize any
threshold change, retraining, or a new rule in this experiment. The fallback
remains SEM-001 until that distinct final audit passes.
