# SEM-171-DECK-CONTINUITY-OOF — protected deck-gap recovery

## Status

Completed on 2026-09-06. All 28 registered calibration candidates failed, so
the 111-view check and reused 74-view development labels remained unopened.

## Question

SEM-170 proved that the frozen binary model cannot safely delete existing deck.
Can spatial closure identify internal cracks and narrow gaps where lower binary
probability is useful, while keeping the entire existing deck mask immutable?

## Frozen protocol

No network is trained. Start from SEM-160's selected additive 0.99 OOF
prediction. For each registered elliptical closing radius, only pixels newly
enclosed by closing the existing class-1 mask are candidates. A candidate is
added only when it is background both after SEM-160 and before the foundation
gate, and its frozen OOF deck probability clears the registered threshold.

The 108-view calibration half selects one registered radius/threshold pair.
Relative to SEM-160, calibration and the one-shot 111-view check both require
non-negative mIoU, at least +0.1 pp deck IoU, at least +0.2 pp deck recall,
deck precision of at least 95%, at most 1.10 times the existing
background-to-deck false positives, and exact preservation of every existing
deck pixel and protected prediction.

## Boundary

This is training-only OOF qualification. The 74-view development set, external
camera, unlabelled images, hidden inputs and real RGB remain unavailable.
Failure closes this post-processing route. Passing permits only a separately
registered final application of the frozen pair; SEM-161 remains unchanged.

## Result

The maximum calibration deck IoU used radius 3 and probability threshold 0.98.
Relative to SEM-160 it changed:

- mIoU: `0.8687901 -> 0.8687713` (-0.0019 pp)
- deck IoU: `0.9453013 -> 0.9451993` (-0.0102 pp)
- deck precision: `0.9562807 -> 0.9561157` (-0.0165 pp)
- deck recall: `0.9880001 -> 0.9880648` (+0.0065 pp)

Only 1,847 pixels were added: 486 true deck, 425 background, 935
`stay_cable`, one tower and no foundation pixels. Thus spatial closing mostly
selected cable crossings and other non-deck gaps rather than missing deck.
Every existing deck and protected prediction remained exact.

Result SHA-256:
`9f9213e7dd32fe4428e90ae6eafcdfd9597df8dda051fc330b533d7d663da14d`.

## Conclusion

Drop morphological continuity recovery. It is safe in the narrow sense that
it never deletes deck, but its spatial predicate is not semantically selective:
the remaining enclosed gaps are dominated by `stay_cable`. Further work must
change the learned deck evidence rather than add another view-space closing or
threshold rule. SEM-161 remains frozen.
