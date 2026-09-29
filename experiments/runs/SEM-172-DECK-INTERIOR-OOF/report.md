# SEM-172-DECK-INTERIOR-OOF — interior-emphasised deck binary model

## Status

Completed on 2026-09-06. The smoke and three final-3000 OOF folds completed,
but all 12 calibration thresholds failed; the 111-view check stayed unopened.

## Question

SEM-170 and SEM-171 show that deletion and image-space morphology cannot
safely repair the remaining fragmented deck. Can the deck binary model itself
rank true deck-interior pixels above cable crossings and nearby background?

## Single change

Repeat SEM-160's exact three-fold spatial OOF recipe, but multiply the
inverse-frequency cross-entropy pixel loss by 2 only on ground-truth deck
pixels surviving an 8-pixel binary erosion. Normalise by the effective
class-weight-times-spatial-multiplier sum, so multiplier 1 exactly reproduces
the original weighted-CE mean. The erosion radius and multiplier are fixed and
will not be swept.

Architecture, pretrained revision, seed, rendered RGB, split, crop,
augmentation, optimiser, learning rate, 3,000 steps, final-only checkpoint,
sliding inference and one-way SEM-150 fusion remain identical to SEM-160.

## Qualification

The 108-view calibration half alone selects a threshold from SEM-160's frozen
grid. Relative to SEM-160's selected OOF result, calibration and the independent
111-view check both require non-negative mIoU, at least +0.1 pp deck IoU and
recall, at least 95% deck precision, no more than 1.10 times background-to-deck
false positives, and exact preservation of every SEM-150 protected pixel.

The reused 74-view development set, external camera, real RGB, unlabelled and
hidden inputs remain unavailable. A gate failure drops this branch. A pass only
permits a separately registered full-data model; SEM-161 stays frozen.

## Result

The three folds trained for 1,477.59 seconds in total (0.41044 GPU-hours),
peaked at 5,575 MiB and produced 95/84/40 mutually exclusive probability
maps. All three final checkpoints reloaded strictly.

Threshold 0.99 had the highest calibration deck IoU. Relative to SEM-160 it
changed:

- mIoU: -0.0052 pp
- deck IoU: `0.9453013 -> 0.9451196` (-0.0182 pp)
- deck precision: `0.9562807 -> 0.9570915` (+0.0811 pp)
- deck recall: `0.9880001 -> 0.9869379` (-0.1062 pp)
- background-to-deck false positives: 0.98639 times SEM-160

Thus the fixed interior loss produced a slightly more precise but less complete
ranking, the opposite of the required continuity gain. The check and reused
74-view development labels remained unopened.

Result SHA-256:
`41340b7629133c5e6141f6c4014178631918c8d9f2143a6ce587823a6f0613a4`.

## Conclusion

Drop the interior-weighted model as a standalone replacement. It is close to
SEM-160 and has a complementary precision/recall shift, so its frozen OOF
probabilities may be tested once in a separately registered ensemble. No loss
radius, multiplier, step or checkpoint search is allowed. SEM-161 remains the
best qualified candidate.
