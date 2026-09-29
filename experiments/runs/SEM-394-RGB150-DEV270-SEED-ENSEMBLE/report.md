# SEM-394-RGB150-DEV270-SEED-ENSEMBLE

- Status: completed on 2026-09-19; pre-registered read-only frozen dual-seed ensemble; **closed as a negative result** (`closed_negative_result`), not promoted.
- Parents: SEM384 (seed `20260805`) and SEM392 (seed `20260919`), both sealed 6000-step native-crop Mask2Former runs on frozen RGB150.
- Split: `dev30_local_composition_v1`, 270 train / 30 holdout labeled, 100 unlabeled; identical to both parents.
- Candidate: the single pre-registered fusion `P_ensemble = 0.5 * P_SEM384 + 0.5 * P_SEM392` in float64, smallest-class-ID tie rule; no training, no alternative weights, no TTA, thresholds, or post-processing.
- Protocol: `docs/experiments/sem394_seed_ensemble_protocol.md` (registered before any evaluation label access).

## Reproduction (hard requirement)

- Both parents were re-inferred on all 400 RGB150 views with the unmodified frozen recipe (native 768 window, stride 512, per-tile score normalization, float32 accumulation, count division, final normalization, cuda fp16 autocast; `src/quality_mask2former.py` SHA256 `debd14ba97e6e8b89db5e281ccc023b53159718a284b332fcce21813345a7d6e`, identical to both parents' sealed code hashes).
- SEM384 reproduction: `400/400` argmax predictions pixel-exact and byte-identical to the sealed parent predictions; median inference `0.436330 s`; peak allocated/reserved `1478.076171875 / 1968.0 MiB` (equal to the parent record).
- SEM392 reproduction: `400/400` pixel-exact and byte-identical; median inference `0.436355 s`; peak `1478.076171875 / 1968.0 MiB`.
- Per-view five-class probabilities (float32, lossless) were saved and hash-sealed for both parents before the ensemble stage; zero labels were read during reproduction and ensembling.

## Ensemble execution

- `0.5/0.5` float64 average over all 400 views; `0` exact-tie pixels; stored float32 probabilities reproduce the float64 argmax on `400/400` views (`0` mismatches).
- All 400 ensemble predictions were generated and hash-sealed before the first label was read.

## Holdout core results (30 views)

| System | five-class mIoU | foreground mIoU |
|---|---|---|
| SEM384 | 93.698606% | 92.276768% |
| SEM392 (primary baseline) | 93.872085% | 92.483224% |
| **SEM394 ensemble** | **93.840877%** | **92.448265%** |

- Train (270 views): SEM384 95.325224% / 94.299557%; SEM392 95.351882% / 94.332759%; ensemble **95.402741% / 94.394154%** (highest training fit, holdout regression).
- Holdout per-class IoU, ensemble vs SEM392: background `-0.016202 pp`, deck `-0.029843 pp`, `stay_cable` `-0.252558 pp`, tower `+0.060875 pp`, foundation `+0.081690 pp`.
- Holdout `stay_cable`: ensemble FP `67,720` / FN `112,568` (SEM392: 71,191 / 103,818; SEM384: 67,057 / 121,543); the average sits between the parents and loses most of SEM392's cable recall.
- Holdout 2px `stay_cable` boundary F1: ensemble `0.811863` vs SEM392 `0.812731` (`-0.086780 pp`, within the protection band); SEM384 `0.810989`.
- Groups (five-class mIoU, ensemble vs SEM392): regular-local `95.084022% -> 95.021008%` (`-0.063014 pp`), oblique-local `+0.030193 pp`, distant-local `+0.031635 pp`.
- Per-view holdout: vs SEM392 17 improved / 13 worse, corrected `12,977` / damaged `18,485` pixels; vs SEM384 22 improved / 8 worse, corrected `20,475` / damaged `10,749`.

## Paired bootstrap (SEM392 baseline, 30 holdout views, 10000 draws, seed 394)

| Metric | median | 95% interval | P(delta > 0) |
|---|---|---|---|
| five-class mIoU | `-0.029544 pp` | `[-0.134886, +0.071428]` | `0.2833` |
| foreground mIoU | `-0.033225 pp` | `[-0.157354, +0.087359]` | `0.2960` |
| `stay_cable` IoU | `-0.240580 pp` | `[-0.674589, +0.126579]` | `0.1090` |
| `stay_cable` boundary F1 | `-0.113149 pp` | `[-0.943154, +0.881773]` | `0.4050` |

## Pre-registered promotion gates

| Gate | Requirement | Observed | Verdict |
|---|---|---|---|
| G1 holdout five-class mIoU vs SEM392 | `>= +0.10 pp` | `-0.031208 pp` | FAIL |
| G2 holdout foreground mIoU vs SEM392 | `>= 0` | `-0.034959 pp` | FAIL |
| G3 any class IoU drop vs SEM392 | `<= 0.10 pp` | `stay_cable` `-0.252558 pp` | FAIL |
| G4 any group five-class drop vs SEM392 | `<= 0.15 pp` | regular-local `-0.063014 pp` | PASS |
| G5 `stay_cable` 2px boundary F1 drop | `<= 0.10 pp` | `-0.086780 pp` | PASS |
| G6 bootstrap median > 0 and P(delta > 0) >= 0.95 | positive | median `-0.029544`, P `0.2833` | FAIL |

Outcome: `closed_negative_result`. SEM394 is **not** worth training an RGB140 second seed for full-data ensemble validation under this protocol; no weight search, remedial fusion, or rerun under this ID is performed or authorized.

## Input integrity diagnostics (independent recomputation, 30 holdout views)

- SEM384/SEM392 prediction disagreement: `0.1633167%` of holdout pixels (matches the read-only reference).
- Only SEM384 correct: `0.0605269%`; only SEM392 correct: `0.0994245%`; disagreement with both wrong: `0.0033653%` (all match the reference).
- Unreachable GT-selected hard-oracle five-class mIoU: `94.5441796%` (reference ~`94.5300558%`; the small difference is the tie choice among wrong predictions on the ~1.1k both-wrong disagreement pixels). The oracle is a complementarity upper bound and input check only, never a candidate score.
- 600 independent parent confusion recomputations matched the sealed parent results exactly; both parents' aggregate train/holdout confusions and holdout boundary counts were reproduced exactly.

## Interpretation

The two seeds are genuinely complementary (each is uniquely correct on ~0.06-0.10% of holdout pixels, oracle bound ~94.54%), but uniform probability averaging mostly interpolates the two error profiles instead of harvesting the complementarity: it inherits part of SEM384's `stay_cable` weakness (IoU `-0.2526 pp` vs SEM392, FN `+8,750`) and gives back the gain as small tower/foundation improvements. The ensemble is better than the weaker parent SEM384 (`+0.142272 pp` five-class) but not than the stronger parent, and the training-fit improvement (`+0.050859 pp` over SEM392) does not transfer. This is a single fixed split and two frozen seeds; it does not prove all seed-ensembles fail, but it closes the pre-registered 0.5/0.5 candidate and does not justify an RGB140 second-seed training budget.

## Access and reproducibility

- Parent checkpoints, results, and predictions were only read; the SEM384/SEM392 directories were never written.
- The 270 train and 30 holdout labels were read only after the 400 ensemble predictions were hash-sealed; the 100 unlabeled views received predictions only.
- Config SHA256 `c2ae09f19a5f863c6966cb9572b8ae754173dad8e9cdfd24a09904f7cfab84c9`; protocol SHA256 `e817d929f7ed86157ccd830c7d46dfc924e4e59d8af422960bea6db941a5a936`; runner SHA256 `ec70afe3da6cbbb10ea5a0677ffb1200b17cf570e6c2076d86409ce70ae74435`.
- Reproduce results: SEM384 `95a98537d42e7fdb9c843f2286f35508bb314496297397514f742c2a96a7567f`, SEM392 `14085436777fd7913efacd9c316929a8a13eee30ff9146e87125a7ed3f05a6e1`.
- Ensemble result `4e2a82594059582ba60d258e3b59577c94373f390fe34628ccf79e731f916495`; evaluation result `d9b89cf24fea84fea26af1c65a849f2fbf32634366bb575f0ab18b30447a4973`; gate `28905803830b81c9c1a63cc8a54557067548a1713f063d816e1c6d57dacccd4b`.
- GPU: shared lock, exclusive compute check; both reproductions ran at peak `1478.076 / 1968.0 MiB`. Output volume ~3.8 GiB under `outputs/quality/SEM-394/r1` (ignored by Git).
- Python: `$ICSHM_OUTPUT_ROOT/r1/envs/r1-cuda128` (torch 2.11.0+cu128), the same toolchain that produced the parents.
- 13 task-specific unit tests passed; 257 JSON-compatible configs validated.

Rollback: delete nothing; the negative result and all artifacts stay. The default pipeline (RGB130+SEM371), the system candidate (RGB140+SEM378), the Dev30 baseline (SEM384), and SEM392 are unchanged. Any future ensemble work needs a new hypothesis, a new ID, and fresh authorization.
