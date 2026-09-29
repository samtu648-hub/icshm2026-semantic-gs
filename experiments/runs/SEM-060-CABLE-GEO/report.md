# SEM-060-CABLE-GEO — training-side stay_cable 3D geometry gate

- Status: **completed; `drop_geometry_not_reliable`**.
- Parent: frozen `SEM-001`; `SEM-004-DIAG` supplied motivation only.
- Run HEAD: `49cdcdb1c9915a9544d568754af95ea20a85ec9c` with pre-existing user changes preserved.
- Fit-freeze tracked-diff / porcelain SHA-256: `87b19f9376a0833302a174d74b1581cb2f54417a3f125676caf4003b04e65141` / `8e4117875ade434ebe30626536d080861a8ddf31e104733c92d7daddd2dd9e44`.

## Frozen protocol and data boundary

The deterministic training-only split selected source fold 2 by the predeclared rule “largest geometry-fit cohort, then lowest fold ID”: 177 geometry-fit, 40 geometry-check and 2 local guard views. The minimum fit-to-check camera-centre distance is `0.5873516`; there are no cross edges inside the inherited `0.5220235` guard radius. Split/config SHA-256 are `86d457ff2e2cbc5d40727770fc6b2c642825db9da0375028bad260f40ffd8d87` and `6155506f011541a6f243a310c94ca6648bb2bf837a956ba27299d6f622dee0e3`.

Only SEM-001 training-side PINHOLE labels, official `CAMERA_FROM_WORLD` cameras, released `images.txt` tracks and the camera graph were used. The fixed 74-view validation set, parent guard labels, 100 unlabeled images, pseudo-labels, SegFormer predictions, hidden-test inputs and external depth models were not loaded. Geometry-check was opened only after the fit candidates/curves and their hashes were frozen.

The configuration fixed mask morphology, Zhang-Suen centerlines, local widths/endpoints/branches, strong-covisibility and baseline rules, mutual epipolar matching, three-view pair-RANSAC plus DLT refinement, candidate voxelization, piecewise straight 3D curve fitting, evaluation tolerances and all gates before check evaluation.

Registered commands:

```bash
export ICSHM_DATA_ROOT="<external Contest Dataset root>"
export ICSHM_OUTPUT_ROOT="<Git-ignored output root>"
python3 scripts/run_sem060_cable_geometry.py \
  --config experiments/runs/SEM-060-CABLE-GEO/config.yaml --stage fit
python3 scripts/run_sem060_cable_geometry.py \
  --config experiments/runs/SEM-060-CABLE-GEO/config.yaml --stage check
```

## Geometry-fit result

Across 177 fit views, extraction retained 602 mask components, 601 skeleton components, 311,031 skeleton pixels, 47,461 deterministic samples, 1,536 endpoints and 131,749 branch pixels. Released tracks contributed 3,035 near-centerline observations across 1,891 track IDs; 142 three-view track groups were considered and 70 survived as track seeds. Together with mutual epipolar matches, the run formed 55,809 seeds, retained 35,408 after voxelization/cap and accepted 24,825 robust candidates.

Every retained candidate has at least three independent fit cameras; support ranges from 3 to 24 views. The fit produced 79 piecewise straight curves of total length `175.1947` official units. Candidate/curve/freeze SHA-256 are:

- candidates: `1f8fe0f3f4cc11733381662ad5e96753a4ee4d05f4a07262d6ede1f5b181fa75`;
- curves: `c560f2b0700e4374064e4622e7b62439cfbe6c2d75c91300b55ac2974d453853`;
- fit freeze: `a3f2d9539828deeb2e4e1404c848488d1e883132ce0af3f0b0630ee9efe62181`.

Pose round-trip error was `5.33e-15`, but the scale audit already failed: the farthest retained candidate is `46.2430` units from its nearest fit camera while the fit-camera diameter is only `8.9776`; the frozen limit is three diameters. Candidate bounds (`[-40.715,-1.157,-24.431]` to `[35.183,17.602,38.385]`) also reveal implausible long-range triangulations.

## Independent 40-view geometry-check

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| minimum independent support views | 3 | ≥3 | yes |
| exact precision / recall / IoU | 0.022891 / 0.114893 / 0.019459 | report-only | — |
| 2 px precision | 0.104230 | ≥0.90 | **no** |
| 2 px recall | 0.410899 | ≥0.20 | yes |
| 4 px precision / recall | 0.178287 / 0.561106 | report-only | — |
| 8 px precision / recall | 0.308899 / 0.732382 | report-only | — |
| check views with ≥2% recall at 2 px | 40/40 (1.0000) | ≥0.50 | yes |
| median projected-curve error | 16.553 px | ≤2 px | **no** |
| P90 projected-curve error | 65.437 px | ≤4 px | **no** |
| predictions farther than 8 px | 310,003 / 448,564 (69.1101%) | ≤10% | **no** |
| coordinate/scale/depth/leakage bundle | scale failed; other audited items passed | all pass | **no** |

The high recall and 40/40 coverage are not a success: they came from projecting far too many incorrect lines. Even the deterministic “success” cases 052/049/051 have 2 px precision only 0.2901/0.2210/0.2175. Failure cases 242/241/243 have 2 px precision 0.0313/0.0363/0.0387 and median errors 50.09/39.00/45.31 px. Git-ignored blank-background overlays use green for GT centerline, red for projected curves and white for overlap; visual review confirms many parallel or crossing red lines far from the green cable.

Check result/per-view/artifact-manifest SHA-256 are `bccbef80a7d9efbff37d6677804b41130054d356f20c19bbcb6b6fec0bc21f75`, `5ac4ab7fe67d3f34d491167b3e307d52d8b45ed969cb8d22f83598d35758bedb` and `441520e295b72d3bcf7be59ad940d76db18eec2fb405a372ec1fe61002cc75f8`. The complete ignored evidence is about 30 MiB.

## Resources, isolation, and decision

Fit/check wall times were `1868.07 s` and `33.81 s`; no GPU or optimizer was used. Sampled host fit RSS was about 1.17 GiB. No model checkpoint, PLY, RGB image or logits were created. The six frozen RGB-030-FULL artifact hashes and the SEM-001 semantic checkpoint hash were rechecked after SEM-060 and remain exactly the registered values; SEM-060 never loaded or rendered RGB, so no RGB path was altered.

Five of eight mandatory geometry gates failed: 2 px precision, median error, P90 error, coordinate/scale audit and far-false-positive fraction. The correct conclusion is **geometry failure**, not an engineering failure and not a semantic quality result. Per pre-registration, thresholds, fit/check split, curve fitting and seed are not changed; fixed 74-view is not opened for this method; `SEM-061-CABLE-EQ`, `SEM-062-CABLE-PF` and `SEM-006` are not implemented or run. `SEM-001` remains the frozen semantic baseline. This stage only tested semantic cable geometry capacity and made no RGB cable-reconstruction claim.

## Verification

- `python3 scripts/validate_configs.py`: 60 configs validated;
- `python3 -m pytest -q tests`: 52 passed, 9 skipped (Torch-only tests are intentionally skipped by the lightweight interpreter);
- `bash scripts/repo_health.sh` with the released dataset root: all seven stages passed, including 61 unittest cases with 9 expected skips and the 300/100/400 data check;
- `git diff --check`: passed;
- the required camera pose/ray, epipolar, positive-depth three-view triangulation, curve tangent, centerline and leakage tests are in `tests/test_cable_geometry.py`.

An initial unscoped `python3 -m pytest -q` collected the Git-ignored `third_party/map-anything` upstream tests and stopped during collection because the lightweight system Python has no Torch. It did not execute a GeometryPrior model or alter artifacts. The repository-scoped `tests/` run and the repository's fixed unittest health entrypoint both passed; the failed broad collection is retained here rather than represented as a model/test result.
