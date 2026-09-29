# QUAL-009 released-dataset profile protocol

Experiment ID: `QUAL-009-DATASET-PROFILE`.

## Purpose

Produce a paper-facing, read-only description of the released IC-SHM 2026
Project 2 data.  The analysis explains class imbalance, thin-structure and
boundary difficulty, component scale, and camera coverage without training a
model or changing any prediction.

## Frozen inputs

- dataset manifest: `data/manifests/contest_dataset_v1.json`;
- native PINHOLE labels and label manifest from `SEM-370-ALL300`;
- official-camera-compatible 400-view COLMAP poses from the frozen all-data
  RGB120 scene;
- `spatial_confirmation_v1` only for reporting the existing development-role
  membership; it does not select statistics or models.

Every label is verified against the frozen label manifest before aggregation.
Only official class IDs 0--4 are accepted, and Class ID 2 is named
`stay_cable` throughout.

## Fixed measurements

For each class, report total pixel share, number of views containing the class,
per-view pixel-fraction quantiles, LabelMe shape count, 8-connected component
counts and area quantiles, component-size bins, and the fraction of class
pixels within one/two/four pixels of an interior boundary.  For cameras,
report the 400-view and 300-labeled-view PCA extents and the nearest labeled
camera distance for each unlabeled view.

The component bins are fixed at `1--64`, `65--256`, `257--1024`, and `>1024`
pixels.  Boundary/thinness measurements use Euclidean distance transforms on
the native 1320x989 masks.  All outputs are descriptive statistics, not model
quality scores and not evidence of hidden-test performance.

## Outputs

The run writes a machine-readable result, per-view and per-component CSVs, and
paper-ready PNG/PDF figures under `outputs/quality/QUAL-009/r1`.  The tracked
report summarizes the findings and preserves hashes of the configuration,
inputs, script, and result.

