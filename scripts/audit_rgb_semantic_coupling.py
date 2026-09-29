#!/usr/bin/env python3
"""Audit association between frozen RGB reconstruction error and semantic errors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rgb_semantic_coupling import (  # noqa: E402
    confusion_matrix,
    pearson_correlation,
    reference_edge_strength,
    semantic_metrics,
    spearman_correlation,
)


RGB_FIELDS = ("PSNR_dB", "SSIM", "LPIPS_Alex")
SEMANTIC_FIELDS = ("mIoU_with_background", "mIoU_without_background", "stay_cable_IoU")
PIXEL_STRATA = (
    "all_correct",
    "all_wrong",
    "background_correct",
    "background_to_stay_cable",
    "stay_cable_true_positive",
    "stay_cable_to_background",
    "stay_cable_to_other",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-metrics", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument(
        "--semantic",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="semantic prediction name and PNG directory; repeat for multiple models",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"semantic image must be single-channel: {path}")
    return array


def parse_semantic_inputs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"semantic input must be NAME=DIR: {value}")
        name, directory = value.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"semantic input name is empty or duplicated: {name}")
        parsed[name] = Path(directory)
    return parsed


def load_rgb_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(field not in rows[0] for field in ("view", *RGB_FIELDS)):
        raise ValueError("RGB metrics CSV is empty or has an incompatible schema")
    return [
        {"view": row["view"], **{field: float(row[field]) for field in RGB_FIELDS}}
        for row in rows
    ]


def new_stratum() -> dict[str, float | int]:
    return {"pixel_count": 0, "rgb_absolute_error_sum": 0.0, "reference_edge_sum": 0.0}


def add_stratum(
    state: dict[str, float | int], mask: np.ndarray, rgb_mae: np.ndarray, edge: np.ndarray
) -> None:
    count = int(mask.sum())
    state["pixel_count"] = int(state["pixel_count"]) + count
    state["rgb_absolute_error_sum"] = float(state["rgb_absolute_error_sum"]) + float(
        rgb_mae[mask].sum(dtype=np.float64)
    )
    state["reference_edge_sum"] = float(state["reference_edge_sum"]) + float(
        edge[mask].sum(dtype=np.float64)
    )


def finish_stratum(state: dict[str, float | int]) -> dict[str, float | int | None]:
    count = int(state["pixel_count"])
    return {
        "pixel_count": count,
        "mean_RGB_MAE": float(state["rgb_absolute_error_sum"]) / count if count else None,
        "mean_reference_edge_strength": float(state["reference_edge_sum"]) / count
        if count
        else None,
    }


def masks(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_correct": target == prediction,
        "all_wrong": target != prediction,
        "background_correct": (target == 0) & (prediction == 0),
        "background_to_stay_cable": (target == 0) & (prediction == 2),
        "stay_cable_true_positive": (target == 2) & (prediction == 2),
        "stay_cable_to_background": (target == 2) & (prediction == 0),
        "stay_cable_to_other": (target == 2) & (prediction != 0) & (prediction != 2),
    }


def main() -> None:
    args = parse_args()
    semantic_dirs = parse_semantic_inputs(args.semantic)
    rows = load_rgb_metrics(args.rgb_metrics)
    view_names = [row["view"] for row in rows]
    if len(set(view_names)) != len(view_names):
        raise ValueError("RGB metrics contain duplicated view names")

    pixel_states = {
        name: {stratum: new_stratum() for stratum in PIXEL_STRATA}
        for name in semantic_dirs
    }
    semantic_per_view: dict[str, list[dict[str, Any]]] = {
        name: [] for name in semantic_dirs
    }

    for row in rows:
        view = row["view"]
        reference = load_rgb(args.reference_dir / view)
        render = load_rgb(args.render_dir / view)
        target = load_label(args.labels_dir / view)
        if reference.shape != render.shape or target.shape != reference.shape[:2]:
            raise ValueError(f"RGB/semantic shape mismatch for {view}")
        rgb_mae = np.abs(reference - render).mean(axis=2)
        edge = reference_edge_strength(reference)

        for name, prediction_dir in semantic_dirs.items():
            prediction = load_label(prediction_dir / view)
            if prediction.shape != target.shape:
                raise ValueError(f"semantic prediction shape mismatch for {name}/{view}")
            local = semantic_metrics(confusion_matrix(target, prediction))
            semantic_per_view[name].append({"view": view, **local})
            for stratum, mask in masks(target, prediction).items():
                add_stratum(pixel_states[name][stratum], mask, rgb_mae, edge)

    correlations: dict[str, dict[str, dict[str, float | None]]] = {}
    for name, semantic_rows in semantic_per_view.items():
        correlations[name] = {}
        for rgb_field in RGB_FIELDS:
            correlations[name][rgb_field] = {}
            x = [float(row[rgb_field]) for row in rows]
            for semantic_field in SEMANTIC_FIELDS:
                y = [float(row[semantic_field]) for row in semantic_rows]
                correlations[name][rgb_field][semantic_field] = {
                    "pearson": pearson_correlation(x, y),
                    "spearman": spearman_correlation(x, y),
                }

    result = {
        "schema_version": 1,
        "experiment_id": "RGB-071-SEMANTIC-COUPLING-DIAG",
        "status": "completed_read_only_diagnostic",
        "view_count": len(rows),
        "scope": "reused 74-view development split; association diagnostic, not causal proof",
        "rgb_error": "per-pixel mean absolute RGB difference in [0,1]",
        "reference_edge": "mean absolute forward luminance difference in x/y; Rec.709 weights",
        "view_level_correlations": correlations,
        "pixel_strata": {
            name: {
                stratum: finish_stratum(pixel_states[name][stratum])
                for stratum in PIXEL_STRATA
            }
            for name in semantic_dirs
        },
        "per_view": semantic_per_view,
        "inputs": {
            "rgb_metrics": str(args.rgb_metrics),
            "reference_dir": str(args.reference_dir),
            "render_dir": str(args.render_dir),
            "labels_dir": str(args.labels_dir),
            "semantic_prediction_dirs": {name: str(path) for name, path in semantic_dirs.items()},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "result.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "view_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

