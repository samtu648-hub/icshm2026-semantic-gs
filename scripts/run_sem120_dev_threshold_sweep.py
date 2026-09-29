#!/usr/bin/env python3
"""Run a labelled 74-view developer sweep of protected-fusion thresholds.

This is intentionally a development tool, not a blind evaluation or formal
replacement gate.  It keeps RGB and SEM-001 protected classes frozen and scans
only the existing two-stream background/stay_cable threshold on the already
opened 74-view development set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as common

from src.semantic_development import development_score, select_development_candidate
from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import require_exact_view_set

CLASS_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-predictions", type=Path, required=True)
    parser.add_argument("--sem102-manifest", type=Path, required=True)
    parser.add_argument("--sem102-audit-root", type=Path, required=True)
    parser.add_argument("--sem111-result", type=Path, required=True)
    parser.add_argument("--sem111-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or int(value.max()) >= CLASS_COUNT:
        raise ValueError(f"{context} must be an official 1320x989 five-class PNG: {path}")
    return value


def probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["probabilities"],
            "SEM-120 teacher NPZ has an unexpected schema",
        )
        value = archive["probabilities"]
    if value.shape != (CLASS_COUNT, 989, 1320):
        raise ValueError("SEM-120 teacher probability shape is invalid")
    return value


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("experiment_id") == "SEM-120-DEV-THRESHOLD-SWEEP",
        "SEM-120 received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-120 may run only from its registered config",
    )


def validate_sources(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    names: list[str],
) -> dict[str, Path]:
    require(
        sha256(args.sem001_result) == config["inputs"]["sem001_result_sha256"],
        "SEM-120 SEM-001 result hash differs from its registered source",
    )
    require(
        common.load_json(args.sem001_result).get("checkpoint_reloaded") is True,
        "SEM-120 requires a reloaded SEM-001 checkpoint",
    )
    require(
        sha256(args.sem111_result) == config["inputs"]["sem111_result_sha256"],
        "SEM-120 SEM-111 result hash differs from its registered source",
    )
    sem111 = common.load_json(args.sem111_result)
    require(
        sem111.get("status") == "completed_formal_retention_gate_passed",
        "SEM-120 requires SEM-111's completed baseline",
    )
    require(
        sha256(args.sem102_manifest)
        == config["inputs"]["sem102_prediction_manifest_sha256"],
        "SEM-120 SEM-102 manifest hash differs from its registered source",
    )
    manifest = common.load_json(args.sem102_manifest)
    require(
        [record.get("name") for record in manifest] == names,
        "SEM-120 teacher does not exactly cover the 74 development views",
    )
    paths: dict[str, Path] = {}
    for record in manifest:
        path = args.sem102_audit_root / str(record["probabilities"])
        require(
            path.is_file() and sha256(path) == record.get("probabilities_sha256"),
            f"SEM-120 teacher hash mismatch: {record.get('name')}",
        )
        paths[str(record["name"])] = path
    return paths


def panel(target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[target != candidate] = np.asarray([255, 48, 48], dtype=np.uint8)
    return np.concatenate(
        [semantic_rgb(target), semantic_rgb(baseline), semantic_rgb(candidate), error],
        axis=1,
    )


def summarise_threshold(
    *,
    threshold: float,
    names: list[str],
    labels_root: Path,
    sem001_predictions: Path,
    teacher_paths: dict[str, Path],
    sem111_predictions: Path,
    baseline_threshold: float,
    write_predictions: Path | None = None,
) -> dict[str, Any]:
    confusions: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    protected_exact = True
    baseline_exact = True
    for name in names:
        target = class_map(labels_root / name, context="SEM-120 development label")
        base = class_map(sem001_predictions / name, context="SEM-120 SEM-001")
        teacher = probabilities(teacher_paths[name])
        candidate, fixed = fuse_background_stay_cable(
            base, teacher, stay_cable_threshold=threshold
        )
        baseline, _baseline_fixed = fuse_background_stay_cable(
            base, teacher, stay_cable_threshold=baseline_threshold
        )
        stored_baseline = class_map(sem111_predictions / name, context="SEM-120 SEM-111")
        protected_exact = protected_exact and bool(
            np.array_equal(candidate[fixed], base[fixed])
        )
        baseline_exact = baseline_exact and bool(np.array_equal(baseline, stored_baseline))
        confusion = local_confusion(target, candidate)
        confusions.append(confusion)
        per_view.append({"view": name, **metrics_from_confusion(confusion)})
        if write_predictions is not None:
            write_predictions.mkdir(parents=True, exist_ok=True)
            Image.fromarray(candidate).save(write_predictions / name)
    metrics = metrics_from_confusion(merge_confusions(confusions))
    worst_count = max(1, int(np.ceil(0.10 * len(per_view))))
    worst = sorted(per_view, key=lambda row: row["mIoU_without_background"])[:worst_count]
    challenge = [
        row for row in per_view if 140 <= int(Path(row["view"]).stem) <= 146
    ]
    metrics.update(
        {
            "view_count": len(names),
            "worst_10_percent_mIoU_without_background": float(
                np.mean([row["mIoU_without_background"] for row in worst])
            ),
            "worst_10_percent_views": [row["view"] for row in worst],
            "challenge_140_146_mIoU_without_background": float(
                np.mean([row["mIoU_without_background"] for row in challenge])
            ),
            "per_view": per_view,
        }
    )
    return {
        "stay_cable_threshold": threshold,
        "metrics": metrics,
        "developer_score": development_score(metrics),
        "protected_prediction_pixels_exact": protected_exact,
        "sem111_baseline_reconstruction_exact": baseline_exact,
    }


def write_leaderboard(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "stay_cable_threshold",
        "developer_score",
        "mIoU_with_background",
        "mIoU_without_background",
        "stay_cable_iou",
        "stay_cable_precision",
        "stay_cable_recall",
        "background_to_stay_cable_pixels",
        "background_to_stay_cable_fraction",
        "worst_10_percent_mIoU_without_background",
        "challenge_140_146_mIoU_without_background",
        "protected_prediction_pixels_exact",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            cable = metrics["per_class"]["stay_cable"]
            writer.writerow(
                {
                    "stay_cable_threshold": record["stay_cable_threshold"],
                    "developer_score": record["developer_score"],
                    "mIoU_with_background": metrics["mIoU_with_background"],
                    "mIoU_without_background": metrics["mIoU_without_background"],
                    "stay_cable_iou": cable["iou"],
                    "stay_cable_precision": cable["precision"],
                    "stay_cable_recall": cable["recall"],
                    "background_to_stay_cable_pixels": metrics["background_to_stay_cable_pixels"],
                    "background_to_stay_cable_fraction": metrics["background_to_stay_cable_fraction"],
                    "worst_10_percent_mIoU_without_background": metrics[
                        "worst_10_percent_mIoU_without_background"
                    ],
                    "challenge_140_146_mIoU_without_background": metrics[
                        "challenge_140_146_mIoU_without_background"
                    ],
                    "protected_prediction_pixels_exact": record[
                        "protected_prediction_pixels_exact"
                    ],
                }
            )


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-120 development labels are missing")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-120 refuses to overwrite developer output",
    )
    names = require_exact_view_set(
        split["semantic_validation_labeled"],
        split["semantic_validation_labeled"],
        context="SEM-120 developer view plan",
    )
    require(len(names) == int(config["data"]["development_images"]), "SEM-120 expects 74 views")
    teacher_paths = validate_sources(args=args, config=config, names=names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        summarise_threshold(
            threshold=float(threshold),
            names=names,
            labels_root=args.labels_root,
            sem001_predictions=args.sem001_predictions,
            teacher_paths=teacher_paths,
            sem111_predictions=args.sem111_predictions,
            baseline_threshold=float(config["baseline"]["sem111_threshold"]),
        )
        for threshold in config["sweep"]["stay_cable_thresholds"]
    ]
    selected = select_development_candidate(records)
    selected_threshold = float(selected["stay_cable_threshold"])
    selected_root = args.output_dir / "selected_predictions"
    materialised = summarise_threshold(
        threshold=selected_threshold,
        names=names,
        labels_root=args.labels_root,
        sem001_predictions=args.sem001_predictions,
        teacher_paths=teacher_paths,
        sem111_predictions=args.sem111_predictions,
        baseline_threshold=float(config["baseline"]["sem111_threshold"]),
        write_predictions=selected_root,
    )
    require(materialised == selected, "SEM-120 selected prediction materialisation drifted")
    panel_root = args.output_dir / "development_panels"
    for name in config["outputs"]["visual_samples"]:
        target = class_map(args.labels_root / name, context="SEM-120 development label")
        baseline = class_map(args.sem111_predictions / name, context="SEM-120 SEM-111")
        candidate = class_map(selected_root / name, context="SEM-120 selected output")
        panel_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(panel(target, baseline, candidate)).save(panel_root / name)
    write_leaderboard(args.output_dir / "leaderboard.csv", records)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_development_sweep",
        "development_data_notice": "All 74 views are an explicitly reused development set. Scores select a developer leader only and are not blind-test or formal-replacement evidence.",
        "threshold_records": records,
        "selected_development_threshold": selected_threshold,
        "selected_development_record": selected,
        "sem111_threshold": config["baseline"]["sem111_threshold"],
        "fixed_validation_views_opened": True,
        "real_rgb_input_opened": False,
        "unlabeled_images_opened": False,
        "external_camera_mode_present": False,
        "rgb_modified": False,
        "outputs": {
            "leaderboard_csv": "leaderboard.csv",
            "selected_predictions": "selected_predictions",
            "panel_order": ["GT", "SEM-111", "selected", "selected_error"],
            "development_panels": [
                str((panel_root / name).resolve())
                for name in config["outputs"]["visual_samples"]
            ],
        },
    }
    common.write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_development_threshold": selected_threshold,
                "developer_score": selected["developer_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
