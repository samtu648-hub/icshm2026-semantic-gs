#!/usr/bin/env python3
"""Select and independently check SEM-160 deck recovery on training OOF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic2d import read_json, sha256, write_json
from src.semantic_deck_binary import (
    deck_gate,
    recover_deck,
    select_calibration_threshold,
)
from src.semantic_protected_fusion import (
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
    partition_training_names,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_training import select_rendered_training_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--oof-split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem001-training-root", type=Path, required=True)
    parser.add_argument("--sem150-root", type=Path, required=True)
    parser.add_argument("--sem150-result", type=Path, required=True)
    parser.add_argument("--sem160-oof-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def class_map(path: Path, *, context: str) -> np.ndarray:
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.shape != (989, 1320) or value.size == 0 or int(value.max()) >= 5:
        raise ValueError(f"{context} is not a 1320x989 five-class PNG: {path}")
    return value


def deck_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["deck_probability"],
            "SEM-160 deck probability NPZ has an unexpected schema",
        )
        return archive["deck_probability"]


def sem001_paths(
    root: Path, names: list[str], *, manifest_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "training_sem001_render_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        "SEM-160 SEM-001 training manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    require(
        manifest.get("status") == "completed_training_sem001_render"
        and [record["name"] for record in manifest["rendered"]] == names,
        "SEM-160 SEM-001 training predictions are incomplete",
    )
    output: dict[str, Path] = {}
    for record in manifest["rendered"]:
        path = root / record["semantic_png"]
        require(
            path.is_file() and sha256(path) == record["sha256"],
            f"SEM-160 SEM-001 prediction hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def base_paths(
    root: Path, names: list[str], *, manifest_sha256: str
) -> dict[str, Path]:
    manifest_path = root / "prediction_manifest.json"
    require(
        manifest_path.is_file() and sha256(manifest_path) == manifest_sha256,
        "SEM-160 SEM-150 prediction manifest hash mismatch",
    )
    manifest = read_json(manifest_path)
    records = manifest["predictions"]
    require(
        manifest.get("experiment_id") == "SEM-150-FOUNDATION-GATE-OOF"
        and manifest.get("fixed_validation_opened") is False
        and [record["name"] for record in records] == names,
        "SEM-160 SEM-150 selected prediction order mismatch",
    )
    output: dict[str, Path] = {}
    for record in records:
        path = root / record["prediction"]
        require(
            path.is_file() and sha256(path) == record["prediction_sha256"],
            f"SEM-160 SEM-150 prediction hash mismatch: {record['name']}",
        )
        output[record["name"]] = path
    return output


def oof_probability_paths(
    root: Path, oof_split: dict[str, Any], names: list[str]
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for fold in oof_split["folds"]:
        fold_index = int(fold["fold"])
        fold_root = root / f"fold-{fold_index}"
        result = read_json(fold_root / "result.json")
        require(
            result.get("experiment_id") == "SEM-160-DECK-BINARY-OOF"
            and result.get("status") == "completed_fold_oof",
            f"SEM-160 fold {fold_index} is incomplete",
        )
        require(
            result.get("fixed_validation_opened") is False
            and result.get("unlabeled_rgb_opened") is False,
            f"SEM-160 fold {fold_index} violated its training-only boundary",
        )
        records = read_json(fold_root / "prediction_manifest.json")
        require(
            [record["name"] for record in records] == list(fold["oof"]),
            f"SEM-160 fold {fold_index} prediction order mismatch",
        )
        for record in records:
            path = fold_root / record["deck_probability"]
            require(
                path.is_file()
                and sha256(path) == record["deck_probability_sha256"],
                f"SEM-160 OOF probability hash mismatch: {record['name']}",
            )
            output[record["name"]] = path
    require(set(output) == set(names), "SEM-160 OOF does not cover all training views")
    return output


def pre_foundation_map(base: np.ndarray, sem001: np.ndarray) -> np.ndarray:
    """Restore only the provenance of class-4 pixels rejected by SEM-150."""
    restored = base.copy()
    restored[(base == 0) & (sem001 == 4)] = 4
    return restored


def evaluate(
    *,
    names: list[str],
    threshold: float,
    labels_root: Path,
    base: dict[str, Path],
    sem001: dict[str, Path],
    probability: dict[str, Path],
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    confusions: list[np.ndarray] = []
    protected_exact = True
    additions_by_gt = np.zeros(5, dtype=np.int64)
    records: list[dict[str, Any]] = []
    for name in names:
        target = class_map(labels_root / name, context="SEM-160 training label")
        baseline = class_map(base[name], context="SEM-160 SEM-150 baseline")
        sem001_map = class_map(sem001[name], context="SEM-160 SEM-001 provenance")
        recovered, eligible, additions = recover_deck(
            baseline,
            pre_foundation_map(baseline, sem001_map),
            deck_probability(probability[name]),
            threshold=threshold,
        )
        protected_exact = protected_exact and bool(
            np.array_equal(recovered[~eligible], baseline[~eligible])
        )
        additions_by_gt += np.bincount(target[additions], minlength=5).astype(
            np.int64
        )
        confusions.append(local_confusion(target, recovered))
        if prediction_root is not None:
            prediction_root.mkdir(parents=True, exist_ok=True)
            path = prediction_root / name
            Image.fromarray(recovered).save(path)
            records.append(
                {
                    "name": name,
                    "prediction": str(path.relative_to(prediction_root.parent)),
                    "prediction_sha256": sha256(path),
                    "added_deck_pixels": int(np.count_nonzero(additions)),
                }
            )
    return {
        "metrics": metrics_from_confusion(merge_confusions(confusions)),
        "protected_predictions_exact": protected_exact,
        "additions_by_gt_class": additions_by_gt.astype(int).tolist(),
        "added_deck_pixels": int(additions_by_gt.sum()),
        "prediction_records": records,
    }


def gate_for(
    evaluation: dict[str, Any], baseline: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    return deck_gate(
        candidate=evaluation["metrics"],
        baseline=baseline,
        protected_predictions_exact=evaluation["protected_predictions_exact"],
        minimum_iou_gain=float(gate["minimum_deck_iou_gain"]),
        minimum_recall_gain=float(gate["minimum_deck_recall_gain"]),
        minimum_precision=float(gate["minimum_deck_precision"]),
        minimum_miou_gain=float(gate["minimum_mIoU_gain"]),
        maximum_background_fp_ratio=float(
            gate["maximum_background_to_deck_fp_ratio"]
        ),
    )


def write_panels(
    *,
    names: list[str],
    labels_root: Path,
    base: dict[str, Path],
    prediction_root: Path,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = class_map(labels_root / name, context="SEM-160 panel target")
        before = class_map(base[name], context="SEM-160 panel baseline")
        after = class_map(prediction_root / name, context="SEM-160 panel candidate")
        outcome = np.zeros((*target.shape, 3), dtype=np.uint8)
        changed = before != after
        outcome[changed & (target == 1)] = np.asarray([48, 220, 96], dtype=np.uint8)
        outcome[changed & (target != 1)] = np.asarray([255, 48, 48], dtype=np.uint8)
        panel = np.concatenate(
            [semantic_rgb(target), semantic_rgb(before), semantic_rgb(after), outcome],
            axis=1,
        )
        Image.fromarray(panel).save(output_root / name)


def baseline_metrics(sem150: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold = float(sem150["selected_threshold"])
    calibration = [
        record
        for record in sem150["calibration_records"]
        if float(record["threshold"]) == threshold
    ]
    require(len(calibration) == 1, "SEM-160 could not resolve SEM-150 calibration")
    return calibration[0]["metrics"], sem150["check_record"]["metrics"]


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    require(
        config.get("experiment_id") == "SEM-160-DECK-BINARY-OOF",
        "SEM-160 summary received another experiment config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-160 summary requires its registered configuration",
    )
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-160 refuses to overwrite its OOF summary",
    )
    require(
        sha256(args.split) == config["data"]["semantic_split_sha256"]
        and sha256(args.oof_split) == config["data"]["oof_split_sha256"],
        "SEM-160 split hash mismatch",
    )
    require(
        sha256(args.sem150_result) == config["inputs"]["sem150_result_sha256"],
        "SEM-160 SEM-150 result hash mismatch",
    )
    sem150 = read_json(args.sem150_result)
    require(
        sem150.get("status") == "completed_oof_gate_passed",
        "SEM-160 requires the passing SEM-150 training baseline",
    )
    semantic_split = read_json(args.split)
    oof_split = read_json(args.oof_split)
    names = select_rendered_training_views(
        semantic_split, expected_count=int(config["data"]["training_images"])
    )
    partition = partition_training_names(names, expected_count=len(names))
    sem001 = sem001_paths(
        args.sem001_training_root,
        names,
        manifest_sha256=config["inputs"]["sem001_training_manifest_sha256"],
    )
    base = base_paths(
        args.sem150_root,
        names,
        manifest_sha256=config["inputs"]["sem150_prediction_manifest_sha256"],
    )
    probability = oof_probability_paths(args.sem160_oof_root, oof_split, names)
    baseline_calibration, baseline_check = baseline_metrics(sem150)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "input_access_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "training_views": names,
            "partition": partition,
            "sem150_result_sha256": sha256(args.sem150_result),
            "all_oof_inputs_validated_before_labels": True,
            "fixed_validation_opened": False,
            "unlabeled_images_opened": False,
            "real_rgb_input_opened": False,
        },
    )

    records: list[dict[str, Any]] = []
    for threshold in config["fusion"]["threshold_grid"]:
        evaluation = evaluate(
            names=partition["calibration"],
            threshold=float(threshold),
            labels_root=args.labels_root,
            base=base,
            sem001=sem001,
            probability=probability,
        )
        record = {
            "threshold": float(threshold),
            **evaluation,
            "gate": gate_for(
                evaluation,
                baseline_calibration,
                config["oof_gates"]["calibration"],
            ),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "phase": "calibration",
                    "threshold": threshold,
                    "deck_iou": evaluation["metrics"]["per_class"]["deck"]["iou"],
                    "gate_passed": record["gate"]["passed"],
                }
            ),
            flush=True,
        )
    selected = select_calibration_threshold(records)
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "threshold_grid": list(config["fusion"]["threshold_grid"]),
        "calibration_records": records,
        "baseline_calibration_metrics": baseline_calibration,
        "baseline_check_metrics": baseline_check,
        "input_access_manifest_sha256": sha256(
            args.output_dir / "input_access_manifest.json"
        ),
        "fixed_validation_opened": False,
        "unlabeled_images_opened": False,
        "real_rgb_input_opened": False,
        "rgb_modified": False,
    }
    if selected is None:
        result.update(
            {
                "status": "completed_calibration_gate_failed",
                "selected_threshold": None,
                "check_labels_opened": False,
                "next_step": "Stop before SEM-161; no calibration threshold safely improved deck.",
            }
        )
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"]}), flush=True)
        return 0

    threshold = float(selected["threshold"])
    prediction_root = args.output_dir / "selected_predictions"
    check = evaluate(
        names=partition["check"],
        threshold=threshold,
        labels_root=args.labels_root,
        base=base,
        sem001=sem001,
        probability=probability,
        prediction_root=prediction_root,
    )
    check_gate = gate_for(check, baseline_check, config["oof_gates"]["check"])
    calibration_materialised = evaluate(
        names=partition["calibration"],
        threshold=threshold,
        labels_root=args.labels_root,
        base=base,
        sem001=sem001,
        probability=probability,
        prediction_root=prediction_root,
    )
    prediction_records = sorted(
        check["prediction_records"] + calibration_materialised["prediction_records"],
        key=lambda record: names.index(record["name"]),
    )
    write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_selected_predictions",
            "selected_threshold": threshold,
            "predictions": prediction_records,
            "fixed_validation_opened": False,
        },
    )
    write_panels(
        names=list(config["outputs"]["training_visual_samples"]),
        labels_root=args.labels_root,
        base=base,
        prediction_root=prediction_root,
        output_root=args.output_dir / "training_panels",
    )
    result.update(
        {
            "status": "completed_oof_gate_passed"
            if check_gate["passed"]
            else "completed_oof_gate_failed",
            "selected_threshold": threshold,
            "selected_calibration_record": selected,
            "check_labels_opened": True,
            "check_record": {**check, "gate": check_gate},
            "prediction_manifest_sha256": sha256(
                args.output_dir / "prediction_manifest.json"
            ),
            "next_step": "A pass permits one separately registered SEM-161 final model; a failure stops deck recovery.",
        }
    )
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_threshold": threshold,
                "check_gate_passed": check_gate["passed"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
