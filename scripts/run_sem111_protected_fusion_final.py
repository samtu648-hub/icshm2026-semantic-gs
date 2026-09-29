#!/usr/bin/env python3
"""Evaluate the fixed SEM-110 protected fusion on the 74 registered views."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sem080_structural_diag as common

from src.semantic_protected_fusion import (
    fuse_background_stay_cable,
    local_confusion,
    merge_confusions,
    metrics_from_confusion,
)
from src.semantic_render_domain import semantic_rgb
from src.semantic_rendered_final import final_retention_gate, require_exact_view_set

CLASS_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--sem110-result", type=Path, required=True)
    parser.add_argument("--sem001-result", type=Path, required=True)
    parser.add_argument("--sem001-predictions", type=Path, required=True)
    parser.add_argument("--sem102-manifest", type=Path, required=True)
    parser.add_argument("--sem102-audit-root", type=Path, required=True)
    parser.add_argument("--sem001-external-root", type=Path, required=True)
    parser.add_argument("--sem102-external-root", type=Path, required=True)
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
        raise ValueError(
            f"{context} must be a 1320x989 official five-class PNG: {path}"
        )
    return value


def teacher_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        require(
            archive.files == ["probabilities"],
            "SEM-111 validation teacher NPZ has an unexpected schema",
        )
        return archive["probabilities"]


def validate_config(config: dict[str, Any]) -> None:
    require(
        config.get("experiment_id") == "SEM-111-PROTECTED-FUSION-FINAL",
        "SEM-111 received another config",
    )
    require(
        config.get("status") == "registered_not_run",
        "SEM-111 may run only from its registered configuration",
    )


def panel(target: np.ndarray, base: np.ndarray, fused: np.ndarray) -> np.ndarray:
    error = np.zeros((*target.shape, 3), dtype=np.uint8)
    error[target != fused] = np.asarray([255, 48, 48], dtype=np.uint8)
    return np.concatenate(
        [semantic_rgb(target), semantic_rgb(base), semantic_rgb(fused), error], axis=1
    )


def one_hot_probabilities(
    prediction: np.ndarray, *, epsilon: float = 1e-6
) -> np.ndarray:
    output = np.full((CLASS_COUNT, *prediction.shape), epsilon, dtype=np.float32)
    for class_id in range(CLASS_COUNT):
        output[class_id, prediction == class_id] = 1.0 - epsilon * (CLASS_COUNT - 1)
    if not np.allclose(output.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("SEM-111 one-hot output is not a simplex")
    return output


def external_output(
    *,
    base_root: Path,
    teacher_root: Path,
    output_root: Path,
    threshold: float,
) -> dict[str, Any]:
    base = class_map(
        base_root / "semantic.png", context="SEM-111 external SEM-001 output"
    )
    teacher_logits = np.load(teacher_root / "semantic_logits.npy").astype(np.float32)
    require(
        teacher_logits.shape == (CLASS_COUNT, 989, 1320),
        "SEM-111 external teacher logits have a wrong shape",
    )
    normalised = teacher_logits - teacher_logits.max(axis=0, keepdims=True)
    teacher_prob = np.exp(normalised)
    teacher_prob /= teacher_prob.sum(axis=0, keepdims=True)
    fused, fixed = fuse_background_stay_cable(
        base, teacher_prob, stay_cable_threshold=threshold
    )
    require(
        np.array_equal(base[fixed], fused[fixed]),
        "SEM-111 external fusion modified protected classes",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rgb_source = teacher_root / "rgb.png"
    rgb_output = output_root / "rgb.png"
    shutil.copyfile(rgb_source, rgb_output)
    probabilities = one_hot_probabilities(fused)
    logits = np.log(np.clip(probabilities, np.finfo(np.float32).tiny, 1.0)).astype(
        np.float32
    )
    np.save(output_root / "semantic_logits.npy", logits)
    Image.fromarray(fused).save(output_root / "semantic.png")
    copied_rgb = np.asarray(Image.open(rgb_output), dtype=np.uint8)
    require(
        copied_rgb.shape == (989, 1320, 3),
        "SEM-111 external RGB is not on the official grid",
    )
    return {
        "input_pose_convention": "CAMERA_FROM_WORLD",
        "input_camera_model": "SIMPLE_RADIAL",
        "output_grid": "official_SIMPLE_RADIAL",
        "rgb_shape": list(copied_rgb.shape),
        "rgb_sha256_source": sha256(rgb_source),
        "rgb_sha256_output": sha256(rgb_output),
        "rgb_bytes_identical": rgb_source.read_bytes() == rgb_output.read_bytes(),
        "semantic_logits_shape": list(logits.shape),
        "semantic_logits_dtype": str(logits.dtype),
        "semantic_png_shape": list(fused.shape),
        "semantic_png_dtype": str(fused.dtype),
        "semantic_png_min": int(fused.min()),
        "semantic_png_max": int(fused.max()),
        "probability_simplex": bool(
            np.allclose(probabilities.sum(axis=0), 1.0, rtol=0.0, atol=1e-6)
        ),
        "protected_prediction_pixels_exact": bool(
            np.array_equal(base[fixed], fused[fixed])
        ),
        "uses_reference_rgb_or_labels": False,
    }


def main() -> int:
    args = parse_args()
    config = common.load_json(args.config)
    split = common.load_json(args.split)
    validate_config(config)
    require(args.labels_root.is_dir(), "SEM-111 fixed validation labels are missing")
    require(
        not args.output_dir.exists() or not any(args.output_dir.iterdir()),
        "SEM-111 refuses to overwrite output",
    )
    names = require_exact_view_set(
        split["semantic_validation_labeled"],
        split["semantic_validation_labeled"],
        context="SEM-111 validation plan",
    )
    require(
        len(names) == int(config["data"]["validation_images"]),
        "SEM-111 has an unexpected validation count",
    )
    sem110 = common.load_json(args.sem110_result)
    require(
        sha256(args.sem110_result) == config["inputs"]["sem110_result_sha256"],
        "SEM-111 SEM-110 result hash differs from config",
    )
    require(
        sem110.get("status") == "completed_oof_gate_passed",
        "SEM-111 requires SEM-110's passing OOF gate",
    )
    require(
        float(sem110.get("selected_threshold"))
        == float(config["fusion"]["stay_cable_threshold"]),
        "SEM-111 threshold differs from SEM-110 selection",
    )
    require(
        sha256(args.sem001_result) == config["inputs"]["sem001_result_sha256"],
        "SEM-111 SEM-001 result hash differs from config",
    )
    sem001_result = common.load_json(args.sem001_result)
    require(
        sem001_result.get("checkpoint_reloaded") is True,
        "SEM-111 parent SEM-001 checkpoint reload failed",
    )
    require(
        sha256(args.sem102_manifest)
        == config["inputs"]["sem102_prediction_manifest_sha256"],
        "SEM-111 SEM-102 manifest hash differs from config",
    )
    require(
        sha256(args.sem102_external_root / "evidence.json")
        == config["inputs"]["sem102_external_evidence_sha256"],
        "SEM-111 SEM-102 external evidence hash differs from config",
    )
    teacher_manifest = common.load_json(args.sem102_manifest)
    require(
        [record.get("name") for record in teacher_manifest] == names,
        "SEM-111 SEM-102 predictions do not exactly match validation views",
    )
    teacher_paths: dict[str, Path] = {}
    for record in teacher_manifest:
        probability = args.sem102_audit_root / str(record["probabilities"])
        require(
            probability.is_file()
            and sha256(probability) == record.get("probabilities_sha256"),
            f"SEM-111 teacher hash mismatch: {record.get('name')}",
        )
        teacher_paths[str(record["name"])] = probability

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, str]] = []
    protected_exact = True
    fused_confusions: list[np.ndarray] = []
    per_view: list[dict[str, Any]] = []
    prediction_root = args.output_dir / "predictions"
    threshold = float(config["fusion"]["stay_cable_threshold"])
    # Every prediction is materialised before a label image is opened.
    for name in names:
        base_path = args.sem001_predictions / name
        base = class_map(base_path, context="SEM-111 SEM-001 validation output")
        fused, fixed = fuse_background_stay_cable(
            base,
            teacher_probabilities(teacher_paths[name]),
            stay_cable_threshold=threshold,
        )
        protected_exact = protected_exact and bool(
            np.array_equal(base[fixed], fused[fixed])
        )
        prediction_path = prediction_root / name
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fused).save(prediction_path)
        sources.append(
            {
                "name": name,
                "sem001_prediction_sha256": sha256(base_path),
                "sem102_probability_sha256": sha256(teacher_paths[name]),
                "fused_prediction": str(prediction_path.relative_to(args.output_dir)),
                "fused_prediction_sha256": sha256(prediction_path),
            }
        )
    common.write_json(
        args.output_dir / "prediction_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "status": "completed_predictions_before_validation_labels",
            "stay_cable_threshold": threshold,
            "predictions": sources,
            "protected_prediction_pixels_exact": protected_exact,
            "validation_access_before_prediction_manifest": False,
            "real_rgb_input_opened": False,
            "unlabeled_images_opened": False,
        },
    )

    panels = args.output_dir / "validation_panels"
    for name in names:
        target = class_map(args.labels_root / name, context="SEM-111 validation label")
        base = class_map(
            args.sem001_predictions / name, context="SEM-111 SEM-001 validation output"
        )
        fused = class_map(
            prediction_root / name, context="SEM-111 fused validation output"
        )
        confusion = local_confusion(target, fused)
        fused_confusions.append(confusion)
        local = metrics_from_confusion(confusion)
        per_view.append({"view": name, **local})
        if name in config["outputs"]["validation_visual_samples"]:
            panels.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel(target, base, fused)).save(panels / name)
    metrics = metrics_from_confusion(merge_confusions(fused_confusions))
    worst_count = max(1, int(np.ceil(len(per_view) * 0.10)))
    worst = sorted(per_view, key=lambda record: record["mIoU_without_background"])[
        :worst_count
    ]
    difficult = [
        record for record in per_view if 140 <= int(Path(record["view"]).stem) <= 146
    ]
    metrics.update(
        {
            "view_count": len(names),
            "per_view": per_view,
            "worst_10_percent": {
                "count": worst_count,
                "views": [record["view"] for record in worst],
                "mean_mIoU_without_background": float(
                    np.mean([record["mIoU_without_background"] for record in worst])
                ),
            },
            "challenge_140_146": {
                "views": [record["view"] for record in difficult],
                "mean_mIoU_without_background": float(
                    np.mean([record["mIoU_without_background"] for record in difficult])
                ),
            },
        }
    )
    external = external_output(
        base_root=args.sem001_external_root,
        teacher_root=args.sem102_external_root,
        output_root=args.output_dir / "external_camera",
        threshold=threshold,
    )
    gate = final_retention_gate(
        metrics, config["baseline_sem001"], config["formal_retention_gates"]
    )
    engineering = {
        "sem001_checkpoint_reload": sem001_result.get("checkpoint_reloaded") is True,
        "protected_prediction_pixels_exact": protected_exact,
        "rgb_bytes_identical": external["rgb_bytes_identical"],
        "official_external_camera": external["probability_simplex"]
        and external["uses_reference_rgb_or_labels"] is False,
    }
    replacement = bool(gate["passed"] and all(engineering.values()))
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_formal_retention_gate_passed"
        if replacement
        else "completed_formal_retention_gate_failed",
        "stay_cable_threshold": threshold,
        "validation_prediction_manifest_sha256": sha256(
            args.output_dir / "prediction_manifest.json"
        ),
        "validation_labels_opened_after_all_predictions": True,
        "metrics": metrics,
        "metric_retention_gate": gate,
        "engineering_checks": engineering,
        "formal_replacement_allowed": replacement,
        "external_camera": external,
        "panels": [
            str((panels / name).resolve())
            for name in config["outputs"]["validation_visual_samples"]
        ],
        "fallback": config["fallback"],
    }
    common.write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {"status": result["status"], "formal_replacement_allowed": replacement},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
