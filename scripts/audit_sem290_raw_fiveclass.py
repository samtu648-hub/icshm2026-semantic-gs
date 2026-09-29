#!/usr/bin/env python3
"""Read-only raw five-class OOF audit for the sealed SEM-270 control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_sem260_error_budget import manifest_paths, png
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem270_foundation_binary_oof import initialize
from src.semantic2d import confusion_metrics, read_json, sha256, write_json
from src.semantic_protected_fusion import local_confusion, partition_training_names
from src.submission_sem161 import sliding

EXP = "SEM-290-NATIVE960-RAW-FIVECLASS-AUDIT"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_png(path: Path, image: np.ndarray) -> str:
    require(image.dtype == np.uint8 and image.ndim == 2, "Invalid raw class map")
    Image.fromarray(image).save(path)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/sem270/full-20260907-r1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/runs/SEM-290-NATIVE960-RAW-FIVECLASS-AUDIT/config.yaml",
    )
    args = parser.parse_args()
    require(args.chunk_count > 0 and 0 <= args.chunk_index < args.chunk_count, "Invalid chunk selection")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "raw_argmax"
    raw_dir.mkdir()

    cfg = read_json(args.config)
    require(cfg["experiment_id"] == EXP, "SEM-290 config mismatch")
    run = read_json(args.run_dir / "result.json")
    seal = read_json(args.run_dir / "prediction_seal.json")
    sealed = read_json(args.run_dir / "calibration_sealed.json")
    manifest = read_json(args.run_dir / "prediction_manifest.json")
    sem270_cfg = read_json(ROOT / "experiments/runs/SEM-270-FOUNDATION-BINARY-OOF/config.yaml")
    require(run["experiment_id"] == sem270_cfg["experiment_id"] == "SEM-270-FOUNDATION-BINARY-OOF", "Parent mismatch")
    require(sha256(args.run_dir / "prediction_seal.json") == run["prediction_seal_sha256"], "SEM-270 seal changed")
    require(sha256(args.run_dir / "calibration_sealed.json") == run["calibration_sha256"], "SEM-270 calibration changed")
    require(sha256(args.run_dir / "prediction_manifest.json") == seal["prediction_manifest_sha256"], "SEM-270 manifest changed")
    require(sha256(ROOT / cfg["semantic_split"]) == cfg["semantic_split_sha256"], "Semantic split changed")
    require(sha256(ROOT / cfg["oof_split"]) == cfg["oof_split_sha256"], "OOF split changed")

    split = read_json(ROOT / cfg["oof_split"])
    names = read_json(ROOT / cfg["semantic_split"])["semantic_train_labeled"]
    require(len(names) == 219 and len(set(names)) == 219, "Unexpected OOF universe")
    selected_names = sorted(names)[args.chunk_index :: args.chunk_count]
    require(selected_names, "Empty audit chunk")
    halves = partition_training_names(names, expected_count=219)
    calibration_names = set(halves["calibration"])
    check_names = set(halves["check"])
    require(len(calibration_names) == 108 and len(check_names) == 111, "Unexpected calibration/check partition")

    rgb_paths = manifest_paths(
        args.artifact_root / cfg["parents"]["rgb"],
        "rendered_dataset_manifest.json",
        "rendered_images",
        "path",
        "sha256",
    )
    labels_dir = args.artifact_root / cfg["parents"]["labels"]
    require(set(rgb_paths) == set(names), "Wrong RGB universe")
    require(set(manifest["five_control"]) == set(names), "Incomplete SEM-270 five-control OOF")
    freeze = verify_freeze(ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json", args.artifact_root)
    require(freeze == run["freeze_before"] == run["freeze_after"], "Frozen submission source changed")

    matrices = np.zeros((5, 5), dtype=np.int64)
    per_view = []
    output_manifest = {}
    target_access = []
    checkpoint_records = []
    torch.set_num_threads(4)
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    for fold in split["folds"]:
        fold_id = fold["fold"]
        held_out = sorted(set(fold["oof"]) & set(selected_names))
        if not held_out:
            continue
        checkpoint = args.run_dir / f"fold{fold_id}_five_control/model.pth"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        require(metadata["experiment_id"] == "SEM-270-FOUNDATION-BINARY-OOF", "Wrong checkpoint experiment")
        require(metadata["arm"] == "five_control" and metadata["fold"] == fold_id, "Wrong checkpoint arm/fold")
        require(set(metadata["oof"]) == set(fold["oof"]), "Checkpoint OOF membership changed")
        require(sha256(checkpoint) == run["folds"][fold_id]["arms"]["five_control"]["training"]["sha256"], "Checkpoint changed")
        model = initialize(args.artifact_root / cfg["parents"]["model_config"], "five_control", cfg["seed"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.cuda().eval().requires_grad_(False)
        checkpoint_records.append({"fold": fold_id, "checkpoint": str(checkpoint), "sha256": sha256(checkpoint), "reload": True})
        for name in held_out:
            rgb_path, rgb_digest = rgb_paths[name]
            require(sha256(rgb_path) == rgb_digest, f"Rendered RGB changed: {name}")
            rgb = png(rgb_path)
            require(rgb.shape == (989, 1320, 3), f"Unexpected RGB grid: {name}")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = sliding(model, rgb, crop=cfg["inference"]["crop"], stride=cfg["inference"]["stride"], classes=5)
            require(probabilities.shape == (5, 989, 1320), f"Unexpected probability shape: {name}")
            require(np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)), f"Invalid probability: {name}")
            mass = probabilities.sum(axis=0)
            require(float(np.max(np.abs(mass - 1.0))) <= 2e-5, f"Probability mass mismatch: {name}")
            raw = probabilities.argmax(axis=0).astype(np.uint8)
            output_path = raw_dir / name
            output_manifest[name] = {
                "fold": fold_id,
                "path": str(output_path),
                "sha256": write_png(output_path, raw),
                "probability_mass_max_abs_error": float(np.max(np.abs(mass - 1.0))),
            }
            if name in calibration_names:
                target = png(labels_dir / name)
                matrices += local_confusion(target, raw)
                target_access.append({"name": name, "label_sha256": sha256(labels_dir / name)})
                per_view.append({"name": name, "raw_argmax": confusion_metrics(local_confusion(target, raw))})
            del probabilities, mass, raw
        del model, payload
        torch.cuda.empty_cache()

    require(len(output_manifest) == len(selected_names) and set(output_manifest) == set(selected_names), "Incomplete raw OOF output")
    require({row["name"] for row in target_access} == (set(selected_names) & calibration_names), "Unexpected target-label access")
    raw_metrics = confusion_metrics(matrices)
    result = {
        "experiment_id": EXP,
        "status": "completed_read_only_diagnostic",
        "config_sha256": sha256(args.config),
        "parent": {
            "experiment_id": run["experiment_id"],
            "result_sha256": sha256(args.run_dir / "result.json"),
            "prediction_seal_sha256": sha256(args.run_dir / "prediction_seal.json"),
            "calibration_sha256": sha256(args.run_dir / "calibration_sealed.json"),
        },
        "protocol": {
            "chunk_index": args.chunk_index,
            "chunk_count": args.chunk_count,
            "raw_predictions": len(output_manifest),
            "calibration_labels_decoded": len(target_access),
            "check_labels_decoded": 0,
            "developer_labels_decoded": 0,
            "training_steps": 0,
            "threshold_search": False,
            "frozen_artifacts_verified": 18,
        },
        "metrics": {
            "raw_five_control_argmax": raw_metrics,
            "sealed_five_control_fusion": sealed["metrics"]["five_control"],
            "sealed_sem160_baseline": sealed["metrics"]["baseline"],
        },
        "raw_confusion_rows_gt_columns_prediction": matrices.tolist(),
        "checkpoints": checkpoint_records,
        "target_access": target_access,
        "per_view_calibration": per_view,
        "raw_prediction_manifest": output_manifest,
        "seconds": time.perf_counter() - started,
        "note": "Raw argmax is a reused-calibration mechanism diagnostic only. It cannot promote a candidate or establish hidden-view generalization.",
    }
    write_json(args.output_dir / "result.json", result)
    write_json(args.output_dir / "prediction_manifest.json", output_manifest)
    print(json.dumps({"experiment_id": EXP, "status": result["status"], "chunk_index": args.chunk_index, "chunk_count": args.chunk_count, "raw_predictions": len(output_manifest), "calibration_labels_decoded": len(target_access), "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
