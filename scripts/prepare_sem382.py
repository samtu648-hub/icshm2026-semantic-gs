#!/usr/bin/env python3
"""Bind the immutable SEM381 inputs for the SEM382 SFP-LR comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_dinov3_mask2former import validate_model_dir
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300"
OUT = ROOT / "outputs/quality/SEM-382/r1"
PARENT = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"

CODE_PATHS = [
    "scripts/prepare_sem382.py",
    "scripts/run_sem382.py",
    "scripts/verify_sem382.py",
    "scripts/audit_sem382.py",
    "scripts/build_sem382_comparison.py",
    "scripts/drive_sem382.py",
    "scripts/run_sem381.py",
    "src/quality_dinov3_mask2former.py",
    "src/quality_mask2former.py",
    "src/quality_mask2former_training.py",
    "src/quality_static_training.py",
    "src/semantic_cable_binary.py",
    "src/semantic2d.py",
    "src/quality_experts.py",
    "src/quality_protocol.py",
    "src/quality_semantic_audit.py",
]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    started = time.monotonic()
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    output_created = False
    try:
        if config["experiment_id"] != EID:
            raise ValueError("Experiment ID mismatch")
        if config["parent_experiment"] != "SEM-381-DINOV3-VITB-SFP-ALL300":
            raise ValueError("SEM381 must be the direct control")
        if config["iterations"] != 6000 or config["seed"] != 20260805:
            raise ValueError("Training budget or seed changed")
        if config["crop"] != 768 or config["stride"] != 512:
            raise ValueError("Native crop/inference protocol changed")
        if config["resize_augmentation"]:
            raise ValueError("Resize augmentation is forbidden")
        if config["optimizer"]["sfp_learning_rate"] != 1e-4:
            raise ValueError("SFP LR treatment mismatch")
        if config["optimizer"]["decoder_learning_rate"] != 1e-4:
            raise ValueError("Decoder LR changed")
        if config["optimizer"]["dinov3_learning_rate"] != 0.0:
            raise ValueError("DINOv3 must remain frozen")
        if (
            config["primary_checkpoint"] != 6000
            or config["analysis_checkpoint"] != 3000
        ):
            raise ValueError("Checkpoint roles changed")
        if shutil.disk_usage(ROOT).free < 100 * 1024**3:
            raise OSError("100GiB disk reserve reached; preserve prior experiments")

        model_dir = Path(os.environ[config["dinov3"]["model_dir_env"]]).resolve()
        dino_hashes = validate_model_dir(model_dir)
        if dino_hashes != config["dinov3"]["file_sha256"]:
            raise ValueError("DINOv3 source differs from the registered source")
        source_manifest = read(model_dir / "download_manifest.json")
        if (
            source_manifest["model_id"] != config["dinov3"]["model_id"]
            or source_manifest["hf_reference_revision"]
            != config["dinov3"]["hf_reference_revision"]
            or not source_manifest["official_weight_hash_matches_modelscope"]
            or source_manifest["status"] != "downloaded_and_verified"
        ):
            raise ValueError("DINOv3 download manifest mismatch")

        parent_binding = read(PARENT / "binding.json")
        parent_prepare = read(PARENT / "prepare/result.json")
        parent_train = read(PARENT / "mask2former/result.json")
        parent_infer = read(PARENT / "inference/result.json")
        parent_export = read(PARENT / "export/result.json")
        parent_audit = read(PARENT / "audit.json")
        parent_verify = read(PARENT / "verification.json")
        if not all(
            record["status"].startswith("completed")
            for record in (
                parent_prepare,
                parent_train,
                parent_infer,
                parent_audit,
                parent_verify,
            )
        ):
            raise ValueError("SEM381 direct control is incomplete")
        expected_state = config["expected_initial_state"]
        if (
            parent_train["initial_trainable_state_sha256"]
            != expected_state["trainable_sha256"]
            or parent_train["initial_dinov3_state_sha256"]
            != expected_state["dinov3_sha256"]
        ):
            raise ValueError("Registered initialization does not match SEM381")
        verify_hashes(PARENT / "mask2former", parent_train["artifacts"])
        verify_hashes(PARENT / "export/images", parent_export["image_hashes"])
        verify_hashes(
            PARENT / "inference/predictions", parent_audit["prediction_hashes"]
        )

        rgb_run = ROOT / config["rgb_run"]
        rgb_eval = ROOT / config["rgb_evaluation"]
        rgb_record = read(rgb_run / "result.json")
        rgb_evaluation = read(rgb_eval / "result.json")
        if (
            rgb_record["status"] != "completed_all400"
            or rgb_evaluation["status"] != "completed"
        ):
            raise ValueError("RGB140 source is incomplete")
        verify_hashes(rgb_run, rgb_record["artifacts"])
        verify_hashes(rgb_eval / "images", rgb_evaluation["image_hashes"])
        if rgb_evaluation["image_hashes"] != parent_export["image_hashes"]:
            raise ValueError("SEM381 export is not the fixed RGB140 evaluation set")

        split_path = ROOT / config["split_path"]
        split = read(split_path)
        train_names = sorted(split["semantic_train"])
        if len(train_names) != 300:
            raise ValueError("Expected exactly 300 training labels")
        label_manifest = read(SEM / "label_manifest.json")
        label_records = {row["image"]: row for row in label_manifest["records"]}
        label_hashes = {name: sha256(SEM / "labels" / name) for name in train_names}
        for name, digest in label_hashes.items():
            if digest != label_records[name]["label_png_sha256"]:
                raise ValueError(f"Label hash mismatch: {name}")

        sample_source = PARENT / "sampling-plan.jsonl"
        if sha256(sample_source) != expected_state["sampling_plan_sha256"]:
            raise ValueError("SEM381 sampling plan differs from registration")
        sample_rows = sample_source.read_text().splitlines()
        if len(sample_rows) != 6000:
            raise ValueError("SEM381 sampling plan is incomplete")

        OUT.mkdir(parents=True, exist_ok=False)
        output_created = True
        shutil.copyfile(sample_source, OUT / "sampling-plan.jsonl")
        export_images = OUT / "export/images"
        export_images.mkdir(parents=True)
        for name in sorted(parent_export["image_hashes"]):
            shutil.copyfile(PARENT / "export/images" / name, export_images / name)
        verify_hashes(export_images, parent_export["image_hashes"])
        write_new_json(
            OUT / "export/result.json",
            {
                **parent_export,
                "experiment_id": EID,
                "source_experiment_id": "SEM-381-DINOV3-VITB-SFP-ALL300",
                "source_export_sha256": sha256(PARENT / "export/result.json"),
                "source_evaluation_sha256": sha256(rgb_eval / "result.json"),
                "rgb_copies_exact": 544,
            },
        )

        code_hashes = {path: sha256(ROOT / path) for path in CODE_PATHS}
        snapshot = OUT / "code"
        for path in CODE_PATHS:
            destination = snapshot / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, destination)
        frozen = {
            row["path"]: row["sha256"]
            for row in read(
                ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json"
            )["artifacts"]
        }
        verify_hashes(args.artifact_root, frozen)

        sem381_hashes = {
            "binding.json": sha256(PARENT / "binding.json"),
            "prepare/result.json": sha256(PARENT / "prepare/result.json"),
            "mask2former/result.json": sha256(PARENT / "mask2former/result.json"),
            "mask2former/model-3000.pt": sha256(PARENT / "mask2former/model-3000.pt"),
            "mask2former/model-6000.pt": sha256(PARENT / "mask2former/model-6000.pt"),
            "mask2former/resolved_model.yaml": sha256(
                PARENT / "mask2former/resolved_model.yaml"
            ),
            "mask2former/training.jsonl": sha256(PARENT / "mask2former/training.jsonl"),
            "inference/result.json": sha256(PARENT / "inference/result.json"),
            "export/result.json": sha256(PARENT / "export/result.json"),
            "audit.json": sha256(PARENT / "audit.json"),
            "verification.json": sha256(PARENT / "verification.json"),
        }
        sem378_hashes = parent_binding["sem378_sha256"]
        verify_hashes(PARENT, sem381_hashes)
        verify_hashes(BASE, sem378_hashes)
        binding = {
            "schema_version": 1,
            "experiment_id": EID,
            "config_sha256": sha256(config_path),
            "code_sha256": code_hashes,
            "manifest_sha256": sha256(ROOT / config["dataset_manifest"]),
            "split_sha256": sha256(split_path),
            "label_manifest_sha256": sha256(SEM / "label_manifest.json"),
            "label_hashes": label_hashes,
            "rgb140_result_sha256": sha256(rgb_run / "result.json"),
            "rgb140_evaluation_sha256": sha256(rgb_eval / "result.json"),
            "rgb140_artifacts": rgb_record["artifacts"],
            "rgb140_image_hashes": rgb_evaluation["image_hashes"],
            "sem381_sha256": sem381_hashes,
            "sem381_prediction_hashes": parent_audit["prediction_hashes"],
            "sem378_sha256": sem378_hashes,
            "official_mask2former_sha256": parent_binding[
                "official_mask2former_sha256"
            ],
            "dinov3_source_sha256": dino_hashes,
            "dinov3_source_manifest_sha256": sha256(
                model_dir / "download_manifest.json"
            ),
            "sampling_plan_sha256": sha256(OUT / "sampling-plan.jsonl"),
            "initial_trainable_state_sha256": expected_state["trainable_sha256"],
            "initial_dinov3_state_sha256": expected_state["dinov3_sha256"],
            "frozen18": frozen,
        }
        write_new_json(OUT / "binding.json", binding)
        (OUT / "prepare").mkdir()
        write_new_json(
            OUT / "prepare/result.json",
            {
                "schema_version": 1,
                "status": "completed_matched_single_variable_preflight",
                "experiment_id": EID,
                "config_sha256": binding["config_sha256"],
                "binding_sha256": sha256(OUT / "binding.json"),
                "sampling_plan_sha256": binding["sampling_plan_sha256"],
                "sampling6000_matches_sem381": True,
                "initial_state_matches_sem381": True,
                "labels_exact": 300,
                "rgb_copies_exact": 544,
                "trajectory_metadata_exact": True,
                "dinov3_files_exact": len(dino_hashes),
                "dinov3_weight_bytes": config["dinov3"]["weight_bytes"],
                "frozen18_exact": True,
                "minimum_free_gib_at_start": 100,
                "seconds": time.monotonic() - started,
                "script_sha256": sha256(Path(__file__)),
            },
        )
        print(OUT / "prepare/result.json")
    except BaseException as exc:
        if output_created:
            failure = OUT / "prepare-failure.json"
            if not failure.exists():
                failure.write_text(
                    json.dumps(
                        {
                            "status": "failed_preserved",
                            "experiment_id": EID,
                            "error": f"{type(exc).__name__}: {exc}",
                            "seconds": time.monotonic() - started,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        raise


if __name__ == "__main__":
    main()
