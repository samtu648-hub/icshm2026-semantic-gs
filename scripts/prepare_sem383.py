#!/usr/bin/env python3
"""Bind immutable SEM382 controls and SEM383 spatial-adapter inputs."""

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

EID = "SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300"
OUT = ROOT / "outputs/quality/SEM-383/r2"
CONTROL = ROOT / "outputs/quality/SEM-382/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"
PHOTO = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
CODE_PATHS = [
    "scripts/prepare_sem383.py",
    "scripts/diagnose_sem383_inputs.py",
    "scripts/run_sem383.py",
    "scripts/verify_sem383.py",
    "scripts/audit_sem383.py",
    "scripts/build_sem383_comparison.py",
    "scripts/drive_sem383.py",
    "scripts/run_sem382.py",
    "scripts/run_sem381.py",
    "src/quality_dinov3_spatial_sfp.py",
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
    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    started = time.monotonic()
    output_created = False
    try:
        if config["experiment_id"] != EID:
            raise ValueError("Experiment ID mismatch")
        if config["parent_experiment"] != "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300":
            raise ValueError("SEM382 must be the direct control")
        if config["iterations"] != 6000 or config["seed"] != 20260805:
            raise ValueError("Training budget or seed changed")
        if config["crop"] != 768 or config["stride"] != 512:
            raise ValueError("Native crop/inference protocol changed")
        if config["resize_augmentation"]:
            raise ValueError("Resize augmentation is forbidden")
        if {
            config["optimizer"][key]
            for key in (
                "sfp_learning_rate",
                "spatial_adapter_learning_rate",
                "decoder_learning_rate",
            )
        } != {1e-4}:
            raise ValueError("Trainable learning rates differ from SEM382")
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
            raise ValueError("DINOv3 source differs from registration")

        control_binding = read(CONTROL / "binding.json")
        control_train = read(CONTROL / "mask2former/result.json")
        control_infer = read(CONTROL / "inference-6000/result.json")
        control_export = read(CONTROL / "export/result.json")
        control_audit = read(CONTROL / "audit.json")
        control_verify = read(CONTROL / "verification.json")
        if not all(
            record["status"].startswith("completed")
            for record in (control_train, control_infer, control_audit, control_verify)
        ):
            raise ValueError("SEM382 direct control is incomplete")
        expected = config["expected_initial_state"]
        if (
            control_train["initial_trainable_state_sha256"]
            != expected["shared_sem382_trainable_sha256"]
            or control_train["initial_dinov3_state_sha256"] != expected["dinov3_sha256"]
        ):
            raise ValueError("Registered shared initialization differs from SEM382")

        rgb_run = ROOT / config["rgb_run"]
        rgb_eval = ROOT / config["rgb_evaluation"]
        rgb_record = read(rgb_run / "result.json")
        rgb_evaluation = read(rgb_eval / "result.json")
        verify_hashes(rgb_run, rgb_record["artifacts"])
        verify_hashes(rgb_eval / "images", rgb_evaluation["image_hashes"])
        if rgb_evaluation["image_hashes"] != control_export["image_hashes"]:
            raise ValueError("SEM382 export is not the fixed RGB140 evaluation set")

        split_path = ROOT / config["split_path"]
        split = read(split_path)
        train_names = sorted(split["semantic_train"])
        if len(train_names) != 300:
            raise ValueError("Expected exactly 300 training labels")
        diagnostic_names = config["input_diagnostic"]["views"]
        if any(name in train_names for name in diagnostic_names):
            raise ValueError("Input diagnostic unexpectedly has training labels")
        label_manifest = read(SEM / "label_manifest.json")
        label_records = {row["image"]: row for row in label_manifest["records"]}
        label_hashes = {name: sha256(SEM / "labels" / name) for name in train_names}
        for name, digest in label_hashes.items():
            if digest != label_records[name]["label_png_sha256"]:
                raise ValueError(f"Label hash mismatch: {name}")

        sample_source = CONTROL / "sampling-plan.jsonl"
        if sha256(sample_source) != expected["sampling_plan_sha256"]:
            raise ValueError("SEM382 sampling plan differs from registration")
        if len(sample_source.read_text().splitlines()) != 6000:
            raise ValueError("SEM382 sampling plan is incomplete")

        sem382_hashes = {
            path: sha256(CONTROL / path)
            for path in (
                "binding.json",
                "prepare/result.json",
                "mask2former/result.json",
                "mask2former/model-3000.pt",
                "mask2former/model-6000.pt",
                "inference-3000/result.json",
                "inference-6000/result.json",
                "audit.json",
                "verification.json",
                "visual-review.json",
            )
        }
        sem378_hashes = control_binding["sem378_sha256"]
        verify_hashes(CONTROL, sem382_hashes)
        verify_hashes(BASE, sem378_hashes)
        control_predictions = {
            name: sha256(CONTROL / "inference-6000/predictions" / name)
            for name in diagnostic_names
        }
        base_predictions = {
            name: sha256(BASE / "inference/predictions" / name)
            for name in diagnostic_names
        }
        photo_hashes = {name: sha256(PHOTO / name) for name in diagnostic_names}

        OUT.mkdir(parents=True, exist_ok=False)
        output_created = True
        shutil.copyfile(sample_source, OUT / "sampling-plan.jsonl")
        export_images = OUT / "export/images"
        export_images.mkdir(parents=True)
        for name in sorted(control_export["image_hashes"]):
            shutil.copyfile(CONTROL / "export/images" / name, export_images / name)
        verify_hashes(export_images, control_export["image_hashes"])
        write_new_json(
            OUT / "export/result.json",
            {
                **control_export,
                "experiment_id": EID,
                "source_experiment_id": config["parent_experiment"],
                "source_export_sha256": sha256(CONTROL / "export/result.json"),
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
        frozen = control_binding["frozen18"]
        verify_hashes(args.artifact_root, frozen)
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
            "sem382_sha256": sem382_hashes,
            "sem382_diagnostic_prediction_hashes": control_predictions,
            "sem378_sha256": sem378_hashes,
            "sem378_diagnostic_prediction_hashes": base_predictions,
            "diagnostic_photo_hashes": photo_hashes,
            "official_mask2former_sha256": control_binding[
                "official_mask2former_sha256"
            ],
            "dinov3_source_sha256": dino_hashes,
            "sampling_plan_sha256": sha256(OUT / "sampling-plan.jsonl"),
            "initial_trainable_state_sha256": expected["trainable_sha256"],
            "shared_sem382_trainable_sha256": expected[
                "shared_sem382_trainable_sha256"
            ],
            "initial_dinov3_state_sha256": expected["dinov3_sha256"],
            "official_adapter_reference": config["official_adapter_reference"],
            "frozen18": frozen,
        }
        write_new_json(OUT / "binding.json", binding)
        (OUT / "prepare").mkdir()
        write_new_json(
            OUT / "prepare/result.json",
            {
                "schema_version": 1,
                "status": "completed_matched_spatial_adapter_preflight",
                "experiment_id": EID,
                "config_sha256": binding["config_sha256"],
                "binding_sha256": sha256(OUT / "binding.json"),
                "sampling_plan_sha256": binding["sampling_plan_sha256"],
                "sampling6000_matches_sem382": True,
                "shared_initial_state_matches_sem382": True,
                "labels_exact": 300,
                "rgb_copies_exact": 544,
                "diagnostic_photos_exact": len(photo_hashes),
                "dinov3_files_exact": len(dino_hashes),
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
