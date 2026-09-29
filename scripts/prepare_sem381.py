#!/usr/bin/env python3
"""Bind immutable RGB140, SEM378, labels and local DINOv3 inputs for SEM381."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_dinov3_mask2former import validate_model_dir
from src.quality_mask2former_training import sampled_crop
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-381-DINOV3-VITB-SFP-ALL300"
OUT = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
SEM = ROOT / "outputs/quality/all_data/SEM-370-r1"

CODE_PATHS = [
    "scripts/prepare_sem381.py",
    "scripts/run_sem381.py",
    "scripts/verify_sem381.py",
    "scripts/audit_sem381.py",
    "scripts/build_sem381_comparison.py",
    "scripts/drive_sem381.py",
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


class LazyArrays(dict):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.value = np.zeros(shape, dtype=np.uint8)

    def __missing__(self, name: str) -> np.ndarray:
        return self.value


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
        assert config["experiment_id"] == EID
        assert config["sampler"] == "original uniform SEM378 sampler"
        assert config["iterations"] == 6000 and config["seed"] == 20260805
        assert config["crop"] == 768 and config["stride"] == 512
        assert not config["resize_augmentation"]
        assert config["dinov3"]["frozen"] and config["sfp"]["trainable"]
        if shutil.disk_usage(ROOT).free < 100 * 1024**3:
            raise OSError("100GiB disk reserve reached; preserve prior experiments")

        model_dir = Path(os.environ[config["dinov3"]["model_dir_env"]]).resolve()
        dino_hashes = validate_model_dir(model_dir)
        assert dino_hashes == config["dinov3"]["file_sha256"]
        source_manifest = read(model_dir / "download_manifest.json")
        assert source_manifest["model_id"] == config["dinov3"]["model_id"]
        assert (
            source_manifest["hf_reference_revision"]
            == config["dinov3"]["hf_reference_revision"]
        )
        assert source_manifest["official_weight_hash_matches_modelscope"]
        assert source_manifest["status"] == "downloaded_and_verified"

        OUT.mkdir(parents=True, exist_ok=False)
        output_created = True
        rgb_run = ROOT / config["rgb_run"]
        rgb_eval = ROOT / config["rgb_evaluation"]
        parent = read(rgb_run / "result.json")
        evaluation = read(rgb_eval / "result.json")
        assert parent["experiment_id"] == "RGB-140-STANDARD-ABSGRAD-ALL400"
        assert parent["status"] == "completed_all400"
        assert evaluation["experiment_id"] == parent["experiment_id"]
        assert evaluation["status"] == "completed"
        verify_hashes(rgb_run, parent["artifacts"])
        verify_hashes(rgb_eval / "images", evaluation["image_hashes"])

        baseline_train = read(BASE / "mask2former/result.json")
        baseline_infer = read(BASE / "inference/result.json")
        baseline_export = read(BASE / "export/result.json")
        assert baseline_train["status"] == baseline_infer["status"] == "completed"
        verify_hashes(ROOT, baseline_train["code_sha256"])
        verify_hashes(BASE / "mask2former", baseline_train["artifacts"])
        verify_hashes(BASE / "export/images", baseline_export["image_hashes"])
        assert evaluation["image_hashes"] == baseline_export["image_hashes"]
        assert (
            evaluation["trajectory_metadata_sha256"]
            == baseline_export["trajectory_metadata_sha256"]
        )

        split_path = ROOT / config["split_path"]
        split = read(split_path)
        train_names = sorted(split["semantic_train"])
        assert len(train_names) == 300
        label_manifest = read(SEM / "label_manifest.json")
        label_records = {row["image"]: row for row in label_manifest["records"]}
        label_hashes = {name: sha256(SEM / "labels" / name) for name in train_names}
        for name, digest in label_hashes.items():
            assert digest == label_records[name]["label_png_sha256"]

        parent_samples = [
            json.loads(line)["sample"]
            for line in (BASE / "mask2former/training.jsonl").read_text().splitlines()
        ]
        assert len(parent_samples) == 6000
        rng = random.Random(config["seed"])
        images = LazyArrays((989, 1320, 3))
        labels = LazyArrays((989, 1320))
        replayed = []
        for index, expected in enumerate(parent_samples):
            _, _, actual = sampled_crop(
                rng, train_names, images, labels, config["crop"]
            )
            assert actual == expected, index
            replayed.append(actual)
        sample_path = OUT / "sampling-plan.jsonl"
        sample_path.write_text("".join(json.dumps(row) + "\n" for row in replayed))

        export_images = OUT / "export/images"
        export_images.mkdir(parents=True)
        for name in sorted(evaluation["image_hashes"]):
            shutil.copyfile(rgb_eval / "images" / name, export_images / name)
        verify_hashes(export_images, evaluation["image_hashes"])
        trajectory_names = [f"frame{index:04d}.png" for index in range(144)]
        assert (
            sorted(
                name for name in evaluation["image_hashes"] if name.startswith("frame")
            )
            == trajectory_names
        )
        write_new_json(
            OUT / "export/result.json",
            {
                "schema_version": 1,
                "status": "completed",
                "experiment_id": EID,
                "rgb_experiment_id": parent["experiment_id"],
                "stage": "export",
                "seed": config["seed"],
                "split_id": config["split_id"],
                "split_sha256": sha256(split_path),
                "source_evaluation_sha256": sha256(rgb_eval / "result.json"),
                "source_model_result_sha256": sha256(rgb_run / "result.json"),
                "trajectory_names": trajectory_names,
                "trajectory_metadata_sha256": evaluation["trajectory_metadata_sha256"],
                "image_hashes": evaluation["image_hashes"],
                "rgb_copies_exact": 544,
                "released_count": 400,
                "labeled_count": 300,
                "unlabeled_count": 100,
                "trajectory_count": 144,
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
        runtime = ROOT / "outputs/quality/mask2former/SEM-340"
        pretrained = runtime / "weights/model_final_6b4a3a.pkl"
        assert sha256(pretrained) == baseline_infer["pretrained_sha256"]
        sem378_hashes = {
            "mask2former/result.json": sha256(BASE / "mask2former/result.json"),
            "mask2former/model-3000.pt": sha256(BASE / "mask2former/model-3000.pt"),
            "mask2former/model-6000.pt": sha256(BASE / "mask2former/model-6000.pt"),
            "mask2former/resolved_model.yaml": sha256(
                BASE / "mask2former/resolved_model.yaml"
            ),
            "mask2former/training.jsonl": sha256(BASE / "mask2former/training.jsonl"),
            "inference/result.json": sha256(BASE / "inference/result.json"),
            "export/result.json": sha256(BASE / "export/result.json"),
        }
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
            "rgb140_artifacts": parent["artifacts"],
            "rgb140_image_hashes": evaluation["image_hashes"],
            "sem378_sha256": sem378_hashes,
            "official_mask2former_sha256": sha256(pretrained),
            "dinov3_source_sha256": dino_hashes,
            "dinov3_source_manifest_sha256": sha256(
                model_dir / "download_manifest.json"
            ),
            "sampling_plan_sha256": sha256(sample_path),
            "frozen18": frozen,
        }
        write_new_json(OUT / "binding.json", binding)
        (OUT / "prepare").mkdir()
        write_new_json(
            OUT / "prepare/result.json",
            {
                "schema_version": 1,
                "status": "completed_matched_protocol_preflight",
                "experiment_id": EID,
                "config_sha256": binding["config_sha256"],
                "binding_sha256": sha256(OUT / "binding.json"),
                "sampling_plan_sha256": binding["sampling_plan_sha256"],
                "sampling6000_matches_sem378": True,
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
