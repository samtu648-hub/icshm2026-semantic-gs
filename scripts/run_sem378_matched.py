#!/usr/bin/env python3
"""Matched SEM371 Mask2Former training on frozen RGB140 renders."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backbone_comparison_stage as shared
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-378-RGB140-ABSGRAD-ALL300"
BASELINE = ROOT / "outputs/quality/backbone_comparison/standard"
SEM370 = ROOT / "outputs/quality/all_data/SEM-370-r1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "mask2former", "inference"), required=True)
    args = parser.parse_args()
    args.artifact_root = args.artifact_root.resolve()
    args.method = "standard-3dgs"

    config_path = ROOT / "experiments/runs" / EID / "config.yaml"
    config = read(config_path)
    args.batch = ROOT / config["output_path"]
    args.rgb_run = ROOT / config["rgb_run"]
    rgb_eval = ROOT / config["rgb_evaluation"]
    args.batch.mkdir(parents=True, exist_ok=True)
    output = args.batch / args.stage
    output.mkdir(exist_ok=False)

    parent = read(args.rgb_run / "result.json")
    evaluation = read(rgb_eval / "result.json")
    split_path = ROOT / config["split_path"]
    split_hash = sha256(split_path)
    sem371_fit = BASELINE / "mask2former/result.json"
    sem371_infer = BASELINE / "inference/result.json"
    code_paths = [
        "scripts/run_sem378_matched.py",
        "scripts/backbone_comparison_stage.py",
        "scripts/run_quality_all300.py",
        "src/quality_mask2former.py",
        "src/quality_mask2former_training.py",
        "src/quality_static_training.py",
    ]
    code = {path: sha256(ROOT / path) for path in code_paths}
    frozen = {
        row["path"]: row["sha256"]
        for row in read(ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json")["artifacts"]
    }
    baseline = {
        "sem371_fit_result": sha256(sem371_fit),
        "sem371_inference_result": sha256(sem371_infer),
        "sem371_model_3000": sha256(BASELINE / "mask2former/model-3000.pt"),
        "sem371_model_6000": sha256(BASELINE / "mask2former/model-6000.pt"),
        "rgb140_result": sha256(args.rgb_run / "result.json"),
        "rgb140_evaluation": sha256(rgb_eval / "result.json"),
    }
    result = {
        "schema_version": 1,
        "status": "starting",
        "experiment_id": EID,
        "rgb_experiment_id": parent.get("experiment_id"),
        "stage": args.stage,
        "git_commit": config["git_commit"],
        "config_sha256": sha256(config_path),
        "seed": config["seed"],
        "dataset_manifest": config["dataset_manifest"],
        "manifest_sha256": sha256(ROOT / config["dataset_manifest"]),
        "split_id": config["split_id"],
        "split_sha256": split_hash,
        "code_sha256": code,
        "baseline_sha256": baseline,
        "quality_scope": "all300 training-view fit; 100 released unlabeled and 144 trajectory frames have no GT",
        "started_unix": time.time(),
    }
    started = time.monotonic()

    def verify() -> None:
        assert read(config_path)["experiment_id"] == EID
        assert sha256(config_path) == result["config_sha256"]
        verify_hashes(ROOT, code)
        verify_hashes(args.rgb_run, parent["artifacts"])
        verify_hashes(rgb_eval / "images", evaluation["image_hashes"])
        assert sha256(args.rgb_run / "result.json") == baseline["rgb140_result"]
        assert sha256(rgb_eval / "result.json") == baseline["rgb140_evaluation"]
        assert sha256(sem371_fit) == baseline["sem371_fit_result"]
        assert sha256(sem371_infer) == baseline["sem371_inference_result"]
        assert sha256(BASELINE / "mask2former/model-3000.pt") == baseline["sem371_model_3000"]
        assert sha256(BASELINE / "mask2former/model-6000.pt") == baseline["sem371_model_6000"]
        verify_hashes(args.artifact_root, frozen)

    try:
        assert parent["experiment_id"] == "RGB-140-STANDARD-ABSGRAD-ALL400"
        assert parent["status"] == "completed_all400"
        assert evaluation["experiment_id"] == parent["experiment_id"]
        assert evaluation["status"] == "completed"
        assert len(evaluation["image_hashes"]) == 544
        names = sorted(evaluation["image_hashes"])
        trajectory_names = [f"frame{index:04d}.png" for index in range(144)]
        assert [name for name in names if name.startswith("frame")] == trajectory_names
        assert len([name for name in names if not name.startswith("frame")]) == 400
        assert split_hash == read(ROOT / "outputs/quality/all_data/RGB-120-r2/split.json").get("sha256", split_hash)
        verify()

        if args.stage == "prepare":
            images = args.batch / "export/images"
            images.mkdir(parents=True, exist_ok=False)
            for name in names:
                shutil.copyfile(rgb_eval / "images" / name, images / name)
            verify_hashes(images, evaluation["image_hashes"])
            export = {
                "schema_version": 1,
                "status": "completed",
                "experiment_id": EID,
                "rgb_experiment_id": parent["experiment_id"],
                "stage": "export",
                "seed": config["seed"],
                "split_id": config["split_id"],
                "split_sha256": split_hash,
                "source_evaluation_sha256": baseline["rgb140_evaluation"],
                "source_model_result_sha256": baseline["rgb140_result"],
                "trajectory_names": trajectory_names,
                "trajectory_metadata_sha256": evaluation["trajectory_metadata_sha256"],
                "image_hashes": evaluation["image_hashes"],
                "rgb_copies_exact": 544,
                "train_count": 400,
                "trajectory_count": 144,
            }
            write_new_json(args.batch / "export/result.json", export)
            snapshot = args.batch / "code"
            snapshot.mkdir(exist_ok=False)
            for name in code_paths:
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            result.update(status="completed", rgb_copies_exact=544, trajectory_count=144)
        else:
            export = read(args.batch / "export/result.json")
            assert export["status"] == "completed" and export["split_sha256"] == split_hash
            verify_hashes(args.batch / "export/images", export["image_hashes"])
            with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
                print("Waiting for shared GPU lock", flush=True)
                queued = time.monotonic()
                fcntl.flock(lock, fcntl.LOCK_EX)
                result["queue_seconds"] = time.monotonic() - queued
                apps = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                    text=True,
                )
                result["gpu_processes_before"] = apps
                if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                    raise RuntimeError("Other GPU compute process active outside the shared lock")
                runtime, torch, np = shared.setup(args)
                torch.set_num_threads(4)
                torch.cuda.reset_peak_memory_stats()
                write_new_json(output / "started.json", result)
                if args.stage == "mask2former":
                    shared.train_stage(args, output, result, torch, np)
                    reference = read(sem371_fit)
                    assert result["initial_model_state_sha256"] == reference["initial_model_state_sha256"]
                    result["initial_state_matches_sem371"] = True
                else:
                    shared.infer_stage(args, output, result, torch, np, runtime)
                result.update(
                    status="completed",
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                )
            verify_hashes(args.batch / "export/images", export["image_hashes"])
        verify()
        result.update(
            frozen18_exact=True,
            rgb140_endpoint_unchanged=True,
            rgb544_hashes_unchanged=True,
            sem371_unchanged=True,
        )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
