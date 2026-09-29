#!/usr/bin/env python3
"""Run the original Standard 3DGS control on spatial_confirmation_v1."""

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
from typing import Any

import run_rgb152_spatial as base


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_quality_rgb import source_manifest
from src.quality_protocol import sha256, write_new_json


EID = "RGB-153-STANDARD-SIGNED-SPATIAL30"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/run_rgb153_spatial_standard.py",
    "scripts/run_rgb152_spatial.py",
    "scripts/backbone_explicit_split_worker.py",
    "scripts/prepare_rgb_colmap_scene.py",
    "scripts/evaluate_rgb.py",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def registered_paths(config: dict[str, Any], artifact_root: Path):
    split = ROOT / config["split_path"]
    scene = ROOT / config["data"]["scene"]
    output_root = ROOT / config["output"]["root"]
    run = ROOT / config["training"]["output"]
    source = artifact_root / config["training"]["source_relative_to_artifact_root"]
    return split, scene, output_root, run, source


def runtime_environment(artifact_root: Path, source: Path):
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    environment = os.environ.copy()
    environment.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + environment.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64")
        + os.pathsep
        + environment.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(
            map(
                str,
                [
                    source,
                    source / "submodules/diff-gaussian-rasterization",
                    source / "submodules/simple-knn",
                    ROOT,
                ],
            )
        ),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return environment, python


def install_base_bindings(config: dict[str, Any], artifact_root: Path) -> None:
    base.EID = EID
    base.CONFIG = CONFIG
    base.CODE_PATHS = CODE_PATHS

    def bound_registered_paths(_config: dict[str, Any]):
        return registered_paths(_config, artifact_root)

    base.registered_paths = bound_registered_paths
    base.runtime_environment = runtime_environment


def train(config: dict[str, Any], artifact_root: Path) -> None:
    split_path, scene, output_root, run, source = registered_paths(config, artifact_root)
    preflight_path = output_root / "preflight-r3/result.json"
    preflight_record = read(preflight_path)
    if preflight_record["status"] != "passed":
        raise ValueError("RGB153 preflight did not pass")
    run.mkdir(parents=True, exist_ok=False)
    environment, python = runtime_environment(artifact_root, source)
    access_path = output_root / "access.json"
    command = [
        str(python),
        str(ROOT / "scripts/backbone_explicit_split_worker.py"),
        "--source",
        str(source),
        "--access",
        str(access_path),
        "--evidence",
        str(run / "worker.json"),
        "--seed",
        str(config["seed"]),
        "--partition",
        str(config["split_id"]),
        "--expected-train",
        str(config["data"]["rgb_train_views"]),
        "--expected-evaluation",
        str(config["data"]["rgb_confirmation_views"]),
        "--expected-universe",
        str(config["data"]["scene_views"]),
        "--",
        "--source_path",
        str(scene),
        "--model_path",
        str(run / "model"),
        "--resolution",
        "1",
        "--data_device",
        "cpu",
        "--iterations",
        str(config["training"]["iterations"]),
        "--densify_grad_threshold",
        str(config["training"]["densify_grad_threshold"]),
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(config["training"]["iterations"]),
        "--checkpoint_iterations",
        str(config["training"]["iterations"]),
        "--eval",
        "--quiet",
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "starting",
        "command": command,
        "seed": config["seed"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "access_sha256": sha256(access_path),
        "preflight_sha256": sha256(preflight_path),
        "initialization_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
        "source": source_manifest(source),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        "free_bytes_start": shutil.disk_usage(run).free,
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            applications = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            result["gpu_processes_before"] = applications
            if any(
                line.strip() and "ToDesk" not in line
                for line in applications.splitlines()
            ):
                raise RuntimeError("Other GPU compute process active outside the shared lock")
            write_new_json(run / "started.json", result)
            with (run / "train.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                with (run / "resources.jsonl").open("x") as monitor:
                    while process.poll() is None:
                        gpu = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        monitor.write(
                            json.dumps(
                                {
                                    "seconds": time.monotonic() - started,
                                    "gpu": gpu.stdout.strip(),
                                    "free_bytes": shutil.disk_usage(run).free,
                                }
                            )
                            + "\n"
                        )
                        monitor.flush()
                        time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"Native process exited {process.returncode}")
            worker = read(run / "worker.json")
            access = read(access_path)
            if worker["actual_train"] != sorted(access["train"]):
                raise RuntimeError("Worker training cameras differ from registered access")
            if worker["actual_evaluation"] != sorted(access["evaluation"]):
                raise RuntimeError("Worker evaluation cameras differ from registered access")
            if worker["initialization"]["common_point_count"] != config["data"]["initial_point_count"]:
                raise RuntimeError("Worker did not load the registered initialization")
            result.update(
                status="completed_spatial_training",
                worker=worker,
                artifacts={
                    str(path.relative_to(run)): sha256(path)
                    for path in (run / "model").rglob("*")
                    if path.is_file()
                },
                free_bytes_end=shutil.disk_usage(run).free,
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(run / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "train", "evaluate"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("RGB153 config identity changed")
    artifact_root = args.artifact_root.resolve()
    install_base_bindings(config, artifact_root)
    if args.stage == "preflight":
        base.preflight(config, artifact_root)
    elif args.stage == "train":
        train(config, artifact_root)
    else:
        base.render_and_evaluate(config, artifact_root)


if __name__ == "__main__":
    main()
