#!/usr/bin/env python3
"""Run one immutable RGB147-149 arm with source/config/data bindings."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_quality_rgb import source_manifest
from src.quality_protocol import sha256, write_new_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    config = json.loads(args.config.read_text())
    preflight_path = ROOT / "outputs/quality/rgb147/preflight-r4/result.json"
    if (
        config["auxiliary_supervision"].get("coefficient_status", "frozen")
        != "frozen_pretraining"
        and config["auxiliary_supervision"]["mode"] != "control"
    ):
        raise ValueError(
            "Auxiliary coefficient is not frozen by pretraining calibration"
        )
    if args.iterations == 30_000 and not args.preflight:
        if (
            not preflight_path.exists()
            or json.loads(preflight_path.read_text()).get("status") != "passed"
        ):
            raise ValueError("Full training requires the passed registered preflight")
    if args.resume_state:
        if not args.output.exists():
            raise FileNotFoundError("Resume output directory does not exist")
        suffix = f"resume-{int(time.time())}"
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        suffix = "initial"

    source = ROOT / config["source"]
    scene = ROOT / config["data"]["scene"]
    if sha256(scene / "sparse/0/points3D.ply") != config["data"]["initial_ply_sha256"]:
        raise ValueError("Frozen RGB140 initialization changed")
    calibration_path = ROOT / "outputs/quality/rgb147/calibration-r2/result.json"
    if config["auxiliary_supervision"]["mode"] != "control":
        calibration = json.loads(calibration_path.read_text())
        if config["coefficient_calibration"]["artifact"] != str(
            calibration_path.relative_to(ROOT)
        ) or config["coefficient_calibration"]["artifact_sha256"] != sha256(
            calibration_path
        ):
            raise ValueError("Config calibration binding differs from frozen artifact")
        key = (
            "lambda_region_rgb"
            if config["auxiliary_supervision"]["mode"] == "region_rgb"
            else "lambda_region_gradient"
        )
        if (
            abs(config["auxiliary_supervision"]["coefficient"] - calibration[key])
            > 1e-12
        ):
            raise ValueError("Config coefficient differs from frozen calibration")

    cuda = args.artifact_root.resolve() / "r1/toolchain/cuda-12.8"
    python = args.artifact_root.resolve() / "r1/envs/r1-cuda128/bin/python"
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
        TORCH_HOME=str(args.artifact_root.resolve() / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    saves = (
        [args.iterations] if args.iterations < 7000 else [7000, 15000, args.iterations]
    )
    worker_evidence = args.output / f"worker-{suffix}.json"
    command = [
        str(python),
        str(ROOT / "scripts/worker_standard_rgb_supervision.py"),
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        "--attempt",
        suffix,
    ]
    if args.resume_state:
        command += ["--resume-state", str(args.resume_state.resolve())]
    command += [
        "--",
        "--source",
        str(source),
        "--access",
        str(ROOT / "outputs/quality/all_data/RGB-120-r2/access.json"),
        "--evidence",
        str(worker_evidence),
        "--seed",
        "0",
        "--expected-init-policy",
        "same RGB130 original 303-view DLT initialization; all400 photometric training",
        "--",
        "--source_path",
        str(scene),
        "--model_path",
        str(args.output / "model"),
        "--resolution",
        "1",
        "--data_device",
        "cpu",
        "--iterations",
        str(args.iterations),
        "--densify_grad_threshold",
        "0.0002",
        "--densify_grad_abs_threshold",
        "0.0004",
        "--test_iterations",
        "-1",
        "--save_iterations",
        *map(str, saves),
        "--checkpoint_iterations",
        str(args.iterations),
        "--quiet",
    ]
    code = [
        "scripts/run_standard_rgb_supervision.py",
        "scripts/worker_standard_rgb_supervision.py",
        "scripts/backbone_all400_worker.py",
        "src/standard_rgb_supervision.py",
    ]
    result_path = args.output / f"result-{suffix}.json"
    result = {
        "experiment_id": config["experiment_id"],
        "status": "starting",
        "command": command,
        "seed": 0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(ROOT / config["dataset_manifest"]),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "label_manifest_sha256": sha256(
            ROOT / config["auxiliary_supervision"]["label_manifest"]
        ),
        "calibration_sha256": None
        if config["auxiliary_supervision"]["mode"] == "control"
        else sha256(calibration_path),
        "preflight_sha256": sha256(preflight_path) if preflight_path.exists() else None,
        "initialization_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
        "source": source_manifest(source),
        "code_sha256": {path: sha256(ROOT / path) for path in code},
        "actual_iterations": args.iterations,
        "auxiliary_supervision": config["auxiliary_supervision"],
        "free_bytes_start": shutil.disk_usage(args.output).free,
    }
    if not args.resume_state:
        for path in code:
            destination = args.output / "code" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, destination)
        shutil.copy2(args.config, args.output / "code/config.yaml")
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["lock_wait_seconds"] = time.monotonic() - started
            apps = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            result["gpu_processes_before"] = apps
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError("Other GPU compute process active")
            write_new_json(args.output / f"started-{suffix}.json", result)
            training_started = time.monotonic()
            with (
                (args.output / f"train-{suffix}.log").open("x") as log,
                (args.output / f"resources-{suffix}.jsonl").open("x") as monitor,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=source,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    gpu = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    monitor.write(
                        json.dumps(
                            {
                                "seconds": time.monotonic() - training_started,
                                "gpu": gpu.stdout.strip(),
                                "free_bytes": shutil.disk_usage(args.output).free,
                            }
                        )
                        + "\n"
                    )
                    monitor.flush()
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f"Native worker exited {process.returncode}")
            worker = json.loads(worker_evidence.read_text())
            if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
                raise RuntimeError("Worker did not use the registered all400 split")
            if worker["initialization"]["common_point_count"] != 72_292:
                raise RuntimeError("Worker did not load the common initialization")
            result.update(
                status="completed_all400"
                if args.iterations == 30_000
                else "completed_preflight",
                worker=worker,
                training_wall_seconds_excluding_lock=time.monotonic()
                - training_started,
                artifacts={
                    str(path.relative_to(args.output)): sha256(path)
                    for path in (args.output / "model").rglob("*")
                    if path.is_file()
                },
            )
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result.update(
            seconds=time.monotonic() - started,
            free_bytes_end=shutil.disk_usage(args.output).free,
        )
        write_new_json(result_path, result)
        if not args.resume_state:
            shutil.copy2(result_path, args.output / "result.json")
        if result.get("status") in {"completed_all400", "completed_preflight"}:
            completed = args.output / "completed.json"
            if not completed.exists():
                shutil.copy2(result_path, completed)
        latest = args.output / "latest.json"
        temporary = latest.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "result": result_path.name,
                    "status": result["status"],
                    "resume_state": None
                    if not args.resume_state
                    else str(args.resume_state.resolve()),
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, latest)


if __name__ == "__main__":
    main()
