#!/usr/bin/env python3
"""Prepare, train, render, and evaluate RGB-150 on the fixed dev30 split."""

from __future__ import annotations

import argparse
import csv
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

from scripts.evaluate_rgb import aggregate, evaluate
from scripts.run_quality_rgb import source_manifest
from src.quality_protocol import sha256, verify_hashes, write_new_json


EID = "RGB-150-STANDARD-ABSGRAD-DEV30"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE_PATHS = (
    "scripts/backbone_explicit_split_worker.py",
    "scripts/prepare_rgb_colmap_scene.py",
    "scripts/run_rgb150_dev30.py",
    "scripts/evaluate_rgb.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_environment(
    artifact_root: Path, source: Path
) -> tuple[dict[str, str], Path]:
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


def registered_paths(config: dict) -> tuple[Path, Path, Path, Path, Path]:
    split = ROOT / config["split_path"]
    scene = ROOT / config["data"]["scene"]
    output_root = ROOT / config["output"]["root"]
    run = ROOT / config["training"]["output"]
    source = ROOT / config["output"]["source"]
    return split, scene, output_root, run, source


def preflight(config: dict, artifact_root: Path) -> None:
    split_path, scene, output_root, _, source = registered_paths(config)
    output = output_root / "preflight-r1"
    output.mkdir(parents=True, exist_ok=False)
    split = read(split_path)
    fold = split["primary_fold"]
    access = {
        "schema_version": 1,
        "partition": split["split_id"],
        "split_sha256": sha256(split_path),
        "train": fold["train_rgb_400"],
        "evaluation": fold["validation_labeled"],
        "policy": "Evaluation RGB excluded from optimization, initialization observations, and point colors",
    }
    access_path = output_root / "access.json"
    write_new_json(access_path, access)
    preparation = read(scene / "preparation_manifest.json")
    if preparation["counts"] != {
        "training_images": 370,
        "evaluation_images": 30,
        "scene_images": 400,
        "retriangulated_points": 79630,
    }:
        raise ValueError("Prepared scene counts differ from registration")
    if preparation["evaluation"]["initialization_uses_evaluation_images"]:
        raise ValueError("Holdout observations entered initialization")
    if sha256(scene / "sparse/0/points3D.ply") != config["data"]["initial_ply_sha256"]:
        raise ValueError("Training-only initialization PLY changed")

    environment, python = runtime_environment(artifact_root, source)
    command = [
        str(python),
        str(ROOT / "scripts/validate_explicit_eval_split.py"),
        "--candidate-root",
        str(source),
        "--candidate",
        "standard-3dgs",
        "--scene",
        str(scene),
        "--expected-train",
        "370",
        "--expected-test",
        "30",
    ]
    checked = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=True
    )
    validation = json.loads(checked.stdout.strip().splitlines()[-1])
    if not validation["passes"]:
        raise ValueError("Upstream explicit split validation failed")
    write_new_json(
        output / "result.json",
        {
            "schema_version": 1,
            "status": "passed",
            "experiment_id": EID,
            "config_sha256": sha256(CONFIG),
            "split_sha256": sha256(split_path),
            "access_sha256": sha256(access_path),
            "preparation_manifest_sha256": sha256(scene / "preparation_manifest.json"),
            "initial_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
            "source": source_manifest(source),
            "validation": validation,
            "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        },
    )


def train(config: dict, artifact_root: Path) -> None:
    split_path, scene, output_root, run, source = registered_paths(config)
    preflight_record = read(output_root / "preflight-r1/result.json")
    if preflight_record["status"] != "passed":
        raise ValueError("RGB-150 preflight did not pass")
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
        str(config["training"]["signed_clone_threshold"]),
        "--densify_grad_abs_threshold",
        str(config["training"]["absolute_split_threshold"]),
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(config["training"]["iterations"]),
        "--checkpoint_iterations",
        str(config["training"]["iterations"]),
        "--eval",
        "--quiet",
    ]
    result: dict[str, object] = {
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
        "preflight_sha256": sha256(output_root / "preflight-r1/result.json"),
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
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )
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
                raise RuntimeError(
                    "Worker training cameras differ from registered access"
                )
            if worker["actual_evaluation"] != sorted(access["evaluation"]):
                raise RuntimeError(
                    "Worker evaluation cameras differ from registered access"
                )
            if worker["initialization"]["common_point_count"] != 79630:
                raise RuntimeError(
                    "Worker did not load the registered 79,630-point initialization"
                )
            result.update(
                status="completed_dev30_training",
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


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_and_evaluate(config: dict, artifact_root: Path) -> None:
    split_path, scene, _, run, source = registered_paths(config)
    parent = read(run / "result.json")
    if parent["status"] != "completed_dev30_training":
        raise ValueError("RGB-150 training is incomplete")
    verify_hashes(run, parent["artifacts"])
    output = run / "evaluation"
    output.mkdir(exist_ok=False)
    image_dir = output / "images"
    image_dir.mkdir()
    split = read(split_path)
    fold = split["primary_fold"]
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "starting",
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(split_path),
        "source_result_sha256": sha256(run / "result.json"),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
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
            if any(
                line.strip() and "ToDesk" not in line
                for line in applications.splitlines()
            ):
                raise RuntimeError(
                    "Other GPU compute process active outside the shared lock"
                )
            environment, _ = runtime_environment(artifact_root, source)
            os.environ.update(environment)
            sys.path[:0] = [
                str(source),
                str(source / "submodules/diff-gaussian-rasterization"),
                str(source / "submodules/simple-knn"),
            ]
            import numpy as np
            import torch  # type: ignore[import-not-found]
            from PIL import Image
            from arguments import PipelineParams  # type: ignore[import-not-found]
            from gaussian_renderer import GaussianModel, render  # type: ignore[import-not-found]
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view

            torch.set_num_threads(4)
            torch.manual_seed(config["seed"])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            model = GaussianModel(3)
            model.load_ply(
                str(run / "model/point_cloud/iteration_30000/point_cloud.ply")
            )
            pipeline_parser = argparse.ArgumentParser()
            pipeline = PipelineParams(pipeline_parser).extract(
                pipeline_parser.parse_args([])
            )
            background = torch.zeros(3, device="cuda")
            cameras = read_camera_requests(scene / "sparse/0")
            if set(cameras) != set(fold["train_rgb_400"]) | set(
                fold["validation_labeled"]
            ):
                raise ValueError("Render cameras differ from the frozen split")
            image_hashes = {}
            render_seconds = []
            with torch.inference_mode():
                for index, (name, camera) in enumerate(sorted(cameras.items())):
                    torch.cuda.synchronize()
                    render_started = time.monotonic()
                    rgb = render(make_native_view(camera), model, pipeline, background)[
                        "render"
                    ]
                    torch.cuda.synchronize()
                    render_seconds.append(time.monotonic() - render_started)
                    array = np.rint(
                        rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    Image.fromarray(array).save(image_dir / name)
                    image_hashes[name] = sha256(image_dir / name)
                    if index % 50 == 0:
                        print(
                            json.dumps({"rendered": index + 1, "total": len(cameras)}),
                            flush=True,
                        )
            del model, background
            torch.cuda.empty_cache()

            partitions = {
                "train": list(fold["train_rgb_400"]),
                "holdout": list(fold["validation_labeled"]),
            }
            metric_rows = {}
            metric_summaries = {}
            implementation = {}
            for partition, names in partitions.items():
                rows, implementation = evaluate(
                    scene / "images",
                    image_dir,
                    names,
                    lpips_device="cuda",
                    skip_lpips=False,
                )
                metric_rows[partition] = rows
                metric_summaries[partition] = {
                    key: aggregate(rows, key, lower_is_worse=key != "LPIPS_Alex")
                    for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
                }
                write_csv(output / f"{partition}.csv", rows)
            holdout_rows = {str(row["view"]): row for row in metric_rows["holdout"]}
            group_metrics = {}
            for group in split["groups"]:
                rows = [holdout_rows[name] for name in group["images"]]
                group_metrics[group["group_id"]] = {
                    key: aggregate(rows, key, lower_is_worse=key != "LPIPS_Alex")
                    for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
                }
            result.update(
                status="completed",
                image_hashes=image_hashes,
                render_seconds=render_seconds,
                render_median_seconds=float(np.median(render_seconds[1:])),
                metrics=metric_summaries,
                group_metrics=group_metrics,
                implementation=implementation,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        verify_hashes(run, parent["artifacts"])
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("preflight", "train", "evaluate"), required=True
    )
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("RGB-150 config identity changed")
    artifact_root = args.artifact_root.resolve()
    if args.stage == "preflight":
        preflight(config, artifact_root)
    elif args.stage == "train":
        train(config, artifact_root)
    else:
        render_and_evaluate(config, artifact_root)


if __name__ == "__main__":
    main()
