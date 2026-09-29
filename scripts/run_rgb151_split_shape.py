#!/usr/bin/env python3
"""Verify or launch the RGB151 fixed-dev30 long-axis split candidate."""

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

from scripts.run_quality_rgb import source_manifest
from scripts.run_rgb150_dev30 import runtime_environment
from scripts.evaluate_rgb import aggregate, evaluate
from src.quality_protocol import sha256, write_new_json


EID = "RGB-151-STANDARD-ABSGRAD-LAS-DEV30"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
PARENT_CONFIG = ROOT / "experiments/runs/RGB-150-STANDARD-ABSGRAD-DEV30/config.yaml"
CODE_PATHS = (
    "scripts/backbone_explicit_split_worker.py",
    "scripts/build_rgb151_split_diagnostic.py",
    "scripts/preflight_rgb151_split_shape.py",
    "scripts/run_rgb151_split_shape.py",
    "src/standard_long_axis_split.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paths(config: dict) -> tuple[Path, Path, Path, Path, Path]:
    split = ROOT / config["split_path"]
    scene = ROOT / config["data"]["scene"]
    root = ROOT / config["output"]["root"]
    run = ROOT / config["training"]["output"]
    source = ROOT / config["output"]["source"]
    return split, scene, root, run, source


def preflight_path(config: dict, root: Path, kind: str) -> Path:
    revision = config.get("preflight_revision", "r1")
    if revision not in {"r1", "r2"}:
        raise ValueError("Unsupported RGB151 preflight revision")
    return root / f"preflight-{kind}-{revision}"


def validate_recipe(config: dict) -> dict:
    parent = read(PARENT_CONFIG)
    expected_equal = {
        "dataset_manifest": (config["dataset_manifest"], parent["dataset_manifest"]),
        "split_id": (config["split_id"], parent["split_id"]),
        "split_path": (config["split_path"], parent["split_path"]),
        "seed": (config["seed"], parent["seed"]),
        "scene": (config["data"]["scene"], parent["data"]["scene"]),
        "initial_point_count": (
            config["data"]["initial_point_count"],
            parent["data"]["initial_point_count"],
        ),
        "initial_ply_sha256": (
            config["data"]["initial_ply_sha256"],
            parent["data"]["initial_ply_sha256"],
        ),
    }
    for key in (
        "iterations",
        "signed_clone_threshold",
        "absolute_split_threshold",
        "densify_from_iter",
        "densify_until_iter",
        "densification_interval",
        "opacity_reset_interval",
    ):
        expected_equal[f"training.{key}"] = (
            config["training"][key],
            parent["training"][key],
        )
    mismatches = {
        key: {"candidate": values[0], "parent": values[1]}
        for key, values in expected_equal.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(
            f"RGB151 differs from RGB150 outside split shape: {mismatches}"
        )
    shape = config["split_shape"]
    if shape != {
        "method": "long_axis",
        "split_distance": 0.45,
        "opacity_multiplier": 0.6,
        "support_radius_sigma": 3.0,
        "child_count": 2,
        "preserve_native_rng_draw": True,
    }:
        raise ValueError("RGB151 split-shape package changed")
    return parent


def verify(config: dict) -> None:
    parent = validate_recipe(config)
    split, scene, root, _, source = paths(config)
    output = preflight_path(config, root, "cpu")
    output.mkdir(parents=True, exist_ok=False)
    preparation = read(scene / "preparation_manifest.json")
    initial_ply = scene / "sparse/0/points3D.ply"
    if preparation["counts"] != {
        "training_images": 370,
        "evaluation_images": 30,
        "scene_images": 400,
        "retriangulated_points": 79630,
    }:
        raise ValueError("RGB150 prepared scene counts changed")
    if sha256(initial_ply) != config["data"]["initial_ply_sha256"]:
        raise ValueError("Training-only initialization changed")
    if preparation["evaluation"]["initialization_uses_evaluation_images"]:
        raise ValueError("Holdout observations entered initialization")
    parent_result = ROOT / parent["training"]["output"] / "result.json"
    parent_evaluation = ROOT / parent["training"]["output"] / "evaluation/result.json"
    result = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "passed_cpu_preflight",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(CONFIG),
        "parent_config_sha256": sha256(PARENT_CONFIG),
        "split_sha256": sha256(split),
        "initial_ply_sha256": sha256(initial_ply),
        "source": source_manifest(source),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        "matched_rgb150_recipe": True,
        "camera_schedule": {
            "seed": config["seed"],
            "upstream_sampler": "Python random randint without replacement per epoch",
            "split_hook_python_random_calls": 0,
            "fixed_by_recipe": True,
        },
        "rng_policy": {
            "native_torch_normal_draw_preserved_per_matching_split_topology": True,
            "full_training_torch_rng_identity_promised": False,
            "reason": "Shape changes may alter later selected-parent counts and topology",
        },
        "parent_assets": {
            "rgb150_checkpoint_required_to_start_candidate": False,
            "rgb150_training_result_present": parent_result.is_file(),
            "rgb150_evaluation_present": parent_evaluation.is_file(),
        },
        "gpu_checks": "pending; do not run while another project process owns the GPU",
    }
    write_new_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


def gpu_preflight(config: dict, artifact_root: Path) -> None:
    validate_recipe(config)
    _, _, root, _, source = paths(config)
    cpu = read(preflight_path(config, root, "cpu") / "result.json")
    if cpu["status"] != "passed_cpu_preflight":
        raise ValueError("RGB151 CPU preflight has not passed")
    output = preflight_path(config, root, "gpu")
    output.mkdir(parents=True, exist_ok=False)
    environment, python = runtime_environment(artifact_root, source)
    command = [
        str(python),
        str(ROOT / "scripts/preflight_rgb151_split_shape.py"),
        "--source",
        str(source),
        "--evidence",
        str(output / "result.json"),
        "--seed",
        str(config["seed"]),
    ]
    with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        )
        if any(line.strip() and "ToDesk" not in line for line in active.splitlines()):
            raise RuntimeError(
                "Other GPU compute process active outside the shared lock"
            )
        with (output / "preflight.log").open("x") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    if process.returncode:
        raise RuntimeError(f"RGB151 GPU preflight exited {process.returncode}")
    result = read(output / "result.json")
    if result["status"] != "passed_gpu_preflight":
        raise RuntimeError("RGB151 GPU preflight did not pass")
    print(json.dumps(result, indent=2))


def point_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError(f"PLY vertex count missing: {path}")


def train(config: dict, artifact_root: Path) -> None:
    validate_recipe(config)
    split, scene, root, run, source = paths(config)
    preflight = read(preflight_path(config, root, "cpu") / "result.json")
    if preflight["status"] != "passed_cpu_preflight":
        raise ValueError("RGB151 CPU preflight has not passed")
    gpu_check = read(preflight_path(config, root, "gpu") / "result.json")
    if gpu_check["status"] != "passed_gpu_preflight":
        raise ValueError("RGB151 GPU preflight has not passed")
    run.mkdir(parents=True, exist_ok=False)
    environment, python = runtime_environment(artifact_root, source)
    access = ROOT / config["data"]["access"]
    shape = config["split_shape"]
    command = [
        str(python),
        str(ROOT / "scripts/backbone_explicit_split_worker.py"),
        "--source",
        str(source),
        "--access",
        str(access),
        "--evidence",
        str(run / "worker.json"),
        "--seed",
        str(config["seed"]),
        "--split-shape",
        "long-axis",
        "--split-distance",
        str(shape["split_distance"]),
        "--split-opacity-multiplier",
        str(shape["opacity_multiplier"]),
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
        "split_sha256": sha256(split),
        "initialization_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
        "code_sha256": {path: sha256(ROOT / path) for path in CODE_PATHS},
        "free_bytes_start": shutil.disk_usage(run).free,
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            active = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            result["gpu_processes_before"] = active
            if any(
                line.strip() and "ToDesk" not in line for line in active.splitlines()
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
            access_record = read(access)
            if worker["actual_train"] != sorted(access_record["train"]):
                raise RuntimeError("Candidate training cameras changed")
            if worker["actual_evaluation"] != sorted(access_record["evaluation"]):
                raise RuntimeError("Candidate evaluation cameras changed")
            if (
                worker["initialization"]["canonical_ply_sha256"]
                != config["data"]["initial_ply_sha256"]
            ):
                raise RuntimeError("Candidate initialization changed")
            endpoint = run / "model/point_cloud/iteration_30000/point_cloud.ply"
            result.update(
                status="completed_dev30_training",
                worker=worker,
                final_gaussian_count=point_count(endpoint),
                endpoint_sha256=sha256(endpoint),
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


def render_and_evaluate(
    config: dict, artifact_root: Path, evaluation_output: str
) -> None:
    split_path, scene, _, run, source = paths(config)
    training = read(run / "result.json")
    if training["status"] != "completed_dev30_training":
        raise ValueError("RGB151 training is incomplete")
    endpoint = run / "model/point_cloud/iteration_30000/point_cloud.ply"
    if sha256(endpoint) != training["endpoint_sha256"]:
        raise ValueError("RGB151 endpoint changed after training")
    output = run / evaluation_output
    output.mkdir(exist_ok=False)
    image_dir = output / "images"
    image_dir.mkdir()
    split = read(split_path)
    fold = split["primary_fold"]
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": EID,
        "status": "starting",
        "evaluation_output": evaluation_output,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
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
            active = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
            if any(
                line.strip() and "ToDesk" not in line for line in active.splitlines()
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
            model.load_ply(str(endpoint))
            pipeline_parser = argparse.ArgumentParser()
            pipeline = PipelineParams(pipeline_parser).extract(
                pipeline_parser.parse_args([])
            )
            background = torch.zeros(3, device="cuda")
            cameras = read_camera_requests(scene / "sparse/0")
            expected_cameras = set(fold["train_rgb_400"]) | set(
                fold["validation_labeled"]
            )
            if set(cameras) != expected_cameras:
                raise ValueError("Render cameras differ from the frozen split")
            image_hashes = {}
            render_seconds = []
            with torch.inference_mode():
                for index, (name, camera) in enumerate(sorted(cameras.items())):
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    rgb = render(make_native_view(camera), model, pipeline, background)[
                        "render"
                    ]
                    torch.cuda.synchronize()
                    render_seconds.append(time.monotonic() - tick)
                    array = np.rint(
                        rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    Image.fromarray(array).save(image_dir / name)
                    image_hashes[name] = sha256(image_dir / name)
                    if index % 50 == 0:
                        print(
                            json.dumps({"rendered": index + 1, "total": len(cameras)})
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
                endpoint_unchanged=sha256(endpoint) == training["endpoint_sha256"],
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
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
        "--stage",
        choices=("verify", "gpu-preflight", "train", "evaluate"),
        required=True,
    )
    parser.add_argument(
        "--evaluation-output",
        choices=("evaluation", "evaluation-r2", "evaluation-r3"),
        default="evaluation",
        help="Preserve a failed evaluation attempt while keeping the metric protocol fixed.",
    )
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("RGB151 config identity changed")
    if args.stage == "verify":
        verify(config)
    elif args.stage == "gpu-preflight":
        gpu_preflight(config, args.artifact_root.resolve())
    elif args.stage == "train":
        train(config, args.artifact_root.resolve())
    else:
        artifact_root = args.artifact_root.resolve()
        source = paths(config)[-1]
        environment, python = runtime_environment(artifact_root, source)
        # Do not resolve the venv symlink: it points at the system interpreter but
        # selects a different site-packages environment via its original path.
        if Path(sys.executable).absolute() != python.absolute():
            os.execve(
                python,
                [str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
                environment,
            )
        render_and_evaluate(config, artifact_root, args.evaluation_output)


if __name__ == "__main__":
    main()
