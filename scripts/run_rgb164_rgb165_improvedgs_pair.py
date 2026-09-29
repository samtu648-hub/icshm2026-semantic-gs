#!/usr/bin/env python3
"""Run the matched no-component/full ImprovedGS all-400 reconstruction pair."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

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
from src.quality_protocol import sha256, write_new_json

SOURCE_REL = Path("outputs/quality/source/improvedgs-full")
ARMS = (
    ("RGB-164-IMPROVEDGS-MATCHED-CONTROL", "RGB-164", {"las": True, "eas": False, "rap": False, "mu": False}),
    ("RGB-165-IMPROVEDGS-FULL-ALL400", "RGB-165", {"las": True, "eas": True, "rap": True, "mu": True}),
)


def source_record(source: Path) -> dict[str, object]:
    return {
        "repository": "https://github.com/XiaoBin2001/Improved-GS",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()),
        "compatibility_header_sha256": sha256(source / "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"),
        "rasterizer_binary_sha256": sha256(next((source / "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization").glob("_C*.so"))),
        "simple_knn_binary_sha256": sha256(next((source / "submodules/simple-knn/simple_knn").glob("_C*.so"))),
        "fused_ssim_binary_sha256": sha256(next((source / "submodules/fused-ssim").glob("fused_ssim_cuda*.so"))),
        "license": "non-commercial research and evaluation",
    }


def build_env(artifact_root: Path, source: Path) -> dict[str, str]:
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(map(str, [
            source,
            source / "submodules/diff-gaussian-rasterization",
            source / "submodules/simple-knn",
            source / "submodules/fused-ssim",
            ROOT,
        ])),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def run_arm(
    *, artifact_root: Path, source: Path, stage: str, iterations: int,
    experiment_id: str, output_key: str, components: dict[str, bool], attempt: str,
    save_iterations: list[int] | None = None, runner_path: Path | None = None,
) -> dict[str, object]:
    output = ROOT / "outputs/quality" / output_key / attempt
    output.mkdir(parents=True, exist_ok=False)
    scene_root = ROOT / "outputs/quality/all_data/RGB-120-r2"
    config = ROOT / "experiments/runs" / experiment_id / "config.yaml"
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    if save_iterations is None:
        save_iterations = [iterations] if stage == "preflight" else [15_000, 30_000]
    saves = [str(value) for value in save_iterations]
    flag = {name: "true" if enabled else "false" for name, enabled in components.items()}
    command = [
        str(python), str(ROOT / "scripts/improvedgs_all400_worker.py"),
        "--source", str(source),
        "--access", str(scene_root / "access.json"),
        "--evidence", str(output / "worker.json"),
        "--seed", "0", "--",
        "--source_path", str(scene_root / "scene"),
        "--model_path", str(output / "model"),
        "--resolution", "1", "--data_device", "cpu", "--eval", "false",
        "--iterations", str(iterations), "--training_method", "improvedgs",
        "--use_las", flag["las"], "--use_eas", flag["eas"],
        "--use_rap", flag["rap"], "--use_mu", flag["mu"],
        "--budget", "1000000", "--edge_sample_cams", "10",
        "--test_iterations", "-1", "--save_iterations", *saves,
        "--empty_cache_interval", "200", "--quiet",
    ]
    record: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "stage": stage,
        "status": "starting",
        "iterations": iterations,
        "components": components,
        "seed": 0,
        "budget": 1_000_000,
        "command": command,
        "config_sha256": sha256(config),
        "source": source_record(source),
        "code_sha256": {
            "worker": sha256(ROOT / "scripts/improvedgs_all400_worker.py"),
            "runner": sha256(runner_path or ROOT / "scripts/run_rgb164_rgb165_improvedgs_pair.py"),
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "initialization_ply_sha256": sha256(scene_root / "scene/sparse/0/points3D.ply"),
        "free_bytes_start": shutil.disk_usage(output).free,
    }
    started = time.monotonic()
    try:
        write_new_json(output / "started.json", record)
        env = build_env(artifact_root, source)
        with (output / "train.log").open("x") as log:
            process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            with (output / "resources.jsonl").open("x") as monitor:
                while process.poll() is None:
                    gpu = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True,
                    )
                    monitor.write(json.dumps({
                        "seconds": time.monotonic() - started,
                        "gpu": gpu.stdout.strip(),
                        "free_bytes": shutil.disk_usage(output).free,
                    }) + "\n")
                    monitor.flush()
                    time.sleep(10)
        if process.returncode:
            raise RuntimeError(f"training exited {process.returncode}; see {output / 'train.log'}")
        worker = json.loads((output / "worker.json").read_text())
        if worker["status"] != "completed_native_execution":
            raise RuntimeError("native worker did not complete")
        if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
            raise RuntimeError("registered all400 split was not preserved")
        point_cloud = output / "model/point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        if not point_cloud.is_file():
            raise FileNotFoundError(point_cloud)
        record.update(
            status="completed",
            worker_status=worker["status"],
            peak_reserved_mib=worker["peak_reserved_mib"],
            peak_allocated_mib=worker["peak_allocated_mib"],
            sample_trace=worker["sample_trace"],
            point_cloud_sha256=sha256(point_cloud),
            point_cloud_bytes=point_cloud.stat().st_size,
        )
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record["free_bytes_end"] = shutil.disk_usage(output).free
        write_new_json(output / "result.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root_default())
    parser.add_argument("--stage", choices=("preflight", "full", "all"), default="all")
    parser.add_argument("--preflight-attempt", default="preflight-r1")
    parser.add_argument("--full-attempt", default="full-r1")
    parser.add_argument("--pair-attempt", default="r1")
    args = parser.parse_args()
    source = ROOT / SOURCE_REL
    if source_record(source)["commit"] != "20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a":
        raise ValueError("ImprovedGS source commit changed")

    stages = ["preflight", "full"] if args.stage == "all" else [args.stage]
    summary: dict[str, object] = {"schema_version": 1, "status": "starting", "stages": {}}
    summary_path = ROOT / f"outputs/quality/RGB-164-RGB-165-pair-summary-{args.pair_attempt}.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True,
            )
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError(f"Other GPU compute process active:\n{apps}")
            for stage in stages:
                iterations = 601 if stage == "preflight" else 30_000
                attempt = args.preflight_attempt if stage == "preflight" else args.full_attempt
                results = []
                for experiment_id, output_key, components in ARMS:
                    results.append(run_arm(
                        artifact_root=args.artifact_root,
                        source=source,
                        stage=stage,
                        iterations=iterations,
                        experiment_id=experiment_id,
                        output_key=output_key,
                        components=components,
                        attempt=attempt,
                    ))
                if results[0]["sample_trace"] != results[1]["sample_trace"]:
                    raise RuntimeError(f"{stage} camera traces differ between matched arms")
                summary["stages"][stage] = [{
                    "experiment_id": result["experiment_id"],
                    "seconds": result["seconds"],
                    "peak_reserved_mib": result["peak_reserved_mib"],
                    "point_cloud_sha256": result["point_cloud_sha256"],
                } for result in results]
        summary["status"] = "completed_training_pair"
    except BaseException as exc:
        summary.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(summary_path, summary)


if __name__ == "__main__":
    main()
