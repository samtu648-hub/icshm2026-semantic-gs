#!/usr/bin/env python3
"""Run RGB140: Standard all400/30k with AbsGS split-gradient eligibility."""
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
from src.quality_protocol import sha256, write_new_json
from scripts.run_quality_rgb import source_manifest

EXPERIMENT = "RGB-140-STANDARD-ABSGRAD-ALL400"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path,
                        default=Path(os.environ["ICSHM_ARTIFACT_ROOT"]) if "ICSHM_ARTIFACT_ROOT" in os.environ else None)
    parser.add_argument("--scene", type=Path, default=ROOT / "outputs/quality/all_data/RGB-120-r2/scene")
    parser.add_argument("--run-name", default="full-r2")
    parser.add_argument("--iterations", type=int, default=30_000)
    args = parser.parse_args()
    if args.artifact_root is None:
        parser.error("--artifact-root or ICSHM_ARTIFACT_ROOT is required")

    scene = args.scene.resolve()
    output = ROOT / "outputs/quality/rgb140" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "outputs/quality/rgb140/source-r1"
    cuda = args.artifact_root.resolve() / "r1/toolchain/cuda-12.8"
    python = args.artifact_root.resolve() / "r1/envs/r1-cuda128/bin/python"
    access = ROOT / "outputs/quality/all_data/RGB-120-r2/access.json"
    config = ROOT / f"experiments/runs/{EXPERIMENT}/config.yaml"
    preflight = ROOT / "outputs/quality/rgb140/preflight-r3/result.json"
    preflight_record = json.loads(preflight.read_text())
    if preflight_record["status"] != "passed":
        raise ValueError("RGB140 gradient preflight is not passed")
    if sha256(scene / "sparse/0/points3D.ply") != "10a88fb018c27530b3bbfa0f7000408693ba75228067f5311e7775e4755e68f2":
        raise ValueError("RGB130 frozen initialization PLY changed")

    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(map(str, [source, source / "submodules/diff-gaussian-rasterization",
                                             source / "submodules/simple-knn", ROOT])),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(args.artifact_root.resolve() / "r1/weights/torch"),
        WANDB_MODE="disabled", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
    )
    command = [
        str(python), str(ROOT / "scripts/backbone_all400_worker.py"),
        "--source", str(source), "--access", str(access),
        "--evidence", str(output / "worker.json"), "--seed", "0",
        "--expected-init-policy", "same RGB130 original 303-view DLT initialization; all400 photometric training",
        "--", "--source_path", str(scene), "--model_path", str(output / "model"),
        "--resolution", "1", "--data_device", "cpu", "--iterations", str(args.iterations),
        "--densify_grad_threshold", "0.0002", "--densify_grad_abs_threshold", "0.0004",
        "--test_iterations", "-1", "--save_iterations", str(args.iterations),
        "--checkpoint_iterations", str(args.iterations), "--quiet",
    ]
    result = {
        "experiment_id": EXPERIMENT, "status": "starting", "command": command, "seed": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config), "preflight_sha256": sha256(preflight),
        "initialization_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
        "source": source_manifest(source),
        "method": {"signed_clone_threshold": 0.0002, "absolute_split_threshold": 0.0004,
                   "all_other_densification_and_optimization_settings": "RGB130 native Standard 3DGS"},
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/backbone_all400_worker.py", "scripts/preflight_rgb140_absgs.py",
            "scripts/run_rgb140_absgs_all400.py")},
        "free_bytes_start": shutil.disk_usage(output).free,
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"
            ], text=True)
            result["gpu_processes_before"] = apps
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError("Other GPU compute process active")
            write_new_json(output / "started.json", result)
            with (output / "train.log").open("x") as log:
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                with (output / "resources.jsonl").open("x") as monitor:
                    while process.poll() is None:
                        gpu = subprocess.run([
                            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm",
                            "--format=csv,noheader,nounits"
                        ], capture_output=True, text=True)
                        monitor.write(json.dumps({"seconds": time.monotonic() - started,
                                                  "gpu": gpu.stdout.strip(),
                                                  "free_bytes": shutil.disk_usage(output).free}) + "\n")
                        monitor.flush()
                        time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"Native process exited {process.returncode}")
            worker = json.loads((output / "worker.json").read_text())
            if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
                raise RuntimeError("Worker did not use the registered all400 split")
            if worker["initialization"]["common_point_count"] != 72_292:
                raise RuntimeError("Native worker did not load RGB130's 72,292-point PLY")
            result.update(
                status="completed_all400", worker=worker,
                artifacts={str(path.relative_to(output)): sha256(path)
                           for path in (output / "model").rglob("*") if path.is_file()},
                free_bytes_end=shutil.disk_usage(output).free,
            )
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
