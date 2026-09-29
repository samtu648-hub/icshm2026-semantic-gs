#!/usr/bin/env python3
"""Run RGB139: Standard 3DGS all400/30k from the fixed-budget RGB139 init."""
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

EXPERIMENT = "RGB-139-STANDARD-RELIABLE-INIT400"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    scene = args.scene.resolve()
    output = ROOT / "outputs/quality/rgb139" / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    source = args.artifact_root / "r1/third_party/standard-3dgs"
    cuda = args.artifact_root / "r1/toolchain/cuda-12.8"
    access = ROOT / "outputs/quality/all_data/RGB-120-r2/access.json"
    config = ROOT / f"experiments/runs/{EXPERIMENT}/config.yaml"
    manifest = json.loads((scene / "initialization_manifest.json").read_text())
    if manifest["status"] != "completed_initialization" or manifest["selection_policy"]["target_points"] != 72_292:
        raise ValueError("RGB139 initialization manifest is not the registered fixed-budget candidate")

    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
        PYTHONPATH=os.pathsep.join(map(str, [source, source / "submodules/diff-gaussian-rasterization", source / "submodules/simple-knn", ROOT])),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        TORCH_HOME=str(args.artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
    )
    command = [
        sys.executable, str(ROOT / "scripts/backbone_all400_worker.py"),
        "--source", str(source), "--access", str(access),
        "--evidence", str(output / "worker.json"), "--seed", "0",
        "--expected-init-policy", "all400 DLT tracks; stratified geometry-reliability selection to RGB130 fixed budget 72292; all400 photometric training",
        "--", "--source_path", str(scene), "--model_path", str(output / "model"),
        "--resolution", "1", "--data_device", "cpu", "--iterations", "30000",
        "--test_iterations", "-1", "--save_iterations", "30000",
        "--checkpoint_iterations", "30000", "--quiet",
    ]
    result = {
        "experiment_id": EXPERIMENT, "status": "starting", "command": command, "seed": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config), "scene_manifest_sha256": sha256(scene / "initialization_manifest.json"),
        "initialization_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
        "source": source_manifest(source),
        "code_sha256": {name: sha256(ROOT / name) for name in ("scripts/backbone_all400_worker.py", "scripts/run_rgb139_reliable_init.py")},
        "free_bytes_start": shutil.disk_usage(output).free,
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
            result["gpu_processes_before"] = apps
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError("Other GPU compute process active")
            write_new_json(output / "started.json", result)
            with (output / "train.log").open("x") as log:
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                with (output / "resources.jsonl").open("x") as monitor:
                    while process.poll() is None:
                        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                        monitor.write(json.dumps({"seconds": time.monotonic() - started, "gpu": gpu.stdout.strip(), "free_bytes": shutil.disk_usage(output).free}) + "\n")
                        monitor.flush()
                        time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"Native process exited {process.returncode}")
            worker = json.loads((output / "worker.json").read_text())
            if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
                raise RuntimeError("Worker did not use the registered all400 split")
            if worker["initialization"]["common_point_count"] != 72_292:
                raise RuntimeError("Native worker did not load the fixed 72,292-point PLY")
            result.update(
                status="completed_all400", worker=worker,
                artifacts={str(path.relative_to(output)): sha256(path) for path in (output / "model").rglob("*") if path.is_file()},
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
