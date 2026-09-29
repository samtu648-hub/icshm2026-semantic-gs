#!/usr/bin/env python3
"""Replay RGB130 and preserve lifecycle checkpoints around native densification."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    output = ROOT / "outputs/quality/RGB-163" / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    source = args.artifact_root / "r1/third_party/standard-3dgs"
    scene_root = ROOT / "outputs/quality/all_data/RGB-120-r2"
    scene = scene_root / "scene"
    config = ROOT / "experiments/runs/RGB-163-STANDARD-DENSIFICATION-LIFECYCLE/config.yaml"
    payload = json.loads(config.read_text())
    cuda = args.artifact_root / "r1/toolchain/cuda-12.8"
    python = args.artifact_root / "r1/envs/r1-cuda128/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    env = os.environ.copy()
    env.update(CUDA_HOME=str(cuda), PATH=str(cuda / "bin") + os.pathsep + env.get("PATH", ""),
               LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + env.get("LD_LIBRARY_PATH", ""),
               PYTHONPATH=os.pathsep.join(map(str, [source, source / "submodules/diff-gaussian-rasterization",
                                                    source / "submodules/simple-knn", ROOT])),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TORCH_HOME=str(args.artifact_root / "r1/weights/torch"), WANDB_MODE="disabled",
               HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    saves = [str(value) for value in payload["save_iterations"]]
    command = [str(python), str(ROOT / "scripts/backbone_all400_worker.py"),
               "--source", str(source), "--access", str(scene_root / "access.json"),
               "--evidence", str(output / "worker.json"), "--seed", "0", "--",
               "--source_path", str(scene), "--model_path", str(output / "model"),
               "--resolution", "1", "--data_device", "cpu", "--iterations", "30000",
               "--test_iterations", "-1", "--save_iterations", *saves, "--quiet"]
    evaluation = [str(python), str(ROOT / "scripts/evaluate_rgb163_lifecycle.py"),
                  "--source", str(source), "--model", str(output / "model"),
                  "--scene", str(scene), "--output", str(output / "evaluation"),
                  "--config", str(config)]
    result = {"schema_version": 1, "experiment_id": payload["experiment_id"], "status": "starting",
              "config_sha256": sha256(config), "source": source_manifest(source),
              "command": command, "evaluation_command": evaluation, "started_unix": time.time(),
              "free_bytes_start": shutil.disk_usage(output).free}
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                            "--format=csv,noheader"], text=True)
            result["gpu_processes_before"] = apps
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError("Other GPU compute process active")
            write_new_json(output / "started.json", result)
            for stage, stage_command in (("train", command), ("evaluate", evaluation)):
                result["status"] = f"running_{stage}"
                with (output / f"{stage}.log").open("x") as log:
                    process = subprocess.Popen(stage_command, cwd=source if stage == "train" else ROOT,
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
                    with (output / f"{stage}-resources.jsonl").open("x") as monitor:
                        while process.poll() is None:
                            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
                                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
                            monitor.write(json.dumps({"seconds": time.monotonic() - started,
                                                      "gpu": gpu.stdout.strip(),
                                                      "free_bytes": shutil.disk_usage(output).free}) + "\n")
                            monitor.flush()
                            time.sleep(10)
                if process.returncode:
                    raise RuntimeError(f"{stage} exited {process.returncode}; see {stage}.log")
            worker = json.loads((output / "worker.json").read_text())
            evaluation_result = json.loads((output / "evaluation/result.json").read_text())
            if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
                raise ValueError("RGB163 did not use the fixed all400 split")
            result.update(status="completed", worker_status=worker["status"],
                          checkpoint_summaries=evaluation_result["checkpoint_summaries"],
                          rgb130_final_exact=evaluation_result["rgb130_final_exact"])
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        result["free_bytes_end"] = shutil.disk_usage(output).free
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
