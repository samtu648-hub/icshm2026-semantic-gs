#!/usr/bin/env python3
"""Run the RGB166 recipe from scratch with the Gaussian budget raised to 1.5M."""
from __future__ import annotations

from _artifact_paths import artifact_root_default, parse_artifact_root

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_rgb164_rgb165_improvedgs_pair import SOURCE_REL, build_env, source_record
from src.quality_protocol import sha256, write_new_json

EXPERIMENT_ID = "RGB-170-IMPROVEDGS-BUDGET15M-H50K"
CONFIG = ROOT / "experiments/runs" / EXPERIMENT_ID / "config.yaml"
OUTPUT = ROOT / "outputs/quality/RGB-170"
ENDPOINTS = (15_000, 20_000, 30_000, 40_000, 50_000)


def gpu_query() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True, capture_output=True,
    )


def wait_for_gpu(queue_path: Path, poll_seconds: int) -> None:
    started = time.monotonic()
    while True:
        probe = gpu_query()
        if probe.returncode == 0:
            return
        queue_path.write_text(json.dumps({
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "waiting_for_gpu_driver",
            "wait_seconds": time.monotonic() - started,
            "last_error": probe.stderr.strip(),
        }, indent=2, allow_nan=False) + "\n")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=artifact_root_default())
    parser.add_argument("--attempt", default="full-r1")
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    args = parser.parse_args()

    cfg = json.loads(CONFIG.read_text())
    source = ROOT / SOURCE_REL
    if source_record(source)["commit"] != "20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a":
        raise ValueError("ImprovedGS source commit changed")
    run = OUTPUT / args.attempt
    run.mkdir(parents=True, exist_ok=False)
    queue_path = run / "queue.json"
    write_new_json(queue_path, {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "status": "queued", "created_unix": time.time(),
    })
    wait_for_gpu(queue_path, args.gpu_poll_seconds)

    scene_root = ROOT / "outputs/quality/all_data/RGB-120-r2"
    python = args.artifact_root / "r1/envs/r1-cuda128/bin/python"
    command = [
        str(python), str(ROOT / "scripts/improvedgs_all400_worker.py"),
        "--source", str(source), "--access", str(scene_root / "access.json"),
        "--evidence", str(run / "worker.json"), "--seed", "0", "--",
        "--source_path", str(scene_root / "scene"), "--model_path", str(run / "model"),
        "--resolution", "1", "--data_device", "cpu", "--eval", "false",
        "--iterations", "50000", "--training_method", "improvedgs",
        "--use_las", "true", "--use_eas", "true", "--use_rap", "true", "--use_mu", "true",
        "--budget", "1500000", "--edge_sample_cams", "10",
        "--densify_until_iter", "15000", "--test_iterations", "-1",
        "--save_iterations", *[str(value) for value in ENDPOINTS],
        "--empty_cache_interval", "200", "--quiet",
    ]
    protected = {
        "rgb166_summary": sha256(ROOT / "outputs/quality/RGB-166/full-r1/summary-r1.json"),
        "rgb166_50k": sha256(ROOT / "outputs/quality/RGB-166/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
        "split": sha256(scene_root / "split.json"),
        "initialization": sha256(scene_root / "scene/sparse/0/points3D.ply"),
    }
    record: dict[str, object] = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "waiting_lock",
        "single_primary_variable": cfg["training"]["single_primary_variable"],
        "command": command, "config_sha256": sha256(CONFIG), "protected_before": protected,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": source_record(source), "seed": 0, "from_scratch": True,
        "start_checkpoint": None, "save_iterations": list(ENDPOINTS),
    }
    started = time.monotonic()
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                probe = gpu_query()
                if probe.returncode != 0:
                    time.sleep(args.gpu_poll_seconds)
                    continue
                active = [line for line in probe.stdout.splitlines() if line.strip() and "ToDesk" not in line]
                if not active:
                    break
                time.sleep(args.gpu_poll_seconds)
            record.update(status="running_from_scratch_50k", lock_wait_seconds=time.monotonic() - started)
            write_new_json(run / "started.json", record)
            with (run / "train.log").open("x") as log:
                subprocess.run(command, cwd=source, env=build_env(args.artifact_root, source),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        worker = json.loads((run / "worker.json").read_text())
        if worker.get("status") != "completed_native_execution":
            raise RuntimeError("Native ImprovedGS worker did not complete")
        if len(worker["actual_train"]) != 400 or worker["actual_evaluation"]:
            raise RuntimeError("RGB170 did not preserve the registered all400 split")
        endpoints = {}
        for iteration in ENDPOINTS:
            ply = run / f"model/point_cloud/iteration_{iteration}/point_cloud.ply"
            if not ply.is_file():
                raise FileNotFoundError(ply)
            endpoints[str(iteration)] = {"sha256": sha256(ply), "bytes": ply.stat().st_size}
        protected_after = {
            "rgb166_summary": sha256(ROOT / "outputs/quality/RGB-166/full-r1/summary-r1.json"),
            "rgb166_50k": sha256(ROOT / "outputs/quality/RGB-166/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
            "split": sha256(scene_root / "split.json"),
            "initialization": sha256(scene_root / "scene/sparse/0/points3D.ply"),
        }
        if protected_after != protected:
            raise RuntimeError("Protected RGB166 or data artifacts changed")
        record.update(
            status="completed_training", worker=worker, endpoints=endpoints,
            training_seconds=worker["seconds"], peak_reserved_mib=worker["peak_reserved_mib"],
            protected_exact=True,
        )
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["free_bytes"] = shutil.disk_usage(run).free
        write_new_json(run / "result.json", record)


if __name__ == "__main__":
    main()
