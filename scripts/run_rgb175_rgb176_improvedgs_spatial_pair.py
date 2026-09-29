#!/usr/bin/env python3
"""Run the matched 1.0M/1.5M ImprovedGS spatial-confirmation pair."""
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

SOURCE_COMMIT = "20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a"
SCENE_ROOT = ROOT / "outputs/quality/RGB-152"
SCENE = SCENE_ROOT / "scene"
ACCESS = SCENE_ROOT / "access.json"
SPLIT = ROOT / "experiments/splits/spatial_confirmation_v1.json"
ENDPOINTS = (15_000, 20_000, 30_000, 40_000, 50_000)
ARMS = (
    ("RGB-175-IMPROVEDGS-SPATIAL-BUDGET10M", "RGB-175", 1_000_000),
    ("RGB-176-IMPROVEDGS-SPATIAL-BUDGET15M", "RGB-176", 1_500_000),
)


def count_vertices(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline().decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                raise ValueError(f"PLY vertex count missing: {path}")


def run_arm(*, artifact_root: Path, stage: str, attempt: str, eid: str, key: str, budget: int) -> dict:
    source = ROOT / SOURCE_REL
    config = ROOT / "experiments/runs" / eid / "config.yaml"
    output = ROOT / "outputs/quality" / key / attempt
    output.mkdir(parents=True, exist_ok=False)
    iterations = 901 if stage == "preflight" else 50_000
    saves = [iterations] if stage == "preflight" else list(ENDPOINTS)
    python = artifact_root / "r1/envs/r1-cuda128/bin/python"
    command = [
        str(python), str(ROOT / "scripts/improvedgs_explicit_split_worker.py"),
        "--source", str(source), "--access", str(ACCESS), "--evidence", str(output / "worker.json"),
        "--seed", "0", "--partition", "spatial_confirmation_v1",
        "--expected-train", "353", "--expected-evaluation", "30", "--expected-universe", "383",
        "--expected-points", "78045", "--",
        "--source_path", str(SCENE), "--model_path", str(output / "model"),
        "--resolution", "1", "--data_device", "cpu", "--eval", "true",
        "--iterations", str(iterations), "--training_method", "improvedgs",
        "--use_las", "true", "--use_eas", "true", "--use_rap", "true", "--use_mu", "true",
        "--budget", str(budget), "--edge_sample_cams", "10", "--densify_until_iter", "15000",
        "--test_iterations", "-1", "--save_iterations", *map(str, saves),
        "--empty_cache_interval", "200", "--quiet",
    ]
    protected = {
        "split": sha256(SPLIT), "access": sha256(ACCESS),
        "initialization": sha256(SCENE / "sparse/0/points3D.ply"),
        "rgb170": sha256(ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
    }
    record: dict[str, object] = {
        "schema_version": 1, "experiment_id": eid, "stage": stage, "status": "starting",
        "budget": budget, "iterations": iterations, "command": command, "seed": 0,
        "config_sha256": sha256(config), "split_sha256": protected["split"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": source_record(source), "protected_before": protected,
        "code_sha256": {
            "worker": sha256(ROOT / "scripts/improvedgs_explicit_split_worker.py"),
            "runner": sha256(ROOT / "scripts/run_rgb175_rgb176_improvedgs_spatial_pair.py"),
        },
        "free_bytes_start": shutil.disk_usage(output).free,
    }
    started = time.monotonic()
    try:
        write_new_json(output / "started.json", record)
        with (output / "train.log").open("x") as log:
            process = subprocess.Popen(command, cwd=source, env=build_env(artifact_root, source),
                                       stdout=log, stderr=subprocess.STDOUT)
            with (output / "resources.jsonl").open("x") as monitor:
                while process.poll() is None:
                    gpu = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True,
                    )
                    monitor.write(json.dumps({"seconds": time.monotonic() - started, "gpu": gpu.stdout.strip(),
                                              "free_bytes": shutil.disk_usage(output).free}) + "\n")
                    monitor.flush()
                    time.sleep(10)
        if process.returncode:
            raise RuntimeError(f"training exited {process.returncode}; see {output / 'train.log'}")
        worker = json.loads((output / "worker.json").read_text(encoding="utf-8"))
        if worker.get("status") != "completed_native_execution":
            raise RuntimeError("native worker did not complete")
        access = json.loads(ACCESS.read_text(encoding="utf-8"))
        if worker["actual_train"] != sorted(access["train"]) or worker["actual_evaluation"] != sorted(access["evaluation"]):
            raise RuntimeError("native worker did not preserve the registered split")
        endpoints = {}
        for iteration in saves:
            ply = output / f"model/point_cloud/iteration_{iteration}/point_cloud.ply"
            if not ply.is_file():
                raise FileNotFoundError(ply)
            endpoints[str(iteration)] = {"sha256": sha256(ply), "bytes": ply.stat().st_size,
                                         "gaussians": count_vertices(ply)}
        protected_after = {
            "split": sha256(SPLIT), "access": sha256(ACCESS),
            "initialization": sha256(SCENE / "sparse/0/points3D.ply"),
            "rgb170": sha256(ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"),
        }
        if protected_after != protected:
            raise RuntimeError("protected split, initialization, or RGB170 changed")
        record.update(status="completed", worker=worker, endpoints=endpoints, protected_exact=True)
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
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
    if source_record(source)["commit"] != SOURCE_COMMIT:
        raise ValueError("ImprovedGS source commit changed")
    if sha256(SCENE / "sparse/0/points3D.ply") != "b7bddeda3759a11e833bd3d60ee554b8e005688ab14f9d41b99d99e2ac59521a":
        raise ValueError("Spatial initialization changed")
    stages = ["preflight", "full"] if args.stage == "all" else [args.stage]
    summary_path = ROOT / f"outputs/quality/RGB-175-RGB-176-spatial-pair-summary-{args.pair_attempt}.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary: dict[str, object] = {"schema_version": 1, "status": "starting", "stages": {}}
    try:
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True,
            )
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError(f"Other GPU compute process active:\n{apps}")
            for stage in stages:
                attempt = args.preflight_attempt if stage == "preflight" else args.full_attempt
                results = [run_arm(artifact_root=args.artifact_root, stage=stage, attempt=attempt,
                                   eid=eid, key=key, budget=budget) for eid, key, budget in ARMS]
                if results[0]["worker"]["sample_trace"] != results[1]["worker"]["sample_trace"]:
                    raise RuntimeError(f"{stage} camera traces differ between matched arms")
                summary["stages"][stage] = [{
                    "experiment_id": result["experiment_id"], "budget": result["budget"],
                    "wall_seconds": result["wall_seconds"], "peak_reserved_mib": result["worker"]["peak_reserved_mib"],
                    "endpoints": result["endpoints"],
                } for result in results]
        summary["status"] = "completed_training_pair"
    except BaseException as exc:
        summary.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_new_json(summary_path, summary)


if __name__ == "__main__":
    main()
