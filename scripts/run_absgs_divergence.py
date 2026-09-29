#!/usr/bin/env python3
"""Run the fixed paired native AbsGS trajectory diagnostic and offline audits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_rgb152_spatial as base
from run_component_drop_extensions import gpu_lock
from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "QUAL-018-ABSGS-DIVERGENCE-DIAG"
CONFIG = ROOT / f"experiments/runs/{EID}/config.yaml"
CODE = [
    "scripts/run_absgs_divergence.py",
    "scripts/worker_absgs_trajectory.py",
    "scripts/analyze_absgs_divergence.py",
    "src/absgs_trajectory.py",
    "scripts/backbone_explicit_split_worker.py",
    "scripts/run_rgb152_spatial.py",
    "src/quality_native_gaussian.py",
    "src/quality_cameras.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def merge(a, b):
    r = dict(a)
    for k, v in b.items():
        r[k] = (
            merge(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else v
        )
    return r


def suite():
    return read(CONFIG)


def output():
    return ROOT / suite()["output_path"]


def arm_config(eid):
    c = read(ROOT / f"experiments/runs/{eid}/config.yaml")
    return merge(read(ROOT / c["base_config"]), c)


def guard():
    verify_hashes(ROOT, read(output() / "binding.json")["files"])


def prepare():
    c = suite()
    o = output()
    o.mkdir(parents=True, exist_ok=False)
    paths = CODE + [
        str(CONFIG.relative_to(ROOT)),
        c["protocol"],
        c["split_path"],
        c["dataset_manifest"],
    ]
    for eid in c["arms"]:
        paths += [f"experiments/runs/{eid}/config.yaml", arm_config(eid)["base_config"]]
    for name in ["cameras.txt", "images.txt", "test.txt", "points3D.ply"]:
        paths.append(c["scene"] + "/sparse/0/" + name)
    for rel in (
        "RGB-152/full-r2",
        "RGB-156/full-r1",
        "RGB-158/full-r1",
        "rgb140/full-r2",
    ):
        paths.append(f"outputs/quality/{rel}/result.json")
        parent = read(ROOT / paths[-1])
        verify_hashes(ROOT / paths[-1].rsplit("/", 1)[0], parent["artifacts"])
    files = {p: sha256(ROOT / p) for p in sorted(set(paths))}
    for p in CODE + [str(CONFIG.relative_to(ROOT)), c["protocol"]]:
        dest = o / "source" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, dest)
    scene = ROOT / c["scene"]
    split = read(ROOT / c["split_path"])
    names = sorted(set(split["primary_fold"]["train_rgb_400"]) | set(c["probe_views"]))
    image_hashes = {n: sha256(scene / "images" / n) for n in names}
    write_new_json(o / "input-image-hashes.json", image_hashes)
    write_new_json(
        o / "binding.json",
        {
            "files": files,
            "source": base.source_manifest(ROOT / c["source"]),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "initial_ply_sha256": sha256(scene / "sparse/0/points3D.ply"),
            "created_unix": time.time(),
            "free_bytes": shutil.disk_usage(o).free,
        },
    )


def train(eid, artifact_root, smoke=False):
    guard()
    c = arm_config(eid)
    q = suite()
    source = ROOT / c["output"]["source"]
    base.EID, base.CONFIG = eid, ROOT / f"experiments/runs/{eid}/config.yaml"
    base.CODE_PATHS = tuple(sorted(set(base.CODE_PATHS) | set(CODE)))
    if not (ROOT / c["output"]["root"] / "preflight-r3/result.json").exists():
        base.preflight(c, artifact_root)
    run = ROOT / c["output"]["root"] / ("trace-smoke-701" if smoke else "full-r1")
    run.mkdir(parents=True, exist_ok=False)
    steps = 701 if smoke else 30000
    env, python = base.runtime_environment(artifact_root, source)
    cmd = [
        str(python),
        str(ROOT / "scripts/worker_absgs_trajectory.py"),
        "--trace-config",
        str(CONFIG),
        "--trace-output",
        str(run / "trace"),
        "--",
        "--source",
        str(source),
        "--access",
        str(ROOT / c["output"]["root"] / "access.json"),
        "--evidence",
        str(run / "worker.json"),
        "--seed",
        "0",
        "--partition",
        c["split_id"],
        "--expected-train",
        "353",
        "--expected-evaluation",
        "30",
        "--expected-universe",
        "383",
        "--",
        "--source_path",
        str(ROOT / c["data"]["scene"]),
        "--model_path",
        str(run / "model"),
        "--resolution",
        "1",
        "--data_device",
        "cpu",
        "--iterations",
        str(steps),
        "--densify_grad_threshold",
        "0.0002",
        "--densify_grad_abs_threshold",
        "0.0004",
        "--test_iterations",
        "-1",
        "--save_iterations",
        str(steps),
        "--checkpoint_iterations",
        str(steps),
        "--eval",
        "--quiet",
    ]
    result = {
        "status": "starting",
        "experiment_id": eid,
        "command": cmd,
        "seed": 0,
        "git_commit": c["git_commit"],
        "config_sha256": sha256(base.CONFIG),
        "split_sha256": sha256(ROOT / c["split_path"]),
        "source": base.source_manifest(source),
        "initialization_ply_sha256": sha256(
            ROOT / c["data"]["scene"] / "sparse/0/points3D.ply"
        ),
        "environment": {
            k: env.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "PYTHONHASHSEED",
            )
        },
    }
    start = time.monotonic()
    try:
        with gpu_lock():
            with (
                (run / "train.log").open("x") as log,
                (run / "resources.jsonl").open("x") as resources,
            ):
                p = subprocess.Popen(
                    cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                while p.poll() is None:
                    g = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    resources.write(
                        json.dumps(
                            {
                                "seconds": time.monotonic() - start,
                                "gpu": g.stdout.strip(),
                            }
                        )
                        + "\n"
                    )
                    resources.flush()
                    time.sleep(10)
            result["exit_status"] = p.returncode
            if p.returncode:
                raise RuntimeError(f"Native trajectory worker exit {p.returncode}")
        worker = read(run / "worker.json")
        if worker["status"] != "completed_native_execution":
            raise ValueError("Incomplete native run")
        cameras = read(run / "trace/camera-trace.json")
        if len(cameras) != steps or [r["step"] for r in cameras] != list(
            range(1, steps + 1)
        ):
            raise ValueError("Invalid camera trace")
        guard()
        verify_hashes(
            ROOT / c["data"]["scene"] / "images",
            read(output() / "input-image-hashes.json"),
        )
        result.update(
            status="passed_smoke" if smoke else "completed_spatial_training",
            worker=worker,
            artifacts={
                str(p.relative_to(run)): sha256(p)
                for p in (run / "model").rglob("*")
                if p.is_file()
            },
            trace_hashes={
                str(p.relative_to(run)): sha256(p)
                for p in (run / "trace").rglob("*")
                if p.is_file()
            },
        )
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write_new_json(run / "result.json", result)


def evaluate(eid, artifact_root):
    guard()
    c = arm_config(eid)
    for other in suite()["arms"]:
        if (
            read(ROOT / arm_config(other)["training"]["output"] / "result.json")[
                "status"
            ]
            != "completed_spatial_training"
        ):
            raise ValueError("Both runs must finish")
    base.EID, base.CONFIG = eid, ROOT / f"experiments/runs/{eid}/config.yaml"
    base.CODE_PATHS = tuple(sorted(set(base.CODE_PATHS) | set(CODE)))
    base.render_and_evaluate(c, artifact_root)


def run(artifact_root):
    prepare()
    result = {"status": "running", "stages": []}
    start = time.monotonic()

    def execute(name, stage, eid=None):
        command = [
            sys.executable,
            str(Path(__file__)),
            "--artifact-root",
            str(artifact_root),
            "--stage",
            stage,
        ]
        if eid:
            command += ["--experiment", eid]
        t = time.monotonic()
        with (output() / f"{name}.log").open("x") as log:
            p = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        row = {
            "name": name,
            "command": command,
            "seconds": time.monotonic() - t,
            "exit_status": p.returncode,
        }
        write_new_json(output() / f"{name}-execution.json", row)
        result["stages"].append(row)
        print(json.dumps(row), flush=True)
        if p.returncode:
            raise RuntimeError(f"Stage {name} failed and was preserved")

    try:
        execute("smoke", "smoke", suite()["arms"][0])
        for eid in suite()["arms"]:
            execute(eid + "-train", "train", eid)
        for eid in suite()["arms"]:
            execute(eid + "-evaluate", "evaluate", eid)
        execute("analyze", "analyze")
        result["status"] = "completed"
    except BaseException as e:
        result.update(status="failed_preserved", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        result["seconds"] = time.monotonic() - start
        write_new_json(output() / "execution-result.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--experiment")
    a = p.parse_args()
    root = a.artifact_root.resolve()
    if a.stage == "run":
        run(root)
    elif a.stage in ("train", "smoke"):
        train(a.experiment, root, a.stage == "smoke")
    elif a.stage == "evaluate":
        evaluate(a.experiment, root)
    elif a.stage == "analyze":
        from analyze_absgs_divergence import analyze

        analyze(root)
    else:
        raise ValueError("Unknown diagnostic stage")


if __name__ == "__main__":
    main()
