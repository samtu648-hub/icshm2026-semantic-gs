#!/usr/bin/env python3
"""Two fresh native rendering reloads, isolated from original training outputs."""
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
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_rgb import source_manifest


def organize_named_renders(source, output, names):
    """PGSR names outputs by COLMAP image, not by an ordinal sequence."""
    names = list(names)
    files = {p.name: p for p in source.glob("*.png")}
    if len(names) != len(set(names)) or set(files) != set(names):
        raise ValueError("Named render set differs from explicit evaluation cameras")
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copy2(files[name], output / name)
    return {"views": [{"view": n, "source": n, "destination": n, "method": "copy"} for n in names]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--cuda", type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    args = p.parse_args()
    run, source, output = args.run.resolve(), args.source.resolve(), args.output.resolve()
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "completed_native_training_pending_independent_reload_and_evaluation":
        raise ValueError("Training has not completed successfully")
    if source_manifest(source) != json.loads((run / "source_manifest.json").read_text()):
        raise ValueError("Different source/extension bytes than training")
    expected_extras = result.get("extra_sources", [])
    if len(expected_extras) != len(args.extra_pythonpath):
        raise ValueError("Different supplemental dependency paths than training")
    for entry, extra in zip(expected_extras, args.extra_pythonpath):
        if source_manifest(extra.resolve()) != json.loads((run / entry["manifest"]).read_text()):
            raise ValueError("Supplemental dependency bytes changed")
    verify_hashes(run, result["artifacts"])
    quality_root = run.parent.parent
    with (quality_root / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        record = {"experiment_id": result["experiment_id"], "mode": result["mode"], "seed": result["seed"],
                  "run_result_sha256": sha256(run / "result.json"), "status": "running", "quality_metrics": None,
                  "config_sha256": result["config_sha256"], "source_manifest_sha256": result["source_manifest_sha256"]}
        record["audit_code_sha256"] = {n: sha256(ROOT / n) for n in
            ("scripts/audit_quality_rgb.py", "scripts/evaluate_rgb.py", "scripts/organize_r1_renders.py", "src/quality_protocol.py")}
        started = time.monotonic()
        try:
            protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
            if sha256(ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml") != result["protocol_sha256"]:
                raise ValueError("Registered protocol changed since training")
            verify_hashes(run / "code", result["code_sha256"])
            if shutil.disk_usage(output).free < protocol["execution"]["minimum_free_gib"] * 2**30:
                raise RuntimeError("Insufficient free disk reserve")
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
            verify_hashes(args.artifact_root, hashes)
            config = json.loads((ROOT / "experiments/runs" / result["experiment_id"] / "config.yaml").read_text())
            if sha256(ROOT / "experiments/runs" / result["experiment_id"] / "config.yaml") != record["config_sha256"]:
                raise ValueError("Registered config changed since training")
            steps = protocol["rgb"]["smoke_iterations" if result["mode"] == "smoke" else "iterations"]
            extension = "diff-plane-rasterization" if config["backbone"] == "pgsr" else "diff-gaussian-rasterization"
            paths = [source / "submodules" / extension, source / "submodules/simple-knn", source, ROOT, *[v.resolve() for v in args.extra_pythonpath]]
            env = dict(os.environ, CUDA_HOME=str(args.cuda), PYTHONPATH=os.pathsep.join(map(str, paths)),
                       TORCH_HOME=str(args.artifact_root / "r1/weights/torch"), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4")
            python = args.python.absolute()
            rendered = []
            for index in range(2):
                replay = output / f"reload{index + 1}"
                model = replay / "model"
                model.mkdir(parents=True)
                for name in ("cfg_args", "config.json", "mesh_args", "rem_args"):
                    if (run / "model" / name).is_file():
                        shutil.copy2(run / "model" / name, model / name)
                shutil.copytree(run / "model/point_cloud" / f"iteration_{steps}", model / "point_cloud" / f"iteration_{steps}")
                command = [str(python), str(run / "code/scripts/quality_upstream_worker.py"), "--source", str(source), "--access", str(run / "access.json"),
                           "--evidence", str(replay / "worker.json"), "--script", "render.py", "--seed", str(result["seed"]), "--",
                           "--source_path", str(run / "scene"), "--model_path", str(model), "--iteration", str(steps), "--skip_train", "--quiet"]
                with (replay / "render.log").open("x") as log:
                    subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                directory = model / "test" / f"ours_{steps}" / "renders"
                files = sorted(directory.glob("*.png"))
                if len(files) != 74:
                    raise ValueError("Native replay must contain exactly all 74 developer cameras")
                rendered.append({f.name: sha256(f) for f in files})
            record["fresh_reloads_exact"] = rendered[0] == rendered[1]
            if not record["fresh_reloads_exact"]:
                raise ValueError("Fresh process render bytes differ")
            write_new_json(output / "render_hashes.json", rendered[0])
            native = output / "reload1/model/test" / f"ours_{steps}" / "renders"
            command = [str(python), str(ROOT / "scripts/organize_r1_renders.py"), "--split", str(ROOT / "experiments/splits/rgb_r1_spatial_v1.json"),
                       "--source-dir", str(native), "--output-dir", str(output / "rgb"), "--mapping-json", str(output / "mapping.json")]
            with (output / "evaluation.log").open("x") as log:
                if config["backbone"] == "pgsr":
                    access = json.loads((run / "access.json").read_text())
                    mapping = organize_named_renders(native, output / "rgb", access["evaluation"])
                    write_new_json(output / "mapping.json", mapping)
                else:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if result["mode"] == "full":
                    command = [str(python), str(ROOT / "scripts/evaluate_rgb.py"), "--split", str(ROOT / "experiments/splits/rgb_r1_spatial_v1.json"),
                               "--reference-dir", str(run / "scene/images"), "--prediction-dir", str(output / "rgb"),
                               "--per-view-csv", str(output / "metrics.csv"), "--summary-json", str(output / "metrics.json"), "--lpips-device", "cuda"]
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    record["quality_metrics"] = json.loads((output / "metrics.json").read_text())
            verify_hashes(args.artifact_root, hashes)
            verify_hashes(run, result["artifacts"])
            record.update(status="completed_rgb_reload_audit", fresh_processes=2, cameras=74, frozen_verified=len(hashes),
                          semantic_validation_completed=False, external_camera_validation_completed=False)
        except BaseException as exc:
            record.update(status="audit_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output), "quality_scored": record["quality_metrics"] is not None}), flush=True)


if __name__ == "__main__":
    main()
