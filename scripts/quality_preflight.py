#!/usr/bin/env python3
"""Verify immutable inputs and derive whole-chain folds without opening GT pixels."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import full_chain_folds, sha256, validation_matrix, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite preflight evidence")
    configpath = ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml"
    config = json.loads(configpath.read_text())
    record = {"experiment_id": config["experiment_id"], "config_sha256": sha256(configpath),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "quality_metrics": None, "gpu_training_performed": False}
    try:
        record["inputs"] = verify_hashes(ROOT, config["input_hashes"])
        frozen = json.loads((ROOT / config["frozen_manifest"]).read_text())
        record["frozen_artifacts"] = verify_hashes(args.artifact_root, {x["path"]: x["sha256"] for x in frozen["artifacts"]})
        rgb = json.loads((ROOT / "experiments/splits/rgb_r1_spatial_v1.json").read_text())
        oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
        graph = json.loads((ROOT / "experiments/geometry_camera_graph.json").read_text())
        primary = rgb["primary_fold"]
        record["full_chain_folds"] = full_chain_folds(primary, oof["folds"], graph["nodes"], config["validation"]["guard_radius"])
        record["validation_matrix"] = validation_matrix()
        record["fold_inputs"] = {n: sha256(ROOT / n) for n in ("experiments/geometry_camera_graph.json", config["frozen_manifest"])}
        scene = args.artifact_root / config["rgb"]["scene"]
        actual_test = (scene / "sparse/0/test.txt").read_text().splitlines()
        if sorted(actual_test) != sorted(primary["validation_labeled"]):
            raise ValueError("Prepared scene evaluation list mismatch")
        actual_images = {p.name for p in (scene / "images").glob("*.png")}
        if actual_images != set(primary["train_rgb_400"]) | set(primary["validation_labeled"]):
            raise ValueError("Prepared scene image universe mismatch")
        record["prepared_scene_inputs"] = {name: sha256(scene / name) for name in
                                            ("preparation_manifest.json", "sparse/0/test.txt", "sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt")}
        record["free_gib"] = shutil.disk_usage(args.output.parent if args.output.parent.exists() else ROOT).free / 2**30
        if record["free_gib"] < config["execution"]["minimum_free_gib"]:
            raise RuntimeError("Less than configured free disk reserve")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        record["gpu"] = {"returncode": gpu.returncode, "output": (gpu.stdout + gpu.stderr).strip()}
        if gpu.returncode:
            raise RuntimeError("GPU access failed in this execution surface; do not infer driver failure")
        record["status"] = "passed_metadata_and_frozen_artifact_preflight"
    except Exception as exc:
        record.update(status="preflight_failed", error=f"{type(exc).__name__}: {exc}")
        write_new_json(args.output, record)
        raise
    write_new_json(args.output, record)
    print(json.dumps({"status": record["status"], "frozen_count": len(record["frozen_artifacts"]),
                      "fold_counts": [{"fold": f["fold"], "rgb_train": len(f["rgb_train"]), "semantic_train": len(f["semantic_train"]), "evaluation": len(f["evaluation"]), "rgb_guard": len(f["rgb_guard"])} for f in record["full_chain_folds"]],
                      "gpu": record["gpu"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
