#!/usr/bin/env python3
"""Independently audit QUAL026 roles, core RGB metrics, bridge regions, and artifacts."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json

ARMS = ("RGB175_1M", "RGB176_15M")
CLASSES = {1: "deck", 2: "stay_cable", 3: "tower", 4: "foundation"}


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean(np.square(reference - prediction), dtype=np.float64))
    return -10.0 * math.log10(mse)


def vertices(path: Path) -> int:
    with path.open("rb") as handle:
        for line in handle:
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                break
    raise ValueError(path)


def main() -> None:
    root = ROOT / "outputs/quality/QUAL-026/r1"
    output = root / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    access = json.loads((ROOT / "outputs/quality/RGB-152/access.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "experiments/splits/spatial_confirmation_v1.json").read_text(encoding="utf-8"))
    roles = {key: set(access[key]) for key in ("train", "evaluation", "buffer")}
    checks = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    require(tuple(map(len, (roles["train"], roles["evaluation"], roles["buffer"]))) == (353, 30, 17),
            "role counts 353/30/17")
    require(not (roles["train"] & roles["evaluation"] or roles["train"] & roles["buffer"] or
                 roles["evaluation"] & roles["buffer"]), "roles disjoint")
    require(len(set().union(*roles.values())) == 400, "all400 accounted")
    require(roles["evaluation"] == set(split["primary_fold"]["validation_labeled"]), "holdout identity")
    for key in ("RGB-175", "RGB-176"):
        worker = json.loads((ROOT / f"outputs/quality/{key}/full-r1/worker.json").read_text(encoding="utf-8"))
        require(set(worker["actual_train"]) == roles["train"], f"{key} train role")
        require(set(worker["actual_evaluation"]) == roles["evaluation"], f"{key} holdout role")
        require(worker["initialization"]["common_point_count"] == 78045, f"{key} training-only initialization")
    worker175 = json.loads((ROOT / "outputs/quality/RGB-175/full-r1/worker.json").read_text())
    worker176 = json.loads((ROOT / "outputs/quality/RGB-176/full-r1/worker.json").read_text())
    require(worker175["sample_trace"] == worker176["sample_trace"], "matched camera trace")

    csv_rows = {row["view"]: row for row in csv.DictReader((root / "per_view.csv").open())}
    require(len(csv_rows) == 400, "per-view row count")
    scene = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
    image_roots = {arm: root / "images" / arm for arm in ARMS}
    by_role = {role: {arm: [] for arm in ARMS} for role in ("train", "holdout", "buffer")}
    role_for = {name: "train" for name in roles["train"]} | {name: "holdout" for name in roles["evaluation"]} | {
        name: "buffer" for name in roles["buffer"]
    }
    for name in sorted(csv_rows):
        reference = load(scene / name)
        role = role_for[name]
        for arm in ARMS:
            value = psnr(reference, load(image_roots[arm] / name))
            require(abs(value - float(csv_rows[name][f"{arm}_PSNR"])) < 5e-7, f"PSNR {arm} {name}")
            by_role[role][arm].append(value)
    for role in by_role:
        for arm in ARMS:
            expected = result["partitions"][role][arm]["PSNR"]
            require(abs(float(np.mean(by_role[role][arm])) - expected) < 5e-7, f"aggregate PSNR {role} {arm}")

    labels = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
    bridge = {arm: {class_name: [] for class_name in CLASSES.values()} for arm in ARMS}
    weights = {class_name: [] for class_name in CLASSES.values()}
    for name in sorted(roles["evaluation"]):
        semantic = np.asarray(Image.open(labels / name))
        reference = load(scene / name)
        for class_id, class_name in CLASSES.items():
            mask = semantic == class_id
            if not mask.any():
                continue
            weights[class_name].append(float(mask.sum()))
            for arm in ARMS:
                bridge[arm][class_name].append(psnr(reference[mask], load(image_roots[arm] / name)[mask]))
    for arm in ARMS:
        for class_name in CLASSES.values():
            value = float(np.average(bridge[arm][class_name], weights=weights[class_name]))
            expected = result["bridge_structure"]["holdout"][arm][class_name]["region_psnr"]
            require(abs(value - expected) < 2e-6, f"holdout bridge PSNR {arm} {class_name}")

    for arm, key in (("RGB175_1M", "RGB-175"), ("RGB176_15M", "RGB-176")):
        ply = ROOT / f"outputs/quality/{key}/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"
        require(vertices(ply) == result["resources"][arm]["gaussians"], f"vertex count {arm}")
        require(sha256(ply) == result["resources"][arm]["ply_sha256"], f"PLY hash {arm}")
    require(result["code_sha256"]["evaluator"] == sha256(ROOT / "scripts/evaluate_rgb175_rgb176_spatial_pair.py"),
            "evaluator hash")
    trajectory = json.loads((root / "trajectory/result.json").read_text(encoding="utf-8"))
    require(trajectory["code_sha256"] == sha256(ROOT / "scripts/render_rgb175_rgb176_trajectory.py"),
            "trajectory code hash")
    require(len(list((root / "trajectory/comparison_frames").glob("frame*.png"))) == 144,
            "trajectory comparison frame count")
    video = ROOT / trajectory["video"]
    require(sha256(video) == trajectory["video_sha256"], "trajectory video hash")
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,width,height,codec_name",
        "-of", "json", str(video),
    ], text=True)
    stream = json.loads(probe)["streams"][0]
    require(stream["codec_name"] == "h264" and int(stream["nb_frames"]) == 144, "trajectory video decodes")
    write_new_json(output, {
        "schema_version": 1, "status": "passed", "checks": len(checks), "check_labels": checks,
        "result_sha256": sha256(root / "result.json"), "trajectory_result_sha256": sha256(root / "trajectory/result.json"),
        "video_stream": stream,
    })


if __name__ == "__main__":
    main()
