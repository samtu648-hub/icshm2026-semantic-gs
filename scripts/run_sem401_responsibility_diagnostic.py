#!/usr/bin/env python3
"""Run the sealed, CPU-only SEM-401 child-responsibility diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/runs/SEM-401-RGB152-RESPONSIBILITY-IDENTIFIABILITY-DIAG/config.yaml"
sys.path.insert(0, str(ROOT))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = np.quantile(values, levels)
    return {str(level): float(value) for level, value in zip(levels, result)}


def read_colmap_poses(path: Path, quaternion_converter) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read only COLMAP image pose rows; point-observation rows are irrelevant here."""
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) < 10 or not fields[9].lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        name = Path(fields[9]).name
        quaternion = np.asarray([[float(value) for value in fields[1:5]]], dtype=np.float64)
        rotation = quaternion_converter(quaternion)[0]
        translation = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        if name in poses:
            raise ValueError(f"Duplicate COLMAP image name: {name}")
        poses[name] = (rotation, translation)
    return poses


def run(config: dict, output: Path, result: dict) -> None:
    import torch
    from PIL import Image

    from src.semantic_responsibility_diagnostic import (
        jensen_shannon_rows,
        normalized_quaternion_rotations,
        project_world_points,
        symmetric_ellipsoid_children,
    )

    split = read(ROOT / config["split_path"])
    train_names = sorted(split["primary_fold"]["semantic_train_labeled"])
    forbidden = set(split["primary_fold"]["semantic_validation_labeled"])
    if set(train_names) & forbidden or len(train_names) != 201 or len(forbidden) != 30:
        raise ValueError("Unexpected or overlapping semantic roles")
    manifest = read(ROOT / config["label_manifest"])
    records = {record["image"]: record for record in manifest["records"]}
    if not set(train_names).issubset(records):
        raise ValueError("Training label manifest is incomplete")

    directions_path = ROOT / config["direction_evidence"]
    common_path = ROOT / config["semantic_common"]
    checkpoint_path = ROOT / config["rgb_parent_checkpoint"]
    directions = torch.load(directions_path, map_location="cpu", weights_only=False)
    common = torch.load(common_path, map_location="cpu", weights_only=False)
    capture, iteration = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if iteration != 30000:
        raise ValueError(f"Unexpected parent iteration {iteration}")
    selected = directions["selected"].numpy().astype(np.int64)
    if len(selected) != config["selected_parent_count"]:
        raise ValueError("Unexpected selected-parent count")
    expected = common["evidence"]["candidate"].numpy().astype(np.int64)
    if not np.array_equal(selected, expected):
        raise ValueError("Directions and common selection disagree")
    xyz = capture[1].detach().numpy()[selected]
    scales = np.exp(capture[4].detach().numpy()[selected].astype(np.float64))
    rotations = normalized_quaternion_rotations(capture[5].detach().numpy()[selected])
    lines = directions["candidate"].numpy()
    plus, minus, radii = symmetric_ellipsoid_children(xyz, scales, rotations, lines)

    cameras = read_colmap_poses(
        ROOT / config["scene"] / "sparse/0/images.txt", normalized_quaternion_rotations
    )
    camera_fields = [
        row.split()
        for row in (ROOT / config["scene"] / "sparse/0/cameras.txt").read_text(encoding="utf-8").splitlines()
        if row and not row.startswith("#")
    ]
    if len(camera_fields) != 1 or camera_fields[0][1] != "PINHOLE":
        raise ValueError("SEM-401 expects one PINHOLE camera")
    _, _, width, height, fx, fy, cx, cy = camera_fields[0]
    width, height = int(width), int(height)
    camera_matrix = np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    class_count = len(config["class_names"])
    plus_counts = np.zeros((len(selected), class_count), dtype=np.int64)
    minus_counts = np.zeros_like(plus_counts)
    paired_views = np.zeros(len(selected), dtype=np.int32)
    resolvable_views = np.zeros(len(selected), dtype=np.int32)
    separation_sum = np.zeros(len(selected), dtype=np.float64)
    all_resolvable_separations: list[np.ndarray] = []
    for index, name in enumerate(train_names, 1):
        if name in forbidden:
            raise RuntimeError(f"Forbidden confirmation label requested: {name}")
        label_path = ROOT / config["labels"] / name
        if sha256(label_path) != records[name]["label_png_sha256"]:
            raise ValueError(f"Label hash mismatch: {name}")
        labels = np.asarray(Image.open(label_path), dtype=np.uint8)
        if labels.shape != (height, width):
            raise ValueError(f"Label size mismatch: {name}")
        camera_rotation, camera_translation = cameras[name]
        plus_px, plus_depth = project_world_points(
            plus, camera_rotation, camera_translation, camera_matrix
        )
        minus_px, minus_depth = project_world_points(
            minus, camera_rotation, camera_translation, camera_matrix
        )
        finite = np.isfinite(plus_px).all(axis=1) & np.isfinite(minus_px).all(axis=1)
        inside = (
            finite
            & (plus_depth > 0)
            & (minus_depth > 0)
            & (plus_px[:, 0] >= 0)
            & (plus_px[:, 0] < width)
            & (plus_px[:, 1] >= 0)
            & (plus_px[:, 1] < height)
            & (minus_px[:, 0] >= 0)
            & (minus_px[:, 0] < width)
            & (minus_px[:, 1] >= 0)
            & (minus_px[:, 1] < height)
        )
        paired_views += inside.astype(np.int32)
        separation = np.linalg.norm(plus_px - minus_px, axis=1)
        resolvable = inside & (separation >= config["minimum_resolvable_separation_px"])
        indices = np.flatnonzero(resolvable)
        if len(indices):
            plus_xy = np.rint(plus_px[indices]).astype(np.int64)
            minus_xy = np.rint(minus_px[indices]).astype(np.int64)
            plus_xy[:, 0] = np.clip(plus_xy[:, 0], 0, width - 1)
            plus_xy[:, 1] = np.clip(plus_xy[:, 1], 0, height - 1)
            minus_xy[:, 0] = np.clip(minus_xy[:, 0], 0, width - 1)
            minus_xy[:, 1] = np.clip(minus_xy[:, 1], 0, height - 1)
            plus_label = labels[plus_xy[:, 1], plus_xy[:, 0]]
            minus_label = labels[minus_xy[:, 1], minus_xy[:, 0]]
            if np.any(plus_label >= class_count) or np.any(minus_label >= class_count):
                raise ValueError(f"Out-of-range label: {name}")
            np.add.at(plus_counts, (indices, plus_label), 1)
            np.add.at(minus_counts, (indices, minus_label), 1)
            resolvable_views[indices] += 1
            separation_sum[indices] += separation[indices]
            all_resolvable_separations.append(separation[indices])
        if index % 25 == 0 or index == len(train_names):
            print(json.dumps({"processed_training_labels": index, "total": len(train_names)}), flush=True)

    covered = resolvable_views >= config["minimum_resolvable_views_per_parent"]
    covered_indices = np.flatnonzero(covered)
    if not len(covered_indices):
        raise RuntimeError("No parents have resolvable paired evidence")
    js = np.zeros(len(selected), dtype=np.float64)
    js[covered] = jensen_shannon_rows(plus_counts[covered], minus_counts[covered])
    plus_main = plus_counts.argmax(axis=1)
    minus_main = minus_counts.argmax(axis=1)
    side_main_differs = plus_main != minus_main
    differentiated = covered & side_main_differs & (js >= config["differentiated_js_threshold"])
    cable = int(config["class_id_of_interest"])
    plus_total = plus_counts.sum(axis=1).clip(min=1)
    minus_total = minus_counts.sum(axis=1).clip(min=1)
    plus_cable = plus_counts[:, cable] / plus_total
    minus_cable = minus_counts[:, cable] / minus_total
    cable_fraction_difference = np.abs(plus_cable - minus_cable)
    cable_main_complement = (plus_main == cable) ^ (minus_main == cable)
    cable_separated = (
        covered
        & cable_main_complement
        & (cable_fraction_difference >= config["minimum_cable_fraction_difference"])
    )
    total = len(selected)
    fractions = {
        "resolvable_parent_fraction": float(covered.mean()),
        "differentiated_parent_fraction_of_all": float(differentiated.mean()),
        "cable_separated_parent_fraction_of_all": float(cable_separated.mean()),
    }
    gates_config = config["gates"]
    gates = {
        "resolvable_parent_fraction": fractions["resolvable_parent_fraction"]
        >= gates_config["resolvable_parent_fraction_at_least"],
        "differentiated_parent_fraction_of_all": fractions["differentiated_parent_fraction_of_all"]
        >= gates_config["differentiated_parent_fraction_of_all_at_least"],
        "cable_separated_parent_fraction_of_all": fractions["cable_separated_parent_fraction_of_all"]
        >= gates_config["cable_separated_parent_fraction_of_all_at_least"],
    }
    separations = np.concatenate(all_resolvable_separations)
    evidence_path = output / "evidence.npz"
    np.savez_compressed(
        evidence_path,
        selected=selected,
        radii=radii,
        paired_views=paired_views,
        resolvable_views=resolvable_views,
        plus_counts=plus_counts,
        minus_counts=minus_counts,
        js=js,
        plus_main=plus_main,
        minus_main=minus_main,
        cable_fraction_difference=cable_fraction_difference,
        covered=covered,
        differentiated=differentiated,
        cable_separated=cable_separated,
    )
    summary = {
        "selected_parents": total,
        "training_labels_read": len(train_names),
        "confirmation_labels_read": 0,
        "paired_parent_view_observations": int(paired_views.sum()),
        "resolvable_parent_view_observations": int(resolvable_views.sum()),
        "covered_parents": int(covered.sum()),
        "differentiated_parents": int(differentiated.sum()),
        "cable_separated_parents": int(cable_separated.sum()),
        **fractions,
        "covered_js_quantiles": quantiles(js[covered]),
        "covered_cable_fraction_difference_quantiles": quantiles(cable_fraction_difference[covered]),
        "resolvable_separation_px_quantiles": quantiles(separations),
        "parent_radius_quantiles": quantiles(radii),
        "gates": gates,
        "training_authorized": all(gates.values()),
        "interpretation_scope": "projection_only_optimistic_upper_bound_without_occlusion_or_rasterized_contribution",
    }
    write(output / "summary.json", summary)
    result.update(summary, evidence_sha256=sha256(evidence_path), summary_sha256=sha256(output / "summary.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="r1")
    args = parser.parse_args()
    config = read(CONFIG)
    output = ROOT / config["output"] / f"diagnostic-{args.attempt}"
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment_id": config["experiment_id"],
        "status": "starting",
        "seed": config["seed"],
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write(output / "started.json", result)
    started = time.monotonic()
    try:
        run(config, output, result)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed_preserved", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write(output / "result.json", result)


if __name__ == "__main__":
    main()
