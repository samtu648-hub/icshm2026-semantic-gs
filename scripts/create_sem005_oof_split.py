#!/usr/bin/env python3
"""Create the frozen three-fold spatial OOF split for SEM-005.

Only the 219 labelled training views from SEM-001 participate.  Deterministic
three-centre Lloyd clustering defines spatial OOF regions.  For each fold,
views outside the OOF region but closer than the parent R1 guard radius to an
OOF camera are removed from that fold's training set as a local guard band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_clusters(points: np.ndarray, seed: int, count: int = 3) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    initial = [int(rng.integers(len(points)))]
    while len(initial) < count:
        distance_sq = np.min(
            np.stack([np.sum((points - points[index]) ** 2, axis=1) for index in initial]),
            axis=0,
        )
        initial.append(int(np.argmax(distance_sq)))
    centres = points[initial].copy()
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(100):
        distance_sq = np.stack([np.sum((points - centre) ** 2, axis=1) for centre in centres], axis=1)
        next_labels = np.argmin(distance_sq, axis=1)
        next_centres = np.stack([points[next_labels == index].mean(axis=0) for index in range(count)])
        labels = next_labels
        if np.allclose(centres, next_centres, rtol=0.0, atol=1e-12):
            centres = next_centres
            break
        centres = next_centres
    if any(int(np.sum(labels == index)) == 0 for index in range(count)):
        raise RuntimeError("Spatial clustering produced an empty OOF fold")
    return labels, centres, initial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-split", type=Path, required=True)
    parser.add_argument("--camera-graph", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--guard-radius", type=float, default=0.5220235426020557)
    args = parser.parse_args()

    parent = read_json(args.parent_split)
    graph = read_json(args.camera_graph)
    names = list(parent["semantic_train_labeled"])
    if len(names) != 219 or len(set(names)) != 219:
        raise ValueError("SEM-005 requires exactly the frozen 219 unique SEM-001 training views")
    validation = set(parent["semantic_validation_labeled"])
    parent_guard = set(parent["buffer_excluded_labeled"])
    nodes = {node["name"]: node for node in graph["nodes"]}
    missing = sorted(set(names) - set(nodes))
    if missing:
        raise ValueError(f"Camera graph is missing SEM-001 training views: {missing}")
    points = np.asarray([nodes[name]["center"] for name in names], dtype=np.float64)
    assignments, centres, initial = deterministic_clusters(points, args.seed)

    all_train = set(names)
    folds: list[dict[str, Any]] = []
    oof_seen: list[str] = []
    for fold_index in range(3):
        oof = {names[index] for index in np.flatnonzero(assignments == fold_index)}
        remaining = all_train - oof
        guard: set[str] = set()
        minimum_distance: dict[str, float] = {}
        for candidate in remaining:
            distance = min(float(np.linalg.norm(np.asarray(nodes[candidate]["center"]) - np.asarray(nodes[name]["center"]))) for name in oof)
            minimum_distance[candidate] = distance
            if distance < args.guard_radius:
                guard.add(candidate)
        train = remaining - guard
        if train & oof or train & guard or oof & guard:
            raise RuntimeError(f"Fold {fold_index} train/OOF/guard sets overlap")
        cross_edge_count = 0
        adjacent_cross_edges = 0
        for edge in graph["edges"]:
            source, target = edge["source"], edge["target"]
            if (source in train and target in oof) or (target in train and source in oof):
                cross_edge_count += 1
                if float(edge["center_distance"]) < args.guard_radius:
                    adjacent_cross_edges += 1
        if adjacent_cross_edges:
            raise RuntimeError(f"Fold {fold_index} retains {adjacent_cross_edges} adjacent train/OOF graph edges")
        oof_seen.extend(sorted(oof))
        folds.append(
            {
                "fold": fold_index,
                "train": sorted(train),
                "oof": sorted(oof),
                "guard": sorted(guard),
                "counts": {"train": len(train), "oof": len(oof), "guard": len(guard)},
                "checks": {
                    "disjoint_train_oof_guard": True,
                    "minimum_train_to_oof_camera_distance": min(minimum_distance[name] for name in train),
                    "guard_radius_strictly_respected": min(minimum_distance[name] for name in train) >= args.guard_radius,
                    "adjacent_camera_graph_edges_train_to_oof": adjacent_cross_edges,
                    "all_camera_graph_edges_train_to_oof": cross_edge_count,
                },
            }
        )

    if sorted(oof_seen) != sorted(names) or len(oof_seen) != len(set(oof_seen)):
        raise RuntimeError("Every SEM-001 training view must occur in exactly one OOF set")
    label_manifest = read_json(args.label_manifest)
    output = {
        "schema_version": 1,
        "split_id": "sem_005_oof_spatial_v1",
        "purpose": "three spatial OOF folds within the frozen 219-view SEM-001 semantic training side",
        "parent_split_id": parent["split_id"],
        "parent_split_sha256": sha256(args.parent_split),
        "label_manifest_id": label_manifest["label_manifest_id"],
        "label_manifest_sha256": sha256(args.label_manifest),
        "camera_graph_sha256": sha256(args.camera_graph),
        "seed": args.seed,
        "method": "deterministic farthest-initialized three-centre Lloyd clustering over official camera centres",
        "guard_policy": {
            "radius": args.guard_radius,
            "source": "frozen rgb_r1_spatial_v1 primary-fold guard radius",
            "rule": "exclude every non-OOF training view whose official camera centre is strictly closer than radius to any OOF camera",
        },
        "pose_convention": "CAMERA_FROM_WORLD; graph centres were computed as C = -R^T t",
        "cluster_initial_indices_in_parent_order": initial,
        "cluster_centres": centres.tolist(),
        "fixed_validation_read_only": sorted(validation),
        "fixed_parent_guard_never_used": sorted(parent_guard),
        "unlabeled_never_used_for_model_training": list(parent["excluded_unlabeled"]),
        "folds": folds,
        "global_checks": {
            "parent_train_count": len(names),
            "oof_union_count": len(oof_seen),
            "every_parent_train_view_exactly_once_oof": True,
            "fixed_validation_disjoint_from_all_fold_inputs": not bool(validation & all_train),
            "parent_guard_disjoint_from_all_fold_inputs": not bool(parent_guard & all_train),
            "unlabeled_training_count": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "fold_counts": [fold["counts"] for fold in folds]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
