#!/usr/bin/env python3
"""Freeze the SEM-060 fit/check/guard split from training-only spatial folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(parent_path: Path, source_path: Path, graph_path: Path) -> dict:
    parent = _read(parent_path)
    source = _read(source_path)
    graph = _read(graph_path)
    parent_train = set(parent["semantic_train_labeled"])
    if len(parent_train) != 219:
        raise ValueError("SEM-060 requires the frozen 219-view SEM-001 training side")
    selected = min(source["folds"], key=lambda fold: (-len(fold["train"]), int(fold["fold"])))
    fit = set(selected["train"])
    check = set(selected["oof"])
    guard = set(selected["guard"])
    if fit | check | guard != parent_train or fit & check or fit & guard or check & guard:
        raise ValueError("Selected fit/check/guard sets must be a disjoint partition of the 219 training views")
    validation = set(parent["semantic_validation_labeled"])
    excluded = validation | set(parent["buffer_excluded_labeled"]) | set(parent["excluded_unlabeled"])
    if parent_train & excluded:
        raise ValueError("Training-side SEM-060 inputs overlap a prohibited parent cohort")
    nodes = {node["name"]: node for node in graph["nodes"]}
    if not parent_train <= set(nodes):
        raise ValueError("Camera graph is missing SEM-001 training views")
    nearby_cross_edges = 0
    all_cross_edges = 0
    guard_radius = float(source["guard_policy"]["radius"])
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if (a in fit and b in check) or (b in fit and a in check):
            all_cross_edges += 1
            if float(edge["center_distance"]) < guard_radius:
                nearby_cross_edges += 1
    if nearby_cross_edges:
        raise ValueError("Fit/check split retains a camera-graph edge inside the guard radius")
    return {
        "schema_version": 1,
        "split_id": "sem_060_cable_geometry_v1",
        "purpose": "independent fit/check geometry gate within the frozen 219-view SEM-001 training side",
        "parent_split_id": parent["split_id"],
        "parent_split_sha256": _sha256(parent_path),
        "source_training_only_split_id": source["split_id"],
        "source_training_only_split_sha256": _sha256(source_path),
        "camera_graph_sha256": _sha256(graph_path),
        "selection_rule": "largest source-fold train cohort; ties resolved by lowest fold id",
        "selected_source_fold": int(selected["fold"]),
        "seed": source["seed"],
        "pose_convention": "CAMERA_FROM_WORLD; C = -R^T t",
        "geometry_fit": sorted(fit),
        "geometry_check": sorted(check),
        "local_guard_band": sorted(guard),
        "fixed_validation_read_only_names": sorted(validation),
        "parent_guard_never_used": sorted(parent["buffer_excluded_labeled"]),
        "unlabeled_never_used": sorted(parent["excluded_unlabeled"]),
        "counts": {"fit": len(fit), "check": len(check), "guard": len(guard)},
        "checks": {
            "disjoint_partition_of_219_training_views": True,
            "prohibited_cohorts_disjoint": True,
            "guard_radius": guard_radius,
            "minimum_fit_to_check_camera_distance": selected["checks"]["minimum_train_to_oof_camera_distance"],
            "nearby_camera_graph_edges_fit_to_check": nearby_cross_edges,
            "all_camera_graph_edges_fit_to_check": all_cross_edges,
            "fixed_validation_images_or_labels_opened": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--camera-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_split(args.parent, args.source, args.camera_graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output), "counts": payload["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
