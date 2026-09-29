"""Pure-Python VGGT-Omega gates and joint co-visibility window planning.

This module deliberately has no third-party imports. It never imports the
model, loads a checkpoint, accesses the network, or initializes CUDA.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

from .output_schema import geometry_output_schema_v2, validate_geometry_manifest_v2


SUPPORTED_VIEW_COUNTS = (16, 25)
SMOKE_ANCHORS = ("115.png", "126.png")
CANONICAL_VIEW_COUNT = 25
SOURCE_COMMIT = "a3ab0141f96838724423541044ff5ba301cfd36a"
CHECKPOINT_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
CHECKPOINT_SIZE_BYTES = 4_576_706_117
LICENSE_SHA256 = "27696e51930daf5d24222e2d9d65339f529153e853de0e2eb457133a1a6f5239"
README_SHA256 = "8526d8578d3246c9042115dd22c211b53842639262833d41eaf717f56567264e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_fixed_artifacts(
    source_root: Path, checkpoint: Path, checkpoint_license: Path
) -> dict:
    """Verify all immutable local artifacts before any model import or load."""
    source_root = source_root.resolve()
    checkpoint = checkpoint.resolve()
    checkpoint_license = checkpoint_license.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"VGGT-Omega source root does not exist: {source_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint does not exist: {checkpoint}")
    if not checkpoint_license.is_file():
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint license does not exist: {checkpoint_license}"
        )
    source_license = source_root / "LICENSE"
    if not source_license.is_file():
        raise FileNotFoundError(
            f"VGGT-Omega source license does not exist: {source_license}"
        )
    source_readme = source_root / "README.md"
    if not source_readme.is_file():
        raise FileNotFoundError(
            f"VGGT-Omega source README does not exist: {source_readme}"
        )

    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    actual_commit = completed.stdout.strip()
    if actual_commit != SOURCE_COMMIT:
        raise ValueError(
            f"VGGT-Omega source commit mismatch: {actual_commit} != {SOURCE_COMMIT}"
        )
    source_status = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if source_status:
        raise ValueError("VGGT-Omega fixed source checkout is not clean")
    source_license_sha256 = sha256_file(source_license)
    if source_license_sha256 != LICENSE_SHA256:
        raise ValueError("VGGT-Omega source LICENSE hash mismatch")
    source_readme_sha256 = sha256_file(source_readme)
    if source_readme_sha256 != README_SHA256:
        raise ValueError("VGGT-Omega source README hash mismatch")
    checkpoint_license_sha256 = sha256_file(checkpoint_license)
    if checkpoint_license_sha256 != LICENSE_SHA256:
        raise ValueError("VGGT-Omega checkpoint LICENSE hash mismatch")
    checkpoint_size = checkpoint.stat().st_size
    if checkpoint_size != CHECKPOINT_SIZE_BYTES:
        raise ValueError(
            f"VGGT-Omega checkpoint size mismatch: {checkpoint_size} != {CHECKPOINT_SIZE_BYTES}"
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("VGGT-Omega checkpoint SHA256 mismatch")
    return {
        "source_root": str(source_root),
        "source_commit": actual_commit,
        "source_checkout_clean": True,
        "source_license_sha256": source_license_sha256,
        "source_readme_sha256": source_readme_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_license": str(checkpoint_license),
        "checkpoint_license_sha256": checkpoint_license_sha256,
    }


def training_names_from_split(split: dict) -> list[str]:
    """Return only primary_fold.train_rgb_400 after rejecting role leakage."""
    primary = split.get("primary_fold")
    if not isinstance(primary, dict):
        raise ValueError("split is missing primary_fold")
    names = primary.get("train_rgb_400")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError(
            "primary_fold.train_rgb_400 must be a nonempty image-name list"
        )
    if len(names) != len(set(names)):
        raise ValueError("primary_fold.train_rgb_400 contains duplicate image names")
    excluded_keys = (
        "holdout_all_cameras",
        "validation_labeled",
        "withheld_unlabeled",
        "buffer_excluded_all_cameras",
        "buffer_excluded_labeled",
        "buffer_excluded_unlabeled",
    )
    excluded: set[str] = set()
    for key in excluded_keys:
        values = primary.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(name, str) for name in values
        ):
            raise ValueError(f"primary_fold.{key} must be an image-name list")
        excluded.update(values)
    overlap = set(names) & excluded
    if overlap:
        raise ValueError(
            f"train_rgb_400 overlaps validation or guard roles: {sorted(overlap)}"
        )
    return sorted(names)


def _ordered_names(graph: dict, available_images: set[str]) -> list[str]:
    names = [
        node["name"]
        for node in graph.get("nodes", [])
        if node.get("name") in available_images
    ]
    if len(names) != len(set(names)):
        raise ValueError("camera graph contains duplicate node image names")
    return sorted(names)


def _strong_neighbors(
    graph: dict, anchor: str, available_images: set[str]
) -> list[dict]:
    """Rank real graph edges, falling back from capped adjacency to all edges."""
    by_target: dict[str, dict] = {}

    def retain(edge: dict, target: object, edge_source: str) -> None:
        if (
            not isinstance(target, str)
            or target == anchor
            or target not in available_images
        ):
            return
        shared = int(edge.get("shared_point3d", 0))
        if shared <= 0:
            return
        candidate = {
            "source": anchor,
            "target": target,
            "shared_point3d": shared,
            "center_distance": edge.get("center_distance"),
            "optical_axis_angle_deg": edge.get("optical_axis_angle_deg"),
            "edge_source": edge_source,
        }
        previous = by_target.get(target)
        if previous is None or shared > previous["shared_point3d"]:
            by_target[target] = candidate

    adjacency = graph.get("adjacency", {})
    if not isinstance(adjacency, dict):
        raise ValueError("camera graph adjacency must be an object")
    for edge in adjacency.get(anchor, []):
        if isinstance(edge, dict):
            retain(edge, edge.get("target"), "adjacency")
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if edge.get("source") == anchor:
            retain(edge, edge.get("target"), "complete_edges")
        elif edge.get("target") == anchor:
            retain(edge, edge.get("source"), "complete_edges")
    return sorted(
        by_target.values(), key=lambda edge: (-edge["shared_point3d"], edge["target"])
    )


def build_joint_covisibility_windows(
    graph: dict,
    image_names: Iterable[str],
    *,
    view_count: int,
    anchors: Iterable[str] = SMOKE_ANCHORS,
) -> list[dict]:
    """Build the two frozen smoke regions from canonical 25-view rankings.

    The 16-view fallback is sliced from the corresponding 25-view region. Every
    non-anchor member therefore has a positive official shared-POINT3D edge;
    filename or video continuity is never used as padding.
    """
    if view_count not in SUPPORTED_VIEW_COUNTS:
        raise ValueError(f"view_count must be one of {SUPPORTED_VIEW_COUNTS}")
    available_images = set(image_names)
    windows = []
    anchor_names = tuple(anchors)
    if not anchor_names or len(anchor_names) != len(set(anchor_names)):
        raise ValueError("smoke anchors must be a nonempty unique sequence")
    graph_names = set(_ordered_names(graph, available_images))
    missing_anchors = set(anchor_names) - graph_names
    if missing_anchors:
        raise ValueError(
            f"fixed smoke anchor(s) unavailable: {sorted(missing_anchors)}"
        )
    for anchor in anchor_names:
        neighbors = _strong_neighbors(graph, anchor, available_images)
        if len(neighbors) < CANONICAL_VIEW_COUNT - 1:
            raise ValueError(
                f"{anchor} has only {len(neighbors)} positive shared-POINT3D neighbors; "
                f"need {CANONICAL_VIEW_COUNT - 1} without weak padding"
            )
        canonical = neighbors[: CANONICAL_VIEW_COUNT - 1]
        selected = canonical[: view_count - 1]
        canonical_images = [anchor, *(edge["target"] for edge in canonical)]
        images = canonical_images[:view_count]
        anchor_stem = Path(anchor).stem
        windows.append(
            {
                "window_manifest_id": f"vggt-omega-smoke-{anchor_stem}-v{view_count}",
                "canonical_25_window_manifest_id": f"vggt-omega-smoke-{anchor_stem}-v25",
                "anchor_image": anchor,
                "images": images,
                "selection_rule": "anchor_plus_highest_shared_point3d_neighbors",
                "effective_image_coverage": {"count": len(images), "images": images},
                "anchor_shared_point3d": {
                    edge["target"]: int(edge.get("shared_point3d", 0))
                    for edge in selected
                },
                "co_visibility_facts": [dict(edge) for edge in selected],
                "canonical_25_images": canonical_images,
                "fallback_prefix_length": view_count,
                "state_reset": "independent_joint_view_set",
            }
        )
    return windows


def build_vggt_omega_manifest(
    graph: dict,
    image_names: Iterable[str],
    *,
    view_count: int,
    anchors: Iterable[str] = SMOKE_ANCHORS,
    split_id: str | None = None,
    scope: str = "smoke",
) -> dict:
    """Create a deterministic, fusion-compatible VGGT-Omega manifest."""
    images = sorted(set(image_names))
    anchor_names = tuple(anchors)
    windows = build_joint_covisibility_windows(
        graph, images, view_count=view_count, anchors=anchor_names
    )
    output_schema = geometry_output_schema_v2()
    output_schema["required_arrays"].update(
        {
            "world_points": "float32[height,width,3]",
            "pinhole_intrinsics": "float64[3,3]",
            "official_cam2world": "float64[4,4]",
            "depth_scale_ratio": "float64 scalar",
        }
    )
    output_schema["optional_arrays"] = {"pred_cam2world": "float64[4,4]"}
    payload = {
        "schema_version": 2,
        "model": "VGGT-Omega",
        "model_execution": "not_run",
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_id": split_id,
        "scope": scope,
        "input_scope": "primary_fold.train_rgb_400_only",
        "image_count_available": len(images),
        "input_organization": {
            "kind": "joint_covisibility_sets",
            "view_count": view_count,
            "canonical_view_count": CANONICAL_VIEW_COUNT,
            "anchors": list(anchor_names),
            "selection_rule": "anchor_plus_highest_shared_point3d_neighbors",
            "camera_graph_pose_convention": graph.get("pose_convention"),
            "state_reset": "independent_joint_view_set",
            "fallback_rule": "16_is_exact_prefix_of_each_25_view_region",
        },
        "window_count": len(windows),
        "unique_image_coverage": {
            "count": len({name for window in windows for name in window["images"]}),
            "images": sorted({name for window in windows for name in window["images"]}),
        },
        "windows": windows,
        "geometry_output_schema": output_schema,
        "output_contract": "one geometry_prior_npz_v2 per image plus this manifest",
    }
    validate_geometry_manifest_v2(payload)
    return payload
