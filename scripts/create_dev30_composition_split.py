#!/usr/bin/env python3
"""Create the fixed 30-view local-composition development holdout.

The split uses official camera centers and optical axes to form local groups.
Ground-truth masks are used only for pre-registered composition strata; no RGB
pixels, rendered images, model predictions, or model scores enter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_semantic_labels import CLASS_IDS, make_label, read_camera  # noqa: E402


GROUP_SIZE = 10
LABELED_COUNT = 300
CAMERA_COUNT = 400
UNLABELED_GUARD = 0.5220235426020557
FOREGROUND_QUANTILE = 0.20
DECK_ANGLE_QUANTILE = 0.80
DUPLICATE_NN_QUANTILE = 0.20
CENTER_RMS_MULTIPLIER = 5.0
AXIS_MAX_TO_MEAN_DEG = 15.0
TARGET_STRATA = ("regular", "oblique", "distant")
EXPECTED_HOLDOUT_SHA256 = (
    "d32f0091a95ba59f0e6fe0f86dd2368f51c5341289e1525c87ff911bf8b5b7ac"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def name_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (sys.maxsize, name)


def pair_key(left: str, right: str) -> tuple[str, str]:
    ordered = sorted((left, right), key=name_key)
    return ordered[0], ordered[1]


def vector_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def deck_axis_angle_deg(label: np.ndarray) -> float:
    y, x = np.nonzero(label == CLASS_IDS["deck"])
    if len(x) < 2:
        raise ValueError("Every labeled view must contain at least two deck pixels")
    coordinates = np.column_stack((x, y)).astype(np.float64)
    covariance = np.cov(coordinates, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    raw_angle = (
        abs(float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))) % 180.0
    )
    return min(raw_angle, 180.0 - raw_angle)


def classify_stratum(
    foreground_fraction: float,
    deck_angle_deg: float,
    foreground_q20: float,
    deck_angle_q80: float,
) -> str:
    low_foreground = foreground_fraction <= foreground_q20
    high_angle = deck_angle_deg >= deck_angle_q80
    if low_foreground and high_angle:
        return "hybrid"
    if low_foreground:
        return "distant"
    if high_angle:
        return "oblique"
    return "regular"


def aggregate_counts(
    names: list[str] | tuple[str, ...], records: dict[str, dict[str, Any]], key: str
) -> dict[str, int]:
    total: Counter[str] = Counter()
    for name in names:
        total.update(
            {
                str(label): int(value)
                for label, value in dict(records[name][key]).items()
            }
        )
    return dict(sorted(total.items()))


def maximum_spanning_tree_bottleneck(
    group: tuple[str, ...], edge_lookup: dict[tuple[str, str], dict[str, Any]]
) -> int:
    parent = {name: name for name in group}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    selected: list[int] = []
    edges = []
    for left, right in combinations(group, 2):
        edge = edge_lookup.get(pair_key(left, right))
        if edge is not None:
            edges.append((int(edge["shared_point3d"]), left, right))
    for shared, left, right in sorted(edges, reverse=True):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        parent[left_root] = right_root
        selected.append(shared)
        if len(selected) == len(group) - 1:
            return min(selected)
    raise ValueError("Candidate group is not connected by covisibility edges")


def build_split(
    dataset_root: Path, camera_graph_path: Path, dataset_manifest_path: Path
) -> dict[str, Any]:
    labeled_names = [f"{index:03d}.png" for index in range(1, LABELED_COUNT + 1)]
    all_names = [f"{index:03d}.png" for index in range(1, CAMERA_COUNT + 1)]
    unlabeled_names = all_names[LABELED_COUNT:]
    graph: dict[str, Any] = json.loads(camera_graph_path.read_text(encoding="utf-8"))
    nodes = {str(record["name"]): record for record in graph["nodes"]}
    if graph.get("image_count") != CAMERA_COUNT or set(nodes) != set(all_names):
        raise ValueError("Camera graph must contain exactly the released 400 cameras")
    if graph.get("min_shared_points") != 8:
        raise ValueError("Camera graph covisibility threshold changed")

    centers = {
        name: np.asarray(nodes[name]["center"], dtype=np.float64) for name in all_names
    }
    axes = {
        name: np.asarray(nodes[name]["optical_axis"], dtype=np.float64)
        for name in all_names
    }
    if any(
        not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1e-12)
        for axis in axes.values()
    ):
        raise ValueError("Camera graph contains a non-unit optical axis")

    def center_distance(left: str, right: str) -> float:
        return float(np.linalg.norm(centers[left] - centers[right]))

    def axis_angle(left: str, right: str) -> float:
        return vector_angle_deg(axes[left], axes[right])

    camera_matrix, distortion = read_camera(
        dataset_root / "camera_parameters" / "cameras.txt"
    )
    composition: dict[str, dict[str, Any]] = {}
    for name in labeled_names:
        annotation = dataset_root / "json" / f"{Path(name).stem}.json"
        label, shape_counts = make_label(annotation, camera_matrix, distortion)
        pixel_values = np.bincount(label.ravel(), minlength=len(CLASS_IDS))
        composition[name] = {
            "foreground_fraction": float(pixel_values[1:].sum() / label.size),
            "deck_axis_angle_deg": deck_axis_angle_deg(label),
            "pixel_counts": {
                str(index): int(value) for index, value in enumerate(pixel_values)
            },
            "shape_counts": dict(sorted(shape_counts.items())),
        }

    foreground_q20 = float(
        np.quantile(
            [float(composition[name]["foreground_fraction"]) for name in labeled_names],
            FOREGROUND_QUANTILE,
            method="linear",
        )
    )
    deck_angle_q80 = float(
        np.quantile(
            [float(composition[name]["deck_axis_angle_deg"]) for name in labeled_names],
            DECK_ANGLE_QUANTILE,
            method="linear",
        )
    )
    for record in composition.values():
        record["stratum"] = classify_stratum(
            float(record["foreground_fraction"]),
            float(record["deck_axis_angle_deg"]),
            foreground_q20,
            deck_angle_q80,
        )

    global_nearest = {
        name: min(center_distance(name, other) for other in all_names if other != name)
        for name in all_names
    }
    global_nearest_values = np.asarray(list(global_nearest.values()), dtype=np.float64)
    global_nn_median = float(np.median(global_nearest_values))
    duplicate_nn_q20 = float(
        np.quantile(global_nearest_values, DUPLICATE_NN_QUANTILE, method="linear")
    )
    center_rms_limit = CENTER_RMS_MULTIPLIER * global_nn_median
    nearest_unlabeled = {
        name: min(center_distance(name, other) for other in unlabeled_names)
        for name in labeled_names
    }
    eligible = {
        stratum: [
            name
            for name in labeled_names
            if composition[name]["stratum"] == stratum
            and nearest_unlabeled[name] >= UNLABELED_GUARD
            and global_nearest[name] >= duplicate_nn_q20
        ]
        for stratum in TARGET_STRATA
    }

    edge_lookup: dict[tuple[str, str], dict[str, Any]] = {
        pair_key(str(edge["source"]), str(edge["target"])): edge
        for edge in graph["edges"]
    }

    def group_geometry(group: tuple[str, ...]) -> dict[str, Any]:
        group_centers = np.stack([centers[name] for name in group])
        group_axes = np.stack([axes[name] for name in group])
        centroid = group_centers.mean(axis=0)
        center_radii = np.linalg.norm(group_centers - centroid, axis=1)
        pair_distances = [
            center_distance(left, right) for left, right in combinations(group, 2)
        ]
        mean_axis = group_axes.sum(axis=0)
        mean_axis /= np.linalg.norm(mean_axis)
        axis_deviations = np.asarray(
            [vector_angle_deg(axis, mean_axis) for axis in group_axes]
        )
        pair_angles = [
            axis_angle(left, right) for left, right in combinations(group, 2)
        ]
        pair_edges = [
            edge_lookup.get(pair_key(left, right))
            for left, right in combinations(group, 2)
        ]
        shared = [
            int(edge["shared_point3d"]) for edge in pair_edges if edge is not None
        ]
        return {
            "centroid": [float(value) for value in centroid],
            "mean_optical_axis": [float(value) for value in mean_axis],
            "center_rms": float(np.sqrt(np.mean(center_radii**2))),
            "center_max_radius": float(center_radii.max()),
            "center_max_pair_distance": max(pair_distances),
            "axis_rms_deg": float(np.sqrt(np.mean(axis_deviations**2))),
            "axis_max_to_mean_deg": float(axis_deviations.max()),
            "axis_max_pair_deg": max(pair_angles),
            "covisible_pair_count": len(shared),
            "possible_pair_count": math.comb(len(group), 2),
            "minimum_shared_point3d": min(shared) if shared else 0,
            "maximum_spanning_tree_bottleneck_shared_point3d": (
                maximum_spanning_tree_bottleneck(group, edge_lookup)
                if len(shared) == math.comb(len(group), 2)
                else 0
            ),
        }

    def nearest_outside(
        group: tuple[str, ...], population: list[str]
    ) -> tuple[list[dict[str, Any]], float, float]:
        outside = [name for name in population if name not in set(group)]
        rows = []
        for name in group:
            nearest = min(
                outside,
                key=lambda other: (
                    center_distance(name, other),
                    axis_angle(name, other),
                    name_key(other),
                ),
            )
            rows.append(
                {
                    "holdout": name,
                    "nearest": nearest,
                    "center_distance": center_distance(name, nearest),
                    "optical_axis_angle_deg": axis_angle(name, nearest),
                }
            )
        distances = np.asarray([float(row["center_distance"]) for row in rows])
        return rows, float(distances.min()), float(np.median(distances))

    candidates: dict[str, list[dict[str, Any]]] = {}
    for stratum in TARGET_STRATA:
        deduplicated: set[tuple[str, ...]] = set()
        for seed in eligible[stratum]:
            nearest_members = sorted(
                eligible[stratum],
                key=lambda name: (
                    center_distance(seed, name),
                    axis_angle(seed, name),
                    name_key(name),
                ),
            )[:GROUP_SIZE]
            if len(nearest_members) == GROUP_SIZE:
                deduplicated.add(tuple(sorted(nearest_members, key=name_key)))
        accepted = []
        for group in sorted(deduplicated):
            geometry = group_geometry(group)
            if int(geometry["covisible_pair_count"]) != math.comb(GROUP_SIZE, 2):
                continue
            if float(geometry["center_rms"]) > center_rms_limit:
                continue
            if float(geometry["axis_max_to_mean_deg"]) > AXIS_MAX_TO_MEAN_DEG:
                continue
            _, minimum_outside, median_outside = nearest_outside(group, all_names)
            accepted.append(
                {
                    "images": list(group),
                    "geometry": geometry,
                    "minimum_nearest_remaining_distance": minimum_outside,
                    "median_nearest_remaining_distance": median_outside,
                    "rank": (
                        -median_outside,
                        -minimum_outside,
                        float(geometry["center_rms"]),
                        float(geometry["axis_rms_deg"]),
                        group,
                    ),
                }
            )
        candidates[stratum] = sorted(accepted, key=lambda record: record["rank"])
        if not candidates[stratum]:
            raise ValueError(f"No accepted local group for stratum {stratum}")

    winners = {
        stratum: tuple(candidates[stratum][0]["images"]) for stratum in TARGET_STRATA
    }
    holdout = sorted(
        (name for group in winners.values() for name in group), key=name_key
    )
    holdout_text = "".join(f"{name}\n" for name in holdout).encode("ascii")
    holdout_sha256 = hashlib.sha256(holdout_text).hexdigest()
    if len(holdout) != 30 or len(set(holdout)) != 30:
        raise ValueError("Selected groups do not form a disjoint 30-view holdout")
    if holdout_sha256 != EXPECTED_HOLDOUT_SHA256:
        raise ValueError(f"Frozen holdout changed: {holdout_sha256}")

    holdout_set = set(holdout)
    train_rgb = [name for name in all_names if name not in holdout_set]
    train_semantic = [name for name in labeled_names if name not in holdout_set]
    nearest_training_rows, minimum_train_distance, median_train_distance = (
        nearest_outside(tuple(holdout), train_rgb)
    )
    strict_buffer = [
        name
        for name in train_rgb
        if min(center_distance(name, held_out) for held_out in holdout)
        < UNLABELED_GUARD
    ]

    group_records = []
    for stratum in TARGET_STRATA:
        winner = candidates[stratum][0]
        names = list(winner["images"])
        group_records.append(
            {
                "group_id": f"{stratum}-local",
                "composition_stratum": stratum,
                "images": names,
                "geometry": winner["geometry"],
                "composition": {
                    "mean_foreground_fraction": float(
                        np.mean(
                            [composition[name]["foreground_fraction"] for name in names]
                        )
                    ),
                    "mean_deck_axis_angle_deg": float(
                        np.mean(
                            [composition[name]["deck_axis_angle_deg"] for name in names]
                        )
                    ),
                    "pixel_counts": aggregate_counts(
                        names, composition, "pixel_counts"
                    ),
                    "shape_counts": aggregate_counts(
                        names, composition, "shape_counts"
                    ),
                },
                "candidate_count_after_gates": len(candidates[stratum]),
                "selection_score": {
                    "minimum_nearest_remaining_distance": winner[
                        "minimum_nearest_remaining_distance"
                    ],
                    "median_nearest_remaining_distance": winner[
                        "median_nearest_remaining_distance"
                    ],
                },
            }
        )

    class_mapping = {str(value): key for key, value in CLASS_IDS.items()}
    payload = {
        "schema_version": 1,
        "split_id": "dev30_local_composition_v1",
        "purpose": "Fixed local-composition development holdout for full-chain RGB and semantic evaluation",
        "claim_scope": (
            "Development holdout excluded from this experiment's optimization, initialization observations, "
            "RGB references, and semantic labels; not a blind test or strict spatially guarded split."
        ),
        "source_coordinate_frame": "official_colmap_reference",
        "pose_convention": "CAMERA_FROM_WORLD; centers C=-R^T t; optical axis is camera-to-world +Z",
        "source_counts": {
            "all_released_rgb": CAMERA_COUNT,
            "labeled_rgb": LABELED_COUNT,
            "unlabeled_rgb": CAMERA_COUNT - LABELED_COUNT,
            "camera_records": len(nodes),
        },
        "source_sha256": {
            "camera_graph": sha256(camera_graph_path),
            "dataset_manifest": sha256(dataset_manifest_path),
            "cameras_txt": sha256(dataset_root / "camera_parameters" / "cameras.txt"),
            "images_txt": sha256(dataset_root / "camera_parameters" / "images.txt"),
            "label_rasterizer": sha256(ROOT / "scripts/prepare_semantic_labels.py"),
        },
        "selection_protocol": {
            "uses_model_outputs_or_scores": False,
            "uses_rgb_pixels": False,
            "composition_source": "Final rasterized official LabelMe masks on the full-frame PINHOLE grid",
            "group_size": GROUP_SIZE,
            "target_strata": list(TARGET_STRATA),
            "foreground_q20": foreground_q20,
            "deck_axis_angle_q80_deg": deck_angle_q80,
            "stratum_boundary_rule": "low foreground <= q20; high deck angle >= q80",
            "unlabeled_camera_exclusion_distance": UNLABELED_GUARD,
            "global_nearest_neighbor_q20": duplicate_nn_q20,
            "global_nearest_neighbor_median": global_nn_median,
            "center_rms_limit": center_rms_limit,
            "axis_max_to_mean_limit_deg": AXIS_MAX_TO_MEAN_DEG,
            "local_group_rule": (
                "For each eligible seed, use the ten center-nearest eligible cameras in the same stratum; "
                "camera-axis angle and filename break exact ties."
            ),
            "group_gates": "Complete >=8-point covisibility clique, center RMS limit, and axis max-to-mean limit",
            "ranking": (
                "Descending median then minimum nearest-remaining-camera distance, ascending center RMS then "
                "axis RMS, and filename tuple only as final tie-break."
            ),
            "stratum_counts_all_labeled": dict(
                sorted(
                    Counter(
                        str(record["stratum"]) for record in composition.values()
                    ).items()
                )
            ),
            "eligible_counts": {
                stratum: len(eligible[stratum]) for stratum in TARGET_STRATA
            },
        },
        "groups": group_records,
        "class_mapping": class_mapping,
        "composition_counts": {
            "all_labeled_pixel_counts": aggregate_counts(
                labeled_names, composition, "pixel_counts"
            ),
            "train_pixel_counts": aggregate_counts(
                train_semantic, composition, "pixel_counts"
            ),
            "holdout_pixel_counts": aggregate_counts(
                holdout, composition, "pixel_counts"
            ),
            "all_labeled_shape_counts": aggregate_counts(
                labeled_names, composition, "shape_counts"
            ),
            "train_shape_counts": aggregate_counts(
                train_semantic, composition, "shape_counts"
            ),
            "holdout_shape_counts": aggregate_counts(
                holdout, composition, "shape_counts"
            ),
        },
        "primary_fold_id": "dev30",
        "primary_fold": {
            "holdout_all_cameras": holdout,
            "validation_labeled": holdout,
            "withheld_unlabeled": [],
            "buffer_excluded_all_cameras": [],
            "buffer_excluded_labeled": [],
            "buffer_excluded_unlabeled": [],
            "train_rgb_400": train_rgb,
            "train_rgb_300": train_semantic,
            "semantic_train_labeled": train_semantic,
            "semantic_validation_labeled": holdout,
            "checks": {
                "no_filename_overlap": not bool(set(train_rgb) & holdout_set),
                "all_unlabeled_rgb_retained": set(unlabeled_names) <= set(train_rgb),
                "minimum_train_to_holdout_distance": minimum_train_distance,
                "median_holdout_to_nearest_train_distance": median_train_distance,
                "strict_guard_applied": False,
            },
        },
        "nearest_training_camera_by_holdout": nearest_training_rows,
        "strict_guard_counterfactual": {
            "guard_radius": UNLABELED_GUARD,
            "additional_excluded_all_cameras": strict_buffer,
            "additional_excluded_labeled": [
                name for name in strict_buffer if name in set(labeled_names)
            ],
            "additional_excluded_unlabeled": [
                name for name in strict_buffer if name in set(unlabeled_names)
            ],
            "rgb_train_count_if_applied": len(train_rgb) - len(strict_buffer),
            "semantic_train_count_if_applied": len(train_semantic)
            - len([name for name in strict_buffer if name in set(labeled_names)]),
        },
        "holdout_list_sha256": holdout_sha256,
        "global_checks": {
            "holdout_count": len(holdout),
            "rgb_train_count": len(train_rgb),
            "semantic_train_count": len(train_semantic),
            "groups_disjoint": sum(len(set(group)) for group in winners.values())
            == len(holdout_set),
            "holdout_is_labeled": holdout_set <= set(labeled_names),
            "all_expected_classes_present_in_every_holdout": all(
                all(
                    int(composition[name]["pixel_counts"][str(class_id)]) > 0
                    for class_id in range(1, 5)
                )
                for name in holdout
            ),
        },
    }
    if payload["global_checks"] != {
        "holdout_count": 30,
        "rgb_train_count": 370,
        "semantic_train_count": 270,
        "groups_disjoint": True,
        "holdout_is_labeled": True,
        "all_expected_classes_present_in_every_holdout": True,
    }:
        raise ValueError(f"Final split checks failed: {payload['global_checks']}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--camera-graph",
        type=Path,
        default=ROOT / "experiments/geometry_camera_graph.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=ROOT / "data/manifests/contest_dataset_v1.json",
    )
    parser.add_argument(
        "--write",
        type=Path,
        default=ROOT / "experiments/splits/dev30_local_composition_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_split(
        dataset_root=args.dataset_root.resolve(),
        camera_graph_path=args.camera_graph.resolve(),
        dataset_manifest_path=args.dataset_manifest.resolve(),
    )
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["global_checks"], sort_keys=True))
    print(f"Wrote development split: {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
