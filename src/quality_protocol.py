"""Quality-route provenance and whole-chain access contracts (no model imports)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    actual = {}
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Artifact escapes root: {name}")
        value = sha256(path)
        if value != digest:
            raise ValueError(f"Artifact hash mismatch: {name}")
        actual[name] = value
    return actual


def unique_names(values, role):
    names = list(values)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate {role} views")
    if any(not isinstance(n, str) or Path(n).name != n or not n.endswith(".png") for n in names):
        raise ValueError(f"Invalid {role} image name")
    return set(names)


def full_chain_folds(rgb_fold, semantic_folds, nodes, radius):
    """Extend frozen semantic outer folds to RGB/unlabelled access, camera-only.

    The supplied training universe is the original 303 RGB images, not all400.
    Outer held-out images can be rendered/evaluated but never fit or act as sources.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Positive finite guard radius required")
    universe = unique_names(rgb_fold["train_rgb_400"], "RGB training")
    labelled = unique_names(rgb_fold["train_rgb_300"], "labelled training")
    if not labelled <= universe:
        raise ValueError("Labelled training outside RGB universe")
    reserved = set(rgb_fold["validation_labeled"]) | set(rgb_fold["buffer_excluded_all_cameras"]) | set(rgb_fold["withheld_unlabeled"])
    if universe & reserved:
        raise ValueError("Reserved RGB views leak into training")
    centres = {}
    for node in nodes:
        if node["name"] in centres:
            raise ValueError("Duplicate camera metadata")
        c = node["center"]
        if len(c) != 3 or not all(math.isfinite(v) for v in c):
            raise ValueError("Invalid official camera centre")
        centres[node["name"]] = c
    if not universe <= centres.keys():
        raise ValueError("Missing training camera metadata")
    output, all_targets, seen_folds = [], set(), set()
    for fold in semantic_folds:
        train, target, guard = (unique_names(fold[k], k) for k in ("train", "oof", "guard"))
        if fold["fold"] in seen_folds or target & all_targets:
            raise ValueError("Duplicate outer fold or prediction target")
        seen_folds.add(fold["fold"])
        all_targets |= target
        if not train or not target or train & target or train & guard or target & guard:
            raise ValueError("Invalid outer semantic partition")
        if train | target | guard != labelled:
            raise ValueError("Outer semantic partition differs from parent training")
        excluded = {n for n in universe - target if min(math.dist(centres[n], centres[t]) for t in target) < radius}
        if excluded & labelled != guard:
            raise ValueError("Official camera guard differs from frozen semantic fold")
        rgb_train = universe - target - excluded
        if rgb_train & labelled != train:
            raise ValueError("RGB/semantic fitting disagreement")
        output.append({"fold": fold["fold"], "rgb_train": sorted(rgb_train),
                       "semantic_train": sorted(train), "evaluation": sorted(target),
                       "rgb_guard": sorted(excluded), "semantic_guard": sorted(guard),
                       "unlabelled_rgb_train": sorted(rgb_train - labelled),
                       "initialization_observation_views": sorted(rgb_train),
                       "feature_teacher_views": sorted(rgb_train),
                       "reference_rgb_views": sorted(train), "reference_label_views": sorted(train),
                       "minimum_train_target_distance": min(math.dist(centres[n], centres[t]) for n in rgb_train for t in target)})
    if all_targets != labelled:
        raise ValueError("Outer predictions must cover all training labels exactly once")
    return output


def validate_component_access(fold, component, accessed):
    roles = {"rgb": "rgb_train", "initialization": "initialization_observation_views",
             "features": "feature_teacher_views", "semantics": "semantic_train",
             "reference_rgb": "reference_rgb_views", "reference_labels": "reference_label_views"}
    if component not in roles:
        raise ValueError("Unknown component access role")
    names = unique_names(accessed, component)
    if not names <= set(fold[roles[component]]):
        raise ValueError(f"Forbidden {component} input")


def validation_matrix():
    return [{"system": system, "kind": kind, "index": index,
             "trial_seed": index if kind == "seed" else 0,
             "outer_fold": index if kind == "spatial" else None}
            for system in ("scaffold_matched", "new_backbone_finalist")
            for kind in ("seed", "spatial") for index in range(3)]


def write_new_json(path: Path, record):
    """Exclusive creation: an existing evidence file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(record, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
