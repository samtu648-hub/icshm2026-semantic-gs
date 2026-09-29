#!/usr/bin/env python3
"""Promote the sealed SEM395 camera-only candidate to a registered split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/quality/SEM-395/r2/prepare/confirmation-split-candidate.json"
OUTPUT = ROOT / "experiments/splits/spatial_confirmation_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: dict[str, object], source_hash: str) -> dict[str, object]:
    confirmation = list(source["confirmation_labeled"])
    buffer_all = list(source["buffer_excluded_all_cameras"])
    buffer_labeled = list(source["buffer_excluded_labeled"])
    buffer_unlabeled = list(source["buffer_excluded_unlabeled"])
    train_rgb = list(source["train_rgb_400"])
    train_semantic = list(source["train_rgb_300"])
    regions = list(source["confirmation_regions"])
    if sorted(name for region in regions for name in region["images"]) != sorted(confirmation):
        raise ValueError("Confirmation regions do not exactly cover the sealed views")
    roles = [set(confirmation), set(buffer_all), set(train_rgb)]
    if any(roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("RGB roles overlap")
    if len(set().union(*roles)) != 400:
        raise ValueError("RGB roles do not cover all 400 released cameras")
    if set(train_semantic) - set(train_rgb):
        raise ValueError("Semantic training views are outside RGB training")
    if set(buffer_labeled) | set(buffer_unlabeled) != set(buffer_all):
        raise ValueError("Buffer subclasses do not cover the buffer")
    checks = dict(source["checks"])
    return {
        "schema_version": 1,
        "split_id": "spatial_confirmation_v1",
        "purpose": "Once-only spatial confirmation of a fresh reconstruction and semantic chain",
        "claim_scope": "Historically exposed spatial development confirmation; not a blind test or official hidden evaluation",
        "source_coordinate_frame": "official_colmap_reference",
        "pose_convention": "CAMERA_FROM_WORLD",
        "selection_protocol": {
            "source": "SEM395 camera-only future confirmation candidate",
            "source_path": "outputs/quality/SEM-395/r2/prepare/confirmation-split-candidate.json",
            "source_sha256": source_hash,
            "selection_uses_model_outputs_or_labels": False,
            "seed": source["seed"],
            "guard_radius_strict_less_than": source["guard_radius_strict_less_than"],
            "history_warning": "All released views participated in earlier research models; this split is spatially guarded for the new chain but not historically unseen.",
        },
        "groups": [
            {
                "group_id": region["region_id"],
                "segment_id": region["segment_id"],
                "description": "Contiguous ten-view spatial confirmation region",
                "images": region["images"],
            }
            for region in regions
        ],
        "class_mapping": {
            "0": "background",
            "1": "deck",
            "2": "stay_cable",
            "3": "tower",
            "4": "foundation",
        },
        "primary_fold_id": "spatial-confirmation-primary",
        "primary_fold": {
            "holdout_all_cameras": confirmation,
            "validation_labeled": confirmation,
            "withheld_unlabeled": [],
            "buffer_excluded_all_cameras": buffer_all,
            "buffer_excluded_labeled": buffer_labeled,
            "buffer_excluded_unlabeled": buffer_unlabeled,
            "train_rgb_400": train_rgb,
            "train_rgb_300": train_semantic,
            "semantic_train_labeled": train_semantic,
            "semantic_validation_labeled": confirmation,
            "checks": {
                "disjoint_roles": checks["disjoint_roles"],
                "all_400_accounted": checks["all_400_accounted"],
                "confirmation_count": len(confirmation),
                "buffer_count": len(buffer_all),
                "rgb_train_count": len(train_rgb),
                "semantic_train_count": len(train_semantic),
                "camera_graph_component_count_not_increased": checks[
                    "camera_graph_component_count_not_increased"
                ],
            },
        },
        "global_checks": checks,
        "execution_contract": source["future_requirement"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    result = build(source, sha256(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
