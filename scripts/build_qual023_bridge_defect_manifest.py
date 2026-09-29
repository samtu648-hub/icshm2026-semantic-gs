#!/usr/bin/env python3
"""Build an exclusive no-support/wrong-support manifest from QUAL022 evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_defect_manifest import aggregate_gaussian_records, classify_view_structure
from src.quality_protocol import sha256, write_new_json

EXPERIMENT_ID = "QUAL-023-BRIDGE-DEFECT-MANIFEST"
CONFIG = ROOT / "experiments/runs" / EXPERIMENT_ID / "config.yaml"
PARENT = ROOT / "outputs/quality/QUAL-022/diagnostic-r1"
PLY170 = ROOT / "outputs/quality/RGB-170/full-r1/model/point_cloud/iteration_50000/point_cloud.ply"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-023/manifest-r1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text())
    per_view = json.loads((PARENT / "per-view.json").read_text())["rows"]
    rows = [row for row in per_view if row["class"] in {"deck", "stay_cable", "tower"}]
    if len(rows) != 900:
        raise ValueError(f"Expected 900 view/class rows, got {len(rows)}")
    for row in rows:
        row["defect"] = classify_view_structure(
            row,
            minimum_projected_coverage=config["classification"]["minimum_projected_coverage"],
            maximum_nonclean_fraction=config["classification"]["maximum_nonclean_fraction"],
        )

    contributors = []
    with gzip.open(PARENT / "contributors.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            contributors.append(json.loads(line))
    mixed_gaussians = aggregate_gaussian_records(
        contributors,
        minimum_views=config["classification"]["minimum_cross_view_observations"],
        minimum_mixed_fraction=config["classification"]["minimum_mixed_observation_fraction"],
        minimum_background_fraction=config["classification"]["minimum_near_background_fraction"],
    )

    counts = Counter(row["defect"] for row in rows)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_class[row["class"]][row["defect"]] += 1
    with gzip.open(args.output / "mixed-gaussians.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in mixed_gaussians:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_new_json(args.output / "view-structure-manifest.json", {"rows": rows})
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_read_only",
        "read_only": True,
        "config_sha256": sha256(CONFIG),
        "parent_result_sha256": sha256(PARENT / "result.json"),
        "parent_contributors_sha256": sha256(PARENT / "contributors.jsonl.gz"),
        "rgb170_ply_sha256": sha256(PLY170),
        "view_structure_rows": len(rows),
        "exclusive_counts": dict(sorted(counts.items())),
        "exclusive_counts_by_class": {key: dict(sorted(value.items())) for key, value in sorted(by_class.items())},
        "mixed_gaussians": len(mixed_gaussians),
        "mixed_gaussians_by_class": dict(sorted(Counter(row["class"] for row in mixed_gaussians).items())),
        "criteria": config["classification"],
        "interpretation_scope": (
            "Final RGB170 training-view contribution/projection evidence only. Labels localize 2D structures; "
            "the manifest creates no 3D points and makes no unknown-view claim."
        ),
    }
    write_new_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()

