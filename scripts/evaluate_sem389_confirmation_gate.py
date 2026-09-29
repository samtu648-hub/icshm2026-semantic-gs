#!/usr/bin/env python3
"""Evaluate the frozen two-seed quantitative SEM389 confirmation gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


OLD = ROOT / "outputs/quality/SEM-386/r2/comparison/result.json"
NEW = ROOT / "outputs/quality/SEM-389/r1/comparison/result.json"
OUTPUT = ROOT / "outputs/quality/SEM-389/r1/confirmation/quantitative-gate.json"
GROUPS = ("regular-local", "oblique-local", "distant-local")


def seed_gate(result: dict[str, Any]) -> dict[str, Any]:
    comparisons = result["comparisons"]
    holdout = comparisons["holdout"]
    class_deltas = holdout["per_class_delta_pp"]
    group_deltas = {
        group: comparisons[group]["five_class_delta_pp"] for group in GROUPS
    }
    checks = {
        "five_class_nonnegative": holdout["five_class_delta_pp"] >= 0.0,
        "foreground_strictly_positive": holdout["foreground_delta_pp"] > 0.0,
        "stay_cable_boundary_nonnegative": holdout["boundary_f1_delta_pp"] >= 0.0,
        "overall_classes_above_minus_0_20_pp": min(class_deltas.values()) >= -0.20,
        "group_five_class_above_minus_0_20_pp": min(group_deltas.values()) >= -0.20,
    }
    return {
        "seed": result["seed"],
        "five_class_delta_pp": holdout["five_class_delta_pp"],
        "foreground_delta_pp": holdout["foreground_delta_pp"],
        "boundary_f1_delta_pp": holdout["boundary_f1_delta_pp"],
        "overall_per_class_delta_pp": class_deltas,
        "group_five_class_delta_pp": group_deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def confirmation_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != 2 or {row["seed"] for row in results} != {20260805, 20260919}:
        raise ValueError(
            "The frozen confirmation requires exactly seeds 20260805 and 20260919"
        )
    seeds = [seed_gate(row) for row in sorted(results, key=lambda row: row["seed"])]
    deltas = [row["five_class_delta_pp"] for row in seeds]
    cross_seed = {
        "both_five_class_nonnegative": all(value >= 0.0 for value in deltas),
        "mean_five_class_strictly_positive": sum(deltas) / len(deltas) > 0.0,
    }
    quantitative_passed = all(row["passed"] for row in seeds) and all(
        cross_seed.values()
    )
    return {
        "seeds": seeds,
        "mean_five_class_delta_pp": sum(deltas) / len(deltas),
        "cross_seed_checks": cross_seed,
        "quantitative_passed": quantitative_passed,
        "visual_review_required": True,
        "migration_authorized": False,
    }


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    if old["status"] != "completed" or new["status"] != "completed":
        raise ValueError("Both paired comparisons must be complete")
    result = {
        "schema_version": 1,
        "experiment_id": "SEM-389-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY",
        "status": "quantitative_gate_evaluated_visual_pending",
        "old_comparison_sha256": sha256(OLD),
        "new_comparison_sha256": sha256(NEW),
        **confirmation_gate([old, new]),
    }
    OUTPUT.parent.mkdir(exist_ok=False)
    write_new_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
