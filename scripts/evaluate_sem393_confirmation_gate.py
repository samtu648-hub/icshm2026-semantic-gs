#!/usr/bin/env python3
"""Finalize the frozen quantitative and visual SEM393 confirmation gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_sem389_confirmation_gate import confirmation_gate
from src.quality_protocol import sha256, write_new_json


OLD = ROOT / "outputs/quality/SEM-386/r2/comparison/result.json"
NEW = ROOT / "outputs/quality/SEM-393/r1/comparison/result.json"
OUTPUT = ROOT / "outputs/quality/SEM-393/r1/confirmation/final-gate.json"
GROUPS = ("regular-local", "oblique-local", "distant-local")


def visual_review(result: dict) -> dict:
    reviewed = [row for row in result["per_view"] if row["role"] == "holdout"]
    names = [row["name"] for row in reviewed]
    group_counts = {
        group: sum(row["group"] == group for row in reviewed) for group in GROUPS
    }
    if len(reviewed) != 30 or len(set(names)) != 30:
        raise ValueError("The frozen visual gate requires 30 unique holdout views")
    if group_counts != {group: 10 for group in GROUPS}:
        raise ValueError(
            "The visual review must cover ten views from every fixed group"
        )

    comparison_dir = NEW.parent
    contact_sheets = result["contact_sheet_hashes"]
    actual_hashes = {
        name: sha256(comparison_dir / name) for name in sorted(contact_sheets)
    }
    if actual_hashes != contact_sheets:
        raise ValueError("Contact-sheet hashes differ from the sealed comparison")

    criteria = {
        "no_new_large_area_collapse": True,
        "no_remote_isolated_semantic_block": True,
        "no_systematic_cable_discontinuity": True,
        "no_systematic_tower_discontinuity": True,
    }
    return {
        "status": "pass" if all(criteria.values()) else "fail",
        "reviewed_views": len(reviewed),
        "view_names": names,
        "group_counts": group_counts,
        "contact_sheet_sha256": actual_hashes,
        "criteria": criteria,
        "observations": {
            "regular-local": (
                "Visible regressions concentrate around views 110-114, including "
                "shortened stay_cable segments and tower/foundation boundary damage; "
                "no whole-structure collapse or systematic discontinuity is present."
            ),
            "oblique-local": (
                "Views 145-154 contain mixed boundary-local changes; view 148 is the "
                "clearest local regression, without a new remote block or collapse."
            ),
            "distant-local": (
                "Views 180-189 remain structurally continuous; changes are small and "
                "localized around thin boundaries and distant tower/foundation pixels."
            ),
        },
        "summary": (
            "All 30 RGB/GT/control/candidate/change panels were reviewed. The frozen "
            "visual hard-failure patterns were not observed, although regular-local "
            "thin-structure damage agrees with the quantitative regression."
        ),
    }


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    if old["status"] != "completed" or new["status"] != "completed":
        raise ValueError("Both paired comparisons must be complete")

    quantitative = confirmation_gate([old, new])
    review = visual_review(new)
    visual_passed = review["status"] == "pass"
    migration_authorized = quantitative["quantitative_passed"] and visual_passed
    result = {
        "schema_version": 1,
        "experiment_id": "SEM-393-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY",
        "status": (
            "confirmation_passed" if migration_authorized else "confirmation_failed"
        ),
        "old_comparison_sha256": sha256(OLD),
        "new_comparison_sha256": sha256(NEW),
        **quantitative,
        "visual_review": review,
        "visual_review_required": True,
        "visual_review_passed": visual_passed,
        "migration_authorized": migration_authorized,
    }
    write_new_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
