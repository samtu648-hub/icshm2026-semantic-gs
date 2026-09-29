#!/usr/bin/env python3
"""Finalize the frozen quantitative and visual SEM391 confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_sem389_confirmation_gate import confirmation_gate
from src.quality_protocol import sha256, write_new_json


OLD = ROOT / "outputs/quality/SEM-386/r2/comparison/result.json"
NEW = ROOT / "outputs/quality/SEM-391/r1/comparison/result.json"
OUTPUT = ROOT / "outputs/quality/SEM-391/r1/confirmation/final-gate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-status", choices=("pass", "fail"), required=True)
    parser.add_argument("--reviewed-views", type=int, required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    if args.reviewed_views != 30:
        raise ValueError("The frozen visual gate requires review of all 30 views")

    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    if old["status"] != "completed" or new["status"] != "completed":
        raise ValueError("Both paired comparisons must be complete")
    quantitative = confirmation_gate([old, new])
    visual_passed = args.visual_status == "pass"
    migration_authorized = quantitative["quantitative_passed"] and visual_passed
    result = {
        "schema_version": 1,
        "experiment_id": "SEM-391-RGB150-DEV270-CONTEXT50-SEED20260919-MEMORY",
        "status": "confirmation_passed"
        if migration_authorized
        else "confirmation_failed",
        "old_comparison_sha256": sha256(OLD),
        "new_comparison_sha256": sha256(NEW),
        **quantitative,
        "visual_review": {
            "status": args.visual_status,
            "reviewed_views": args.reviewed_views,
            "summary": args.summary,
            "criteria": [
                "no new large-area collapse",
                "no remote isolated semantic blocks",
                "no systematic cable discontinuity",
                "no systematic tower discontinuity",
            ],
        },
        "visual_review_required": True,
        "visual_review_passed": visual_passed,
        "migration_authorized": migration_authorized,
    }
    OUTPUT.parent.mkdir(exist_ok=False)
    write_new_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
