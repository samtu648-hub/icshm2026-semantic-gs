#!/usr/bin/env python3
"""Audit SEM397 against its frozen SEM396 promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


CONFIG = ROOT / "experiments/runs/SEM-397-RGB152-SPATIAL256-BOUNDARY/config.yaml"
BASELINE = ROOT / "outputs/quality/SEM-396/r1/inference/result.json"
CANDIDATE = ROOT / "outputs/quality/SEM-397/r1/inference/result.json"
OUTPUT = ROOT / "outputs/quality/SEM-397/r1/comparison.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    baseline = json.loads(BASELINE.read_text())
    candidate = json.loads(CANDIDATE.read_text())
    if baseline["status"] != "completed" or candidate["status"] != "completed":
        raise ValueError("Both sealed inference results are required")

    base = baseline["metrics"]["holdout"]
    cand = candidate["metrics"]["holdout"]
    base_boundary = baseline["boundary_metrics"]["holdout"]["boundary_f1"]
    cand_boundary = candidate["boundary_metrics"]["holdout"]["boundary_f1"]
    class_names = ["background", "deck", "stay_cable", "tower", "foundation"]
    regions = ["region-1", "region-2", "region-3"]
    class_iou_delta = {
        name: cand["per_class"][name]["IoU"] - base["per_class"][name]["IoU"]
        for name in class_names
    }
    region_delta = {
        name: candidate["metrics"][name]["mIoU_with_background"]
        - baseline["metrics"][name]["mIoU_with_background"]
        for name in regions
    }
    cable_base = base["per_class"]["stay_cable"]
    cable_cand = cand["per_class"]["stay_cable"]
    gates = config["promotion_gates"]
    checks = {
        "confirmation_five_class_miou": cand["mIoU_with_background"]
        > gates["confirmation_five_class_miou_strictly_above"],
        "confirmation_boundary_f1_gain": cand_boundary - base_boundary
        >= gates["confirmation_boundary_f1_gain_at_least"],
        "confirmation_stay_cable_recall": cable_cand["recall"]
        > gates["confirmation_stay_cable_recall_strictly_above"],
        "confirmation_stay_cable_precision_retention": cable_cand["precision"]
        - cable_base["precision"]
        >= -gates["maximum_confirmation_stay_cable_precision_drop"],
        "all_confirmation_class_iou_retention": min(class_iou_delta.values())
        >= -gates["maximum_any_confirmation_class_iou_drop"],
        "all_region_five_class_miou_retention": min(region_delta.values())
        >= -gates["maximum_any_region_five_class_miou_drop"],
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "input_sha256": {
            "config": sha256(CONFIG),
            "baseline_inference": sha256(BASELINE),
            "candidate_inference": sha256(CANDIDATE),
        },
        "baseline": {
            "five_class_miou": base["mIoU_with_background"],
            "foreground_miou": base["mIoU_without_background"],
            "boundary_f1": base_boundary,
            "stay_cable": cable_base,
        },
        "candidate": {
            "five_class_miou": cand["mIoU_with_background"],
            "foreground_miou": cand["mIoU_without_background"],
            "boundary_f1": cand_boundary,
            "stay_cable": cable_cand,
        },
        "delta": {
            "five_class_miou": cand["mIoU_with_background"] - base["mIoU_with_background"],
            "foreground_miou": cand["mIoU_without_background"] - base["mIoU_without_background"],
            "boundary_f1": cand_boundary - base_boundary,
            "stay_cable_iou": cable_cand["IoU"] - cable_base["IoU"],
            "stay_cable_precision": cable_cand["precision"] - cable_base["precision"],
            "stay_cable_recall": cable_cand["recall"] - cable_base["recall"],
            "class_iou": class_iou_delta,
            "region_five_class_miou": region_delta,
        },
        "gate_checks": checks,
        "all_gates_pass": all(checks.values()),
        "decision": "promote_to_second_seed" if all(checks.values()) else "closed_not_promoted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
