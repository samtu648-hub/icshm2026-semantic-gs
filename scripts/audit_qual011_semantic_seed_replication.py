#!/usr/bin/env python3
"""Aggregate the two sealed semantic-seed backbone comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EID = "QUAL-011-SEMANTIC-SEED-REPLICATION-AUDIT"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
PROTOCOL = ROOT / "docs/experiments/qual011_semantic_seed_replication_audit_protocol.md"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(result: dict[str, Any], role: str) -> dict[str, float]:
    metrics = result["metrics"][role]
    cable = metrics["per_class"]["stay_cable"]
    boundary = result["boundary_metrics"][role]
    return {
        "five_class_miou": metrics["mIoU_with_background"],
        "foreground_miou": metrics["mIoU_without_background"],
        "stay_cable_iou": cable["IoU"],
        "stay_cable_precision": cable["precision"],
        "stay_cable_recall": cable["recall"],
        "boundary_f1": boundary["boundary_f1"],
    }


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left}


def mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-011/r1")
    args = parser.parse_args()
    config = read(CONFIG)
    paths = {key: ROOT / value for key, value in config["inputs"].items()}
    if args.output.exists():
        raise FileExistsError(args.output)

    raw = {key: read(path) for key, path in paths.items()}
    if any(result["status"] != "completed" for result in raw.values()):
        raise ValueError("All four sealed inference results must be completed")
    if any(
        result["checkpoint_sha256_before_holdout_access"]
        != result["checkpoint_sha256_after_holdout_access"]
        for result in raw.values()
    ):
        raise ValueError("A checkpoint changed after confirmation access")

    roles = ["train", "holdout", "region-1", "region-2", "region-3"]
    seeds = {
        "20260805": ("seed_20260805_absgs", "seed_20260805_standard"),
        "20260919": ("seed_20260919_absgs", "seed_20260919_standard"),
    }
    paired: dict[str, Any] = {}
    for seed, (abs_key, std_key) in seeds.items():
        abs_rows = {role: extract(raw[abs_key], role) for role in roles}
        std_rows = {role: extract(raw[std_key], role) for role in roles}
        paired[seed] = {
            "absgs": abs_rows,
            "standard": std_rows,
            "absgs_minus_standard": {
                role: subtract(abs_rows[role], std_rows[role]) for role in roles
            },
        }

    averages: dict[str, Any] = {}
    for arm in ["absgs", "standard"]:
        averages[arm] = {
            role: mean([paired[seed][arm][role] for seed in seeds]) for role in roles
        }
    averages["absgs_minus_standard"] = {
        role: subtract(averages["absgs"][role], averages["standard"][role])
        for role in roles
    }

    required = ["five_class_miou", "stay_cable_iou", "boundary_f1"]
    gates = {
        seed: {
            role: {metric: paired[seed]["absgs_minus_standard"][role][metric] > 0 for metric in required}
            for role in ["holdout", "region-1"]
        }
        for seed in seeds
    }
    all_gates_pass = all(
        passed
        for seed_gate in gates.values()
        for role_gate in seed_gate.values()
        for passed in role_gate.values()
    )
    result = {
        "schema_version": 1,
        "status": "completed_seed_replicated_absgs_advantage" if all_gates_pass else "completed_replication_gate_failed",
        "experiment_id": EID,
        "git_commit": config["git_commit"],
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "paired": paired,
        "two_seed_means": averages,
        "pre_registered_gates": gates,
        "all_gates_pass": all_gates_pass,
        "decision": {
            "semantic_seed_replication": "passed" if all_gates_pass else "failed",
            "recommendation": "retain_absgs_as_final_backbone_candidate" if all_gates_pass else "do_not_promote_from_seed_evidence",
            "remaining_limit": "single historically exposed spatial split and single RGB seed; disjoint spatial replication remains the next high-value experiment",
        },
        "claim_scope": config["scope"],
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
