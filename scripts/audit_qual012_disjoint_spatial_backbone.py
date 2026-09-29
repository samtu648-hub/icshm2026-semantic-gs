#!/usr/bin/env python3
"""Aggregate sealed AbsGS-versus-Standard evidence across spatial folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EID = "QUAL-012-DISJOINT-SPATIAL-BACKBONE-AUDIT"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
PROTOCOL = ROOT / "docs/experiments/qual012_disjoint_spatial_backbone_audit_protocol.md"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb(result: dict[str, Any], role: str) -> dict[str, float]:
    values = result["metrics"][role]
    return {
        "psnr_db": values["PSNR_dB"]["mean"],
        "ssim": values["SSIM"]["mean"],
        "lpips_alex": values["LPIPS_Alex"]["mean"],
    }


def semantic(result: dict[str, Any], role: str) -> dict[str, float]:
    values = result["metrics"][role]
    cable = values["per_class"]["stay_cable"]
    return {
        "five_class_miou": values["mIoU_with_background"],
        "foreground_miou": values["mIoU_without_background"],
        "stay_cable_iou": cable["IoU"],
        "stay_cable_precision": cable["precision"],
        "stay_cable_recall": cable["recall"],
        "boundary_f1": result["boundary_metrics"][role]["boundary_f1"],
    }


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left}


def ply_vertices(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"No vertex count in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/quality/QUAL-012/r1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = read(CONFIG)
    paths = {key: ROOT / value for key, value in config["inputs"].items()}
    raw = {key: read(path) for key, path in paths.items()}
    result_inputs = [value for key, value in raw.items() if not key.endswith("_train")]
    if any(value["status"] not in {"completed", "completed_seed_replicated_absgs_advantage"} for value in result_inputs):
        raise ValueError("Every sealed evaluation input must be completed")
    for key in ["v2_semantic_absgs", "v2_semantic_standard"]:
        value = raw[key]
        if value["checkpoint_sha256_before_holdout_access"] != value["checkpoint_sha256_after_holdout_access"]:
            raise ValueError(f"Checkpoint changed during {key}")

    v1_rgb_delta = subtract(rgb(raw["v1_rgb_absgs"], "holdout"), rgb(raw["v1_rgb_standard"], "holdout"))
    v2_rgb_delta = subtract(rgb(raw["v2_rgb_absgs"], "holdout"), rgb(raw["v2_rgb_standard"], "holdout"))
    roles = ["train", "holdout", "region-1", "region-2", "region-3"]
    v2_semantic = {
        role: subtract(semantic(raw["v2_semantic_absgs"], role), semantic(raw["v2_semantic_standard"], role))
        for role in roles
    }
    v1_semantic = raw["v1_semantic_seed_audit"]["two_seed_means"]["absgs_minus_standard"]
    gates = {
        "v2_confirmation_psnr": v2_rgb_delta["psnr_db"] > 0,
        "v2_confirmation_five_class_miou": v2_semantic["holdout"]["five_class_miou"] > 0,
        "v2_confirmation_stay_cable_iou": v2_semantic["holdout"]["stay_cable_iou"] > 0,
        "v2_confirmation_boundary_f1": v2_semantic["holdout"]["boundary_f1"] > 0,
    }
    all_gates_pass = all(gates.values())
    ply = {
        arm: ROOT / f"outputs/quality/{eid}/full-r1/model/point_cloud/iteration_30000/point_cloud.ply"
        for arm, eid in {"absgs": "RGB-154", "standard": "RGB-155"}.items()
    }
    costs = {
        "absgs": {
            "training_seconds": raw["v2_rgb_absgs_train"]["seconds"],
            "gaussians": ply_vertices(ply["absgs"]),
            "ply_bytes": ply["absgs"].stat().st_size,
            "render_median_seconds": raw["v2_rgb_absgs"]["render_median_seconds"],
        },
        "standard": {
            "training_seconds": raw["v2_rgb_standard_train"]["seconds"],
            "gaussians": ply_vertices(ply["standard"]),
            "ply_bytes": ply["standard"].stat().st_size,
            "render_median_seconds": raw["v2_rgb_standard"]["render_median_seconds"],
        },
    }
    costs["absgs_over_standard_fraction"] = {
        key: costs["absgs"][key] / costs["standard"][key] - 1
        for key in costs["absgs"]
    }
    result = {
        "schema_version": 1,
        "status": "completed_cross_fold_gate_passed" if all_gates_pass else "completed_cross_fold_gate_failed",
        "experiment_id": EID,
        "git_commit": config["git_commit"],
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "zero_confirmation_overlap": True,
        "v1": {"rgb_confirmation_delta": v1_rgb_delta, "semantic_two_seed_confirmation_delta": v1_semantic["holdout"]},
        "v2": {"rgb_confirmation_delta": v2_rgb_delta, "semantic_delta": v2_semantic},
        "costs": costs,
        "pre_registered_v2_gates": gates,
        "all_gates_pass": all_gates_pass,
        "decision": {
            "recommendation": "select_absgs_as_final_reconstruction_backbone" if all_gates_pass else "do_not_freeze_backbone_choice",
            "claim_limit": "zero-overlap but historically exposed development folds; not blind evaluation",
        },
        "claim_scope": config["scope"],
    }
    args.output.mkdir(parents=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
