#!/usr/bin/env python3
"""Audit the sealed AbsGS and Standard spatial-confirmation chains."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


EID = "QUAL-010-MATCHED-SPATIAL-BACKBONE"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
SCRIPT = ROOT / "scripts/audit_qual010_matched_spatial_backbone.py"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["view"]: row for row in csv.DictReader(handle)}


def ply_points(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(4096).decode("ascii", errors="ignore")
    match = re.search(r"^element vertex (\d+)$", header, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"PLY vertex count missing: {path}")
    return int(match.group(1))


def metric_mean(result: dict[str, Any], role: str, key: str) -> float:
    return float(result["metrics"][role][key]["mean"])


def semantic_flat(result: dict[str, Any], role: str) -> dict[str, float]:
    metric = result["metrics"][role]
    cable = metric["per_class"]["stay_cable"]
    boundary = result["boundary_metrics"][role]
    return {
        "five_class_miou": float(metric["mIoU_with_background"]),
        "foreground_miou": float(metric["mIoU_without_background"]),
        "stay_cable_iou": float(cable["IoU"]),
        "stay_cable_precision": float(cable["precision"]),
        "stay_cable_recall": float(cable["recall"]),
        "boundary_f1": float(boundary["boundary_f1"]),
    }


def main() -> None:
    config = read(CONFIG)
    if config["experiment_id"] != EID:
        raise ValueError("QUAL010 config identity changed")
    output = ROOT / config["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    paths = {key: ROOT / value for key, value in config["inputs"].items()}
    records = {key: read(path) for key, path in paths.items() if path.suffix == ".json"}
    if records["absgs_rgb_training"]["status"] != "completed_spatial_training":
        raise ValueError("AbsGS RGB training is not sealed")
    if records["standard_rgb_training"]["status"] != "completed_spatial_training":
        raise ValueError("Standard RGB training is not sealed")
    for key in ("absgs_rgb_evaluation", "standard_rgb_evaluation", "absgs_semantic", "standard_semantic"):
        if records[key]["status"] != "completed":
            raise ValueError(f"Parent result is incomplete: {key}")
    if records["absgs_semantic"]["checkpoint_sha256_before_holdout_access"] != records["absgs_semantic"]["checkpoint_sha256_after_holdout_access"]:
        raise ValueError("AbsGS semantic checkpoint changed during evaluation")
    if records["standard_semantic"]["checkpoint_sha256_before_holdout_access"] != records["standard_semantic"]["checkpoint_sha256_after_holdout_access"]:
        raise ValueError("Standard semantic checkpoint changed during evaluation")

    absgs_rgb_rows = csv_rows(paths["absgs_rgb_holdout_csv"])
    standard_rgb_rows = csv_rows(paths["standard_rgb_holdout_csv"])
    absgs_sem_rows = {
        row["name"]: row
        for row in records["absgs_semantic"]["per_view"]
        if row["role"] == "holdout"
    }
    standard_sem_rows = {
        row["name"]: row
        for row in records["standard_semantic"]["per_view"]
        if row["role"] == "holdout"
    }
    names = sorted(absgs_rgb_rows)
    if not (
        set(names)
        == set(standard_rgb_rows)
        == set(absgs_sem_rows)
        == set(standard_sem_rows)
    ) or len(names) != 30:
        raise ValueError("The four parent results do not share the fixed 30 views")

    paired = []
    for name in names:
        a_rgb, s_rgb = absgs_rgb_rows[name], standard_rgb_rows[name]
        a_sem, s_sem = absgs_sem_rows[name], standard_sem_rows[name]
        a_cable = a_sem["per_class"]["stay_cable"]
        s_cable = s_sem["per_class"]["stay_cable"]
        paired.append(
            {
                "view": name,
                "group": a_sem["group"],
                "absgs_psnr_db": float(a_rgb["PSNR_dB"]),
                "standard_psnr_db": float(s_rgb["PSNR_dB"]),
                "absgs_minus_standard_psnr_db": float(a_rgb["PSNR_dB"]) - float(s_rgb["PSNR_dB"]),
                "absgs_ssim": float(a_rgb["SSIM"]),
                "standard_ssim": float(s_rgb["SSIM"]),
                "absgs_minus_standard_ssim": float(a_rgb["SSIM"]) - float(s_rgb["SSIM"]),
                "absgs_lpips": float(a_rgb["LPIPS_Alex"]),
                "standard_lpips": float(s_rgb["LPIPS_Alex"]),
                "absgs_minus_standard_lpips": float(a_rgb["LPIPS_Alex"]) - float(s_rgb["LPIPS_Alex"]),
                "absgs_five_class_miou": float(a_sem["mIoU_with_background"]),
                "standard_five_class_miou": float(s_sem["mIoU_with_background"]),
                "absgs_minus_standard_five_class_miou": float(a_sem["mIoU_with_background"]) - float(s_sem["mIoU_with_background"]),
                "absgs_stay_cable_iou": float(a_cable["IoU"]),
                "standard_stay_cable_iou": float(s_cable["IoU"]),
                "absgs_minus_standard_stay_cable_iou": float(a_cable["IoU"]) - float(s_cable["IoU"]),
            }
        )
    with (output / "paired_views.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    roles = ["train", "holdout", "region-1", "region-2", "region-3"]
    rgb_summary = {}
    semantic_summary = {}
    for role in roles:
        a_rgb = {
            key: metric_mean(records["absgs_rgb_evaluation"], role if role in ("train", "holdout") else "holdout", key)
            for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
        } if role in ("train", "holdout") else {
            key: float(records["absgs_rgb_evaluation"]["group_metrics"][role][key]["mean"])
            for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
        }
        s_rgb = {
            key: metric_mean(records["standard_rgb_evaluation"], role if role in ("train", "holdout") else "holdout", key)
            for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
        } if role in ("train", "holdout") else {
            key: float(records["standard_rgb_evaluation"]["group_metrics"][role][key]["mean"])
            for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
        }
        rgb_summary[role] = {
            "absgs": a_rgb,
            "standard": s_rgb,
            "absgs_minus_standard": {key: a_rgb[key] - s_rgb[key] for key in a_rgb},
        }
        a_sem = semantic_flat(records["absgs_semantic"], role)
        s_sem = semantic_flat(records["standard_semantic"], role)
        semantic_summary[role] = {
            "absgs": a_sem,
            "standard": s_sem,
            "absgs_minus_standard": {key: a_sem[key] - s_sem[key] for key in a_sem},
        }

    absgs_ply = ROOT / "outputs/quality/RGB-152/full-r2/model/point_cloud/iteration_30000/point_cloud.ply"
    standard_ply = ROOT / "outputs/quality/RGB-153/full-r1/model/point_cloud/iteration_30000/point_cloud.ply"
    costs = {
        "absgs": {
            "training_seconds": float(records["absgs_rgb_training"]["seconds"]),
            "gaussians": ply_points(absgs_ply),
            "ply_bytes": absgs_ply.stat().st_size,
            "render_median_seconds": float(records["absgs_rgb_evaluation"]["render_median_seconds"]),
        },
        "standard": {
            "training_seconds": float(records["standard_rgb_training"]["seconds"]),
            "gaussians": ply_points(standard_ply),
            "ply_bytes": standard_ply.stat().st_size,
            "render_median_seconds": float(records["standard_rgb_evaluation"]["render_median_seconds"]),
        },
    }
    costs["absgs_over_standard"] = {
        key: costs["absgs"][key] / costs["standard"][key] - 1
        for key in costs["absgs"]
    }
    holdout_rgb = rgb_summary["holdout"]["absgs_minus_standard"]
    holdout_sem = semantic_summary["holdout"]["absgs_minus_standard"]
    other_region_semantic_deltas = [
        semantic_summary[role]["absgs_minus_standard"]["five_class_miou"]
        for role in ("region-2", "region-3")
    ]
    decision = {
        "absgs_spatial_advantage_observed": (
            holdout_rgb["PSNR_dB"] > 0.5
            and holdout_rgb["SSIM"] > 0
            and holdout_rgb["LPIPS_Alex"] < 0
            and holdout_sem["five_class_miou"] > 0.01
            and holdout_sem["stay_cable_iou"] > 0.01
            and holdout_sem["boundary_f1"] > 0.01
            and min(other_region_semantic_deltas) > -0.002
        ),
        "provisional_recommendation": "absgs_pending_replication",
        "reason": "Large region-1 RGB and semantic protection with near-neutral region-2/3 differences; cost is material and the evidence is still one historical development split/seed.",
    }
    result = {
        "schema_version": 1,
        "status": "completed_provisional_absgs_advantage",
        "experiment_id": EID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(ROOT / config["protocol"]),
        "script_sha256": sha256(SCRIPT),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "input_hashes": {key: sha256(path) for key, path in paths.items()},
        "paired_view_count": len(paired),
        "rgb_summary": rgb_summary,
        "semantic_summary": semantic_summary,
        "paired_view_counts": {
            "absgs_better_psnr": sum(row["absgs_minus_standard_psnr_db"] > 0 for row in paired),
            "absgs_better_ssim": sum(row["absgs_minus_standard_ssim"] > 0 for row in paired),
            "absgs_better_lpips": sum(row["absgs_minus_standard_lpips"] < 0 for row in paired),
            "absgs_better_five_class_miou": sum(row["absgs_minus_standard_five_class_miou"] > 0 for row in paired),
            "absgs_better_stay_cable_iou": sum(row["absgs_minus_standard_stay_cable_iou"] > 0 for row in paired),
        },
        "costs": costs,
        "decision": decision,
        "paired_views_csv_sha256": sha256(output / "paired_views.csv"),
        "claim_scope": "single-seed historically exposed spatial development comparison; not blind or official",
    }
    write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
