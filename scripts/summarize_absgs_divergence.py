#!/usr/bin/env python3
"""Build a compact, data-only QUAL018 causal evidence summary."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/quality/QUAL-018/r1"


def read(path):
    return json.loads(Path(path).read_text())


def metric(path):
    value = read(path)
    return {
        "train": {k: v["mean"] for k, v in value["metrics"]["train"].items()},
        "holdout": {k: v["mean"] for k, v in value["metrics"]["holdout"].items()},
        "regions": {
            name: {k: v["mean"] for k, v in metrics.items()}
            for name, metrics in value["group_metrics"].items()
        },
    }


def contribution_summary(records):
    output = []
    for record in records:
        top = record["top_contributors"]
        first32 = top[:32]
        births = Counter(
            str(point["ancestry"][0]["birth_step"] if point["ancestry"] else 0)
            for point in first32
        )
        output.append(
            {
                "step": record["step"],
                "view": record["view"],
                "baseline_psnr_db": record["baseline_psnr_db"],
                "removal_delta_psnr_db": {
                    str(row["top_k"]): row["delta_psnr_db"]
                    for row in record["removals"]
                },
                "all_removals_restored_exact": all(
                    row["restored_exact"] for row in record["removals"]
                ),
                "top32_current_birth_steps": dict(
                    sorted(births.items(), key=lambda item: int(item[0]))
                ),
                "top32_initial_ancestor_count": len(
                    {point["initial_ancestor"] for point in first32}
                ),
                "top32_radius_gt20_count": sum(
                    point["radius_px"] > 20 for point in first32
                ),
                "top32_max_radius_px": max(point["radius_px"] for point in first32),
                "top32_max_scale": max(max(point["scales"]) for point in first32),
                "top32_median_opacity": statistics.median(
                    point["opacity"] for point in first32
                ),
                "top1_fraction_of_mass": top[0]["fraction_of_mass"],
                "top32_fraction_of_mass": sum(
                    point["fraction_of_mass"] for point in first32
                ),
            }
        )
    return output


def main():
    analysis = read(OUTPUT / "analysis-result.json")
    independent = read(OUTPUT / "independent-audit.json")
    trajectory = analysis["trajectory_comparison"]
    probes = {}
    for arm, rows in trajectory["probe_curves"].items():
        probes[arm] = [row for row in rows if row["view"] in ("156.png", "159.png")]
    numerical = {}
    for arm, rows in analysis["numerical_probes"].items():
        numerical[arm] = []
        for row in rows:
            trials = row["trials"]
            numerical[arm].append(
                {
                    "step": row["step"],
                    "camera": row["camera"],
                    "forward_exact": row["forward_exact"],
                    "backward_exact": row["backward_exact"],
                    "max_xyz_gradient_difference": max(
                        trial["xyz_max_abs_difference_from_first"] for trial in trials
                    ),
                    "max_screen_gradient_difference": max(
                        trial["screen_max_abs_difference_from_first"]
                        for trial in trials
                    ),
                    "selection_differences": [
                        trial.get("selection_differences_from_first")
                        for trial in trials[1:]
                    ],
                    "rng_unchanged": all(trial["rng_unchanged"] for trial in trials),
                }
            )
    evaluations = {
        "RGB-152-historical": metric(
            ROOT / "outputs/quality/RGB-152/full-r2/evaluation/result.json"
        ),
        "RGB-156-native-new-control": metric(
            ROOT / "outputs/quality/RGB-156/full-r1/evaluation/result.json"
        ),
        "RGB-158-native-replay": metric(
            ROOT / "outputs/quality/RGB-158/full-r1/evaluation/result.json"
        ),
        "RGB-159-trace-A": metric(
            ROOT / "outputs/quality/RGB-159/full-r1/evaluation/result.json"
        ),
        "RGB-160-trace-B": metric(
            ROOT / "outputs/quality/RGB-160/full-r1/evaluation/result.json"
        ),
    }
    summary = {
        "status": "completed_data_summary",
        "trajectory_identity": {
            key: trajectory[key]
            for key in (
                "initial_parameters_exact",
                "initial_rng_exact",
                "runtime_exact",
                "camera_sequence_exact",
                "loaded_image_tensors_exact",
                "camera_tensor_metadata_exact",
            )
        },
        "first_differences": trajectory["first_differences"],
        "first_topology_selection_difference": trajectory[
            "first_topology_selection_difference"
        ],
        "first_prune_difference": trajectory["first_prune_difference"],
        "screen_pruning": trajectory["screen_pruning"],
        "numerical_probes": numerical,
        "endpoint_metrics": evaluations,
        "paired_endpoint_delta_b_minus_a": {
            role: {
                key: evaluations["RGB-160-trace-B"][role][key]
                - evaluations["RGB-159-trace-A"][role][key]
                for key in evaluations["RGB-159-trace-A"][role]
            }
            for role in ("train", "holdout")
        },
        "difficult_probe_curves": probes,
        "contributions": {
            arm: contribution_summary(rows)
            for arm, rows in analysis["contributions"].items()
        },
        "independent_audit_sha256": __import__("hashlib")
        .sha256((OUTPUT / "independent-audit.json").read_bytes())
        .hexdigest(),
        "independent_audit_status": independent["status"],
    }
    destination = OUTPUT / "causal-summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
