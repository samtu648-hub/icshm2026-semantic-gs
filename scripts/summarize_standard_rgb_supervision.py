#!/usr/bin/env python3
"""Aggregate matched RGB147-149 fit, region, and resource evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json
from src.standard_rgb_supervision import (
    load_structure_mask,
    masked_rgb_l1,
    multiscale_gradient_l1,
)


ARMS = {
    "A_control": ROOT / "outputs/quality/rgb147/control-r1",
    "B_region_rgb": ROOT / "outputs/quality/rgb147/region-rgb-r1",
    "C_region_gradient": ROOT / "outputs/quality/rgb147/region-gradient-r1",
}
METRIC_DIRECTIONS = {"PSNR": 1, "SSIM": 1, "LPIPS": -1}


def image_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def paired_summary(base: dict[str, dict], candidate: dict[str, dict]) -> dict:
    output = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        deltas = [candidate[name][metric] - base[name][metric] for name in sorted(base)]
        output[metric] = {
            "candidate_minus_base_mean": float(np.mean(deltas)),
            "candidate_minus_base_median": float(np.median(deltas)),
            "candidate_better_views": sum(direction * value > 0 for value in deltas),
            "candidate_equal_views": sum(value == 0 for value in deltas),
            "candidate_worse_views": sum(direction * value < 0 for value in deltas),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/quality/rgb147/comparison-r1"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "status": "running",
        "scope": "all400 training-view fit; 300 released labels for region diagnostics; fixed144 trajectory has no GT",
        "selection_or_tuning_performed": False,
        "code_sha256": sha256(Path(__file__)),
    }
    try:
        evaluations = {}
        training = {}
        per_view = {}
        for arm, root in ARMS.items():
            evaluation_path = root / "evaluation/result.json"
            completed_path = root / "completed.json"
            evaluations[arm] = json.loads(evaluation_path.read_text())
            training[arm] = json.loads(completed_path.read_text())
            if (
                evaluations[arm]["status"] != "completed"
                or evaluations[arm]["train_count"] != 400
                or evaluations[arm]["trajectory_count"] != 144
                or training[arm]["status"] != "completed_all400"
            ):
                raise ValueError(f"Incomplete input arm: {arm}")
            per_view[arm] = {row["name"]: row for row in evaluations[arm]["per_view"]}
            if set(per_view[arm]) != {f"{index:03d}.png" for index in range(1, 401)}:
                raise ValueError(f"Unexpected all400 rows: {arm}")

        camera_hashes = {sha256(root / "camera_sequence.txt") for root in ARMS.values()}
        if len(camera_hashes) != 1:
            raise ValueError("A/B/C full camera sequences differ")

        labels = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
        photos = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
        region_rows = []
        region_values = {
            arm: {"rgb_l1_01": [], "multiscale_gradient_l1_01": []} for arm in ARMS
        }
        for index in range(1, 301):
            name = f"{index:03d}.png"
            photo = image_tensor(photos / name)
            mask = load_structure_mask(labels / name, dilation_px=4)
            row = {"view": name, "region_pixels": int(mask.sum())}
            for arm, root in ARMS.items():
                rendered = image_tensor(root / "evaluation/images" / name)
                metrics = {
                    "rgb_l1_01": float(masked_rgb_l1(rendered, photo, mask)),
                    "multiscale_gradient_l1_01": float(
                        multiscale_gradient_l1(rendered, photo, mask, (1, 4))
                    ),
                }
                row[arm] = metrics
                for key, value in metrics.items():
                    region_values[arm][key].append(value)
            region_rows.append(row)
            if index % 50 == 0:
                print(f"region diagnostics {index}/300", flush=True)

        global_metrics = {
            arm: evaluation["metrics"] for arm, evaluation in evaluations.items()
        }
        class_metrics = {
            arm: evaluation["class_rgb_mae_255"]
            for arm, evaluation in evaluations.items()
        }
        region_metrics = {
            arm: {
                key: {
                    "macro_mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
                for key, values in arm_values.items()
            }
            for arm, arm_values in region_values.items()
        }
        fixed_regions = {
            arm: {row["region"]: row for row in evaluation["regions"]}
            for arm, evaluation in evaluations.items()
        }
        if not (
            fixed_regions["A_control"].keys()
            == fixed_regions["B_region_rgb"].keys()
            == fixed_regions["C_region_gradient"].keys()
        ):
            raise ValueError("Fixed-region evaluation sets differ")

        resources = {}
        for arm, root in ARMS.items():
            evaluation = evaluations[arm]
            run = training[arm]
            resources[arm] = {
                "final_gaussian_count": evaluation["final_gaussian_count"],
                "model_mib": evaluation["model_mib"],
                "training_wall_seconds": run["training_wall_seconds_excluding_lock"],
                "training_peak_allocated_mib": run["worker"]["peak_allocated_mib"],
                "fixed9_cuda_render_median_ms": evaluation[
                    "fixed9_cuda_render_median_ms"
                ],
                "evaluation_peak_allocated_mib": evaluation["peak_allocated_mib"],
            }

        paired = {
            "A_to_B": paired_summary(per_view["A_control"], per_view["B_region_rgb"]),
            "A_to_C": paired_summary(
                per_view["A_control"], per_view["C_region_gradient"]
            ),
            "B_to_C": paired_summary(
                per_view["B_region_rgb"], per_view["C_region_gradient"]
            ),
        }
        region_deltas = {}
        for comparison, base, candidate in (
            ("A_to_B", "A_control", "B_region_rgb"),
            ("A_to_C", "A_control", "C_region_gradient"),
            ("B_to_C", "B_region_rgb", "C_region_gradient"),
        ):
            region_deltas[comparison] = {
                key: region_metrics[candidate][key]["macro_mean"]
                - region_metrics[base][key]["macro_mean"]
                for key in ("rgb_l1_01", "multiscale_gradient_l1_01")
            }

        result.update(
            status="completed",
            arm_bindings={
                arm: {
                    "run": str(root.relative_to(ROOT)),
                    "completed_sha256": sha256(root / "completed.json"),
                    "evaluation_sha256": sha256(root / "evaluation/result.json"),
                    "camera_sequence_sha256": sha256(root / "camera_sequence.txt"),
                }
                for arm, root in ARMS.items()
            },
            global_metrics=global_metrics,
            global_deltas={
                comparison: {
                    metric: global_metrics[candidate][metric]
                    - global_metrics[base][metric]
                    for metric in METRIC_DIRECTIONS
                }
                for comparison, base, candidate in (
                    ("A_to_B", "A_control", "B_region_rgb"),
                    ("A_to_C", "A_control", "C_region_gradient"),
                    ("B_to_C", "B_region_rgb", "C_region_gradient"),
                )
            },
            paired_all400=paired,
            class_rgb_mae_255=class_metrics,
            common_dilated_structure_region=region_metrics,
            common_region_deltas=region_deltas,
            region_per_view=region_rows,
            fixed_regions=fixed_regions,
            resources=resources,
            camera_sequences_exact=True,
            camera_sequence_sha256=camera_hashes.pop(),
            preflight_sha256=sha256(
                ROOT / "outputs/quality/rgb147/preflight-r4/result.json"
            ),
            queue_sha256=sha256(ROOT / "outputs/quality/rgb147/queue-r1/result.json"),
            warnings=[
                "All 400 RGB metrics are training-view fit, not held-out generalization.",
                "The 300 label-region diagnostics use released training labels.",
                "The fixed 144 trajectory has no ground truth and is qualitative only.",
                "Different final Gaussian counts are an intended consequence of auxiliary-loss-driven native densification.",
            ],
        )
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
