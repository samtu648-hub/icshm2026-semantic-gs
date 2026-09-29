#!/usr/bin/env python3
"""Motion-contaminated temporal consistency proxy for the fixed RGB140 trajectory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def summarize(folder: Path) -> dict:
    names = [f"frame{i:04d}.png" for i in range(144)]
    regions = {"full": (0, 989), "sky": (0, 380), "lower_scene": (380, 989)}
    accum = {key: {"sum": 0.0, "count": 0, "above_20": 0, "frame_mean": []}
             for key in regions}
    previous = np.asarray(Image.open(folder / names[0]).convert("RGB"), dtype=np.int16)
    current = np.asarray(Image.open(folder / names[1]).convert("RGB"), dtype=np.int16)
    for name in names[2:]:
        following = np.asarray(Image.open(folder / name).convert("RGB"), dtype=np.int16)
        acceleration = np.abs(following - 2 * current + previous)
        for key, (y0, y1) in regions.items():
            values = acceleration[y0:y1]
            entry = accum[key]
            entry["sum"] += float(values.sum())
            entry["count"] += values.size
            entry["above_20"] += int((values > 20).sum())
            entry["frame_mean"].append(float(values.mean()))
        previous, current = current, following
    return {key: {
        "mean_abs_second_difference_255": value["sum"] / value["count"],
        "fraction_channel_values_above_20": value["above_20"] / value["count"],
        "max_frame_mean_abs_second_difference_255": max(value["frame_mean"]),
        "max_frame_index_center": int(np.argmax(value["frame_mean"])) + 1,
    } for key, value in accum.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    roots = {
        "rgb130": ROOT / "outputs/quality/backbone_comparison/standard/export/images",
        "rgb139": ROOT / "outputs/quality/rgb139/full-r1/evaluation/images",
        "rgb140": ROOT / "outputs/quality/rgb140/full-r2/evaluation/images",
    }
    result = {
        "experiment_id": "RGB-140-STANDARD-ABSGRAD-ALL400",
        "status": "completed_visual_proxy",
        "scope": "same 144 moving cameras, no GT; second temporal difference is contaminated by camera motion and is not a quality metric",
        "models": {name: summarize(folder) for name, folder in roots.items()},
    }
    result["delta_rgb140_vs_rgb130"] = {
        region: {metric: result["models"]["rgb140"][region][metric] - result["models"]["rgb130"][region][metric]
                 for metric in ("mean_abs_second_difference_255", "fraction_channel_values_above_20",
                                "max_frame_mean_abs_second_difference_255")}
        for region in ("full", "sky", "lower_scene")}
    result["seconds"] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
