#!/usr/bin/env python3
"""Build a compact, read-only RGB151 motivation sheet from RGB144 evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


SOURCES = (
    ("Near-field streaking", "outputs/quality/rgb144/review-r1/panels/159-near-0.png"),
    (
        "Distant gray-band region",
        "outputs/quality/rgb144/review-r1/panels/159-far-occlusion-0.png",
    ),
    (
        "Real bridge and stay-cable context",
        "outputs/quality/rgb144/review-r1/panels/159-inherited-structure-crop.png",
    ),
)


def main() -> None:
    output = ROOT / "outputs/quality/RGB-151/diagnostic-r1"
    output.mkdir(parents=True, exist_ok=False)
    width = 1200
    panels = []
    records = []
    for title, relative in SOURCES:
        path = ROOT / relative
        image = Image.open(path).convert("RGB")
        height = round(image.height * width / image.width)
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        panels.append((title, resized))
        records.append({"title": title, "source": relative, "sha256": sha256(path)})
    heading = 34
    canvas = Image.new(
        "RGB", (width, sum(heading + image.height for _, image in panels)), 0xFFFFFF
    )
    draw = ImageDraw.Draw(canvas)
    y = 0
    for title, image in panels:
        draw.text((8, y + 8), title, fill="black")
        y += heading
        canvas.paste(image, (0, y))
        y += image.height
    sheet = output / "shape-motivation.png"
    canvas.save(sheet)
    result = {
        "schema_version": 1,
        "status": "completed_readonly",
        "experiment_id": "RGB-151-STANDARD-ABSGRAD-LAS-DEV30",
        "sources": records,
        "output": str(sheet.relative_to(ROOT)),
        "output_sha256": sha256(sheet),
        "analytical_split_shape": {
            "child_long_axis_scale_ratio": 0.55,
            "child_short_axis_scale_ratio": 0.8930285549745876,
            "child_projected_major_variance_ratio": 0.30250000000000005,
            "child_projected_minor_variance_ratio": 0.7975,
            "single_child_projected_ellipse_area_ratio": 0.4911657052360232,
            "pair_opacity_weighted_area_ratio_first_order": 0.5893988462832278,
            "pair_outer_long_axis_three_sigma_extent_ratio": 1.0,
        },
        "interpretation": "Historical same-position training-view evidence motivates a split-shape test; it is not RGB151 holdout evidence and does not establish causality.",
    }
    write_new_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
