#!/usr/bin/env python3
"""Build the read-only RGB170 bridge/context responsibility diagnostic."""

from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bridge_responsibility_clone import summarize_population, summarize_responsibility
from src.quality_protocol import sha256, write_new_json

CONFIG = ROOT / "experiments/runs/QUAL-024-BRIDGE-RESPONSIBILITY-CLONE-DIAG/config.yaml"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def overlay(path: Path, rows: list[dict], stable: set[tuple[int, str]], output: Path) -> None:
    image = Image.open(path).convert("RGB")
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    colors = {"deck": (255, 170, 0), "stay_cable": (255, 40, 40), "tower": (40, 180, 255)}
    selected = [row for row in rows if (int(row["gaussian_id"]), row["class"]) in stable]
    selected.sort(key=lambda row: float(row["structure_weight_density"]), reverse=True)
    for row in selected[:80]:
        x, y = float(row["u"]), float(row["v"])
        length = min(max(float(row["major_radius_px"]), 5.0), 45.0)
        angle = float(row["angle_rad"])
        dx, dy = math.cos(angle) * length, math.sin(angle) * length
        color = colors[row["class"]]
        draw.line((x - dx, y - dy, x + dx, y + dy), fill=color, width=3)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 255, 255), width=1)
    canvas = Image.new("RGB", (image.width * 2, image.height), "white")
    canvas.paste(image, (0, 0))
    canvas.paste(marked, (image.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    config = read_json(CONFIG)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    contributor_path = ROOT / config["inputs"]["contributors"]
    mixed_path = ROOT / config["inputs"]["mixed_gaussians"]
    parent_result = ROOT / config["inputs"]["rgb170_result"]
    parent_ply = ROOT / config["inputs"]["rgb170_ply"]
    mixed: dict[tuple[int, str], dict] = {}
    with gzip.open(mixed_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mixed[(int(row["gaussian_id"]), row["class"])] = row
    observations: dict[tuple[int, str], list[dict]] = defaultdict(list)
    by_view: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(contributor_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["gaussian_id"]), row["class"])
            if key not in mixed:
                continue
            total = float(row["structure_weight_density"]) + float(row["near_background_weight_density"])
            if total <= 0:
                continue
            row["bridge_responsibility_fraction"] = float(row["structure_weight_density"]) / total
            observations[key].append(row)
            by_view[row["view"]].append(row)
    criteria = config["classification"]
    rows: list[dict] = []
    for key, parent in sorted(mixed.items(), key=lambda item: (item[0][1], item[0][0])):
        values = [row["bridge_responsibility_fraction"] for row in observations[key]]
        summary = summarize_responsibility(
            values,
            float(parent["opacity_p50"]),
            minimum_bridge_fraction=float(criteria["minimum_bridge_fraction"]),
            maximum_bridge_fraction=float(criteria["maximum_bridge_fraction"]),
            maximum_iqr=float(criteria["maximum_bridge_fraction_iqr"]),
        )
        rows.append({
            "gaussian_id": key[0], "class": key[1],
            "views": sorted({row["view"] for row in observations[key]}),
            **summary,
        })
    if len(rows) != len(mixed):
        raise RuntimeError("responsibility diagnostic did not cover every QUAL023 mixed Gaussian")
    stable = {(row["gaussian_id"], row["class"]) for row in rows if row["stable_two_responsibility"]}
    record_path = output / "responsibility-gaussians.jsonl.gz"
    with gzip.open(record_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    review_names = config["review_views"]
    image_root = ROOT / config["inputs"]["images"]
    for name in review_names:
        overlay(image_root / name, by_view[name], stable, output / "review" / f"{Path(name).stem}.jpg")
    by_class = {
        class_name: summarize_population([row for row in rows if row["class"] == class_name])
        for class_name in ("deck", "stay_cable", "tower")
    }
    population = summarize_population(rows)
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "status": "completed_read_only",
        "read_only": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(CONFIG),
        "inputs": {
            "contributors_sha256": sha256(contributor_path),
            "mixed_gaussians_sha256": sha256(mixed_path),
            "rgb170_result_sha256": sha256(parent_result),
            "rgb170_ply_sha256": sha256(parent_ply),
        },
        "criteria": criteria,
        "population": population,
        "by_class": by_class,
        "review_views": review_names,
        "responsibility_record_sha256": sha256(record_path),
        "interpretation_scope": (
            "Native RGB170 front-to-back alpha contribution mass aggregated over label-localized structure "
            "and adjacent rays. Labels localize 2D evidence only; no 3D points are created."
        ),
        "candidate_authorized": bool(
            population["stable_two_responsibility"] >= int(criteria["minimum_stable_population"])
            and by_class["stay_cable"]["stable_two_responsibility"]
            >= int(criteria["minimum_stable_stay_cable_population"])
            and population["alpha_conservation_abs_error_max"] <= 1e-12
        ),
    }
    write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
