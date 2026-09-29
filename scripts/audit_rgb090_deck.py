#!/usr/bin/env python3
"""Training-only RGB/deck error strata; descriptive evidence, not mask labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.semantic2d import sha256
from src.semantic_render_domain import semantic_rgb


def load(path):
    return np.array(Image.open(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    output = root / "r2/rgb_090/diagnostic/full-20260906-r1"
    output.mkdir(parents=True, exist_ok=False)
    config_path = REPO / "experiments/runs/RGB-090-TRANSIENT-DIAG/config.yaml"
    config = json.loads(config_path.read_text())
    names = json.loads((REPO / config["data"]["semantic_split"]).read_text())["semantic_train_labeled"]
    rendered = root / "r2/sem_101/rendered_training_rgb/full-20260902-r3"
    predroot = root / "r2/sem_160/deck_binary_oof_summary/full-20260904-r1"
    manifest = json.loads((predroot / "prediction_manifest.json").read_text())
    records = manifest.get("predictions", manifest.get("records"))
    paths = {r["name"]: r for r in records}
    rgb_manifest = json.loads((rendered / "rendered_dataset_manifest.json").read_text())
    rgb_paths = {r["name"]: r for r in rgb_manifest["rendered_images"]}
    totals = {}
    views = []
    accessed = []
    for name in names:
        reference_path = root / "rgb_r1_spatial_v1/scene_400_eval/images" / name
        render_path = rendered / rgb_paths[name]["path"]
        pred_path = predroot / paths[name]["prediction"]
        label_path = root / "r2/sem_001/labels" / name
        if sha256(render_path) != rgb_paths[name]["sha256"] or sha256(pred_path) != paths[name]["prediction_sha256"]:
            raise ValueError("Frozen training evidence changed")
        reference, render = load(reference_path).astype(np.float32) / 255, load(render_path).astype(np.float32) / 255
        label, prediction = load(label_path), load(pred_path)
        if reference.shape != (989, 1320, 3) or label.shape != prediction.shape or label.shape != (989, 1320):
            raise ValueError("Training grid mismatch")
        mae = np.abs(reference - render).mean(-1)
        gray = reference.mean(-1)
        edge = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        interior = cv2.erode((label == 1).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        strata = {"deck_true_positive": (label == 1) & (prediction == 1),
                  "deck_missed_as_background": (label == 1) & (prediction == 0),
                  "deck_missed_as_other_structure": (label == 1) & (prediction > 1),
                  "deck_interior_missed": interior & (prediction != 1),
                  "false_deck_on_background": (prediction == 1) & (label == 0),
                  "false_deck_on_stay_cable": (prediction == 1) & (label == 2),
                  "true_cable_next_to_deck": (label == 2) & (cv2.dilate((prediction == 1).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)}
        row = {"name": name, "strata": {}}
        for key, mask in strata.items():
            n = int(mask.sum())
            item = {"pixels": n, "mae_sum": float(mae[mask].sum(dtype=np.float64)),
                    "edge_sum": float(edge[mask].sum(dtype=np.float64)),
                    "mae_below_0_05_pixels": int((mae[mask] < config["diagnostic"]["mae_cutoff"]).sum())}
            row["strata"][key] = item
            total = totals.setdefault(key, {k: 0 for k in item})
            for k, v in item.items():
                total[k] += v
        views.append(row)
        accessed.append({"name": name, "reference_sha256": sha256(reference_path), "render_sha256": sha256(render_path),
                         "label_sha256": sha256(label_path), "prediction_sha256": sha256(pred_path)})
    for item in totals.values():
        item["mean_mae"] = item["mae_sum"] / max(1, item["pixels"])
        item["mean_reference_edge"] = item["edge_sum"] / max(1, item["pixels"])
        item["low_rgb_error_fraction"] = item["mae_below_0_05_pixels"] / max(1, item["pixels"])
    selected = sorted(views, key=lambda row: (-row["strata"]["deck_missed_as_background"]["pixels"], row["name"]))[:6]
    panels = []
    for row in selected:
        name = row["name"]
        images = [Image.open(root / "rgb_r1_spatial_v1/scene_400_eval/images" / name).convert("RGB"),
                  Image.open(rendered / rgb_paths[name]["path"]).convert("RGB"),
                  Image.fromarray(semantic_rgb(load(root / "r2/sem_001/labels" / name))),
                  Image.fromarray(semantic_rgb(load(predroot / paths[name]["prediction"])))]
        panel = Image.new("RGB", (1320 * 4, 1025), "white")
        draw = ImageDraw.Draw(panel)
        for i, (im, title) in enumerate(zip(images, ["Reference RGB", "RGB-030 render", "GT", "SEM-160 OOF"])):
            panel.paste(im, (1320 * i, 36))
            draw.text((1320 * i + 10, 10), name + " | " + title, fill="black")
        path = output / (Path(name).stem + "_native_panel.png")
        panel.save(path)
        panels.append(path.name)
    result = {"experiment_id": config["experiment_id"], "status": "completed_training_only_diagnostic",
              "config_sha256": sha256(config_path), "training_views": len(names), "validation_views": 0,
              "strata": totals, "views": views, "access": accessed, "panels": panels,
              "interpretation": "MAE<0.05 is a descriptive proxy, not proof of clarity or causality. GT labels distinguish true deck misses from cable/background. No diagnostic mask enters training."}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(output), "strata": totals, "sha256": sha256(output / "result.json")}))


if __name__ == "__main__":
    main()
