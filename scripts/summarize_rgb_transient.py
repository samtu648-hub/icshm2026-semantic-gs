#!/usr/bin/env python3
"""Frozen final-only numeric gates and native RGB comparison panels."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.semantic2d import sha256


def metrics(run):
    summary = json.loads((run / "r1_metrics_summary.json").read_text())
    with (run / "r1_metrics_per_view.csv").open() as f:
        rows = {row["view"]: row for row in csv.DictReader(f)}
    if len(rows) != 74 or summary["view_count"] != 74:
        raise ValueError("Expected complete 74-view audit")
    return {"PSNR": summary["metrics"]["PSNR_dB"]["mean"],
            "SSIM": summary["metrics"]["SSIM"]["mean"],
            "LPIPS": summary["metrics"]["LPIPS_Alex"]["mean"],
            "worst10_PSNR": summary["metrics"]["PSNR_dB"]["worst_10_percent_mean"],
            "challenge_PSNR": float(np.mean([float(rows[f"{i:03}.png"]["PSNR_dB"]) for i in range(140, 147)]))}, rows


def gate(candidate, baseline, cfg, primary):
    delta = {k: candidate[k] - baseline[k] for k in candidate}
    checks = {"SSIM_retention": delta["SSIM"] >= -cfg["max_ssim_drop"],
              "worst10_retention": delta["worst10_PSNR"] >= -cfg["max_worst10_psnr_drop"],
              "challenge_retention": delta["challenge_PSNR"] >= -cfg["max_challenge_psnr_drop"]}
    if primary:
        checks["PSNR_or_LPIPS_gain"] = delta["PSNR"] >= cfg["min_psnr_gain"] or delta["LPIPS"] <= -cfg["min_lpips_reduction"]
    return {"delta": delta, "checks": checks, "passes": all(checks.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    output = root / "r2/rgb_092/final_audit/full-20260906-r1"
    output.mkdir(parents=True, exist_ok=False)
    cfg_path = REPO / "experiments/runs/RGB-092-TRANSIENT/config.yaml"
    cfg = json.loads(cfg_path.read_text())
    ids = ["RGB-030-FULL", "RGB-091-CTRL", "RGB-092-TRANSIENT"]
    runs = {i: root / "r1/runs" / i for i in ids}
    provenance = {}
    for ident in ids[1:]:
        execution = json.loads((runs[ident] / "execution.json").read_text())
        if execution["status"] != "completed" or not execution["frozen_parent_unchanged"]:
            raise ValueError("Run incomplete or frozen parent changed")
        if sha256(runs[ident] / "chkpnt30000.pth") != execution["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        provenance[ident] = execution
    values = {i: metrics(runs[i])[0] for i in ids}
    control_gate = gate(values[ids[2]], values[ids[1]], cfg["evaluation"], True)
    historical_gate = gate(values[ids[2]], values[ids[0]], cfg["evaluation"], False)
    panels = []
    closeups = []
    for name in cfg["evaluation"]["visual_views"]:
        panel = Image.new("RGB", (1320 * 4, 1025), "white")
        draw = ImageDraw.Draw(panel)
        paths = [root / "rgb_r1_spatial_v1/scene_400_eval/images" / name,
                 *[runs[i] / "r1_validation_renders" / name for i in ids]]
        for j, (path, title) in enumerate(zip(paths, ["Reference", *ids])):
            im = Image.open(path).convert("RGB")
            if im.size != (1320, 989):
                raise ValueError("Native image size changed")
            panel.paste(im, (j * 1320, 36))
            draw.text((j * 1320 + 12, 10), name + " | " + title, fill="black")
        target = output / (Path(name).stem + "_native_comparison.png")
        panel.save(target)
        # Fixed half-frame tiles preserve all original pixels, with no filtering.
        for tile, (x, y) in enumerate([(0, 0), (660, 0), (0, 494), (660, 494)]):
            crop = Image.new("RGB", (660 * 4, 531), "white")
            for j in range(4):
                crop.paste(panel.crop((j * 1320 + x, y + 36, j * 1320 + x + 660, min(y + 36 + 495, 1025))), (j * 660, 36))
                ImageDraw.Draw(crop).text((j * 660 + 10, 10), ["Reference", *ids][j], fill="black")
            crop.save(output / f"{Path(name).stem}_tile{tile}.png")
        panels.append(target.name)
        # Presentation-only bridge crops, fixed from the OLD RGB before the
        # candidate final renders exist. Never feed these coordinates to training.
        bridge_boxes = {"054.png": (350, 550, 850, 950),
                        "159.png": (400, 450, 900, 850),
                        "160.png": (400, 450, 900, 850)}
        if name in bridge_boxes:
            box = bridge_boxes[name]
            detail = Image.new("RGB", (2000, 436), "white")
            for j, (path, title) in enumerate(zip(paths, ["Reference", *ids])):
                detail.paste(Image.open(path).convert("RGB").crop(box), (j * 500, 36))
                ImageDraw.Draw(detail).text((j * 500 + 10, 10), title, fill="black")
            detail_path = output / f"{Path(name).stem}_deck_closeup_native.png"
            detail.save(detail_path)
            closeups.append({"view": name, "file": detail_path.name, "box_xyxy": box,
                             "pixel_scale": 1, "sha256": sha256(detail_path)})
    result = {"experiment_id": ids[2], "status": "numeric_audit_complete_visual_review_pending",
              "config_sha256": sha256(cfg_path), "metrics": values, "versus_control": control_gate,
              "versus_historical": historical_gate, "numeric_gate_passes": control_gate["passes"] and historical_gate["passes"],
              "visual_gate_passes": None, "SEM180_allowed": False, "panels": panels,
              "deck_closeups": closeups,
              "provenance": provenance, "developer_only_not_blind_test": True}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["metrics", "versus_control", "versus_historical", "numeric_gate_passes"]}))


if __name__ == "__main__":
    main()
