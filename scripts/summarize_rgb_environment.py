#!/usr/bin/env python3
"""Fixed final audit for the RGB-100/101 direction-background comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic2d import sha256


IDS = ("RGB-030-FULL", "RGB-100-ENV-CTRL", "RGB-101-ENVIRONMENT")


def metrics(run: Path) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    summary = json.loads((run / "r1_metrics_summary.json").read_text(encoding="utf-8"))
    with (run / "r1_metrics_per_view.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["view"]: row for row in csv.DictReader(handle)}
    if len(rows) != 74 or int(summary["view_count"]) != 74:
        raise ValueError(f"Incomplete 74-view RGB audit: {run}")
    return {
        "PSNR": float(summary["metrics"]["PSNR_dB"]["mean"]),
        "SSIM": float(summary["metrics"]["SSIM"]["mean"]),
        "LPIPS": float(summary["metrics"]["LPIPS_Alex"]["mean"]),
        "worst10_PSNR": float(summary["metrics"]["PSNR_dB"]["worst_10_percent_mean"]),
        "challenge_PSNR": float(np.mean([float(rows[f"{index:03}.png"]["PSNR_dB"]) for index in range(140, 147)])),
    }, rows


def gate(candidate: dict[str, float], baseline: dict[str, float], config: dict, *, primary: bool) -> dict:
    delta = {key: candidate[key] - baseline[key] for key in candidate}
    checks = {
        "SSIM_retention": delta["SSIM"] >= -float(config["max_ssim_drop"]),
        "worst10_retention": delta["worst10_PSNR"] >= -float(config["max_worst10_psnr_drop"]),
        "challenge_retention": delta["challenge_PSNR"] >= -float(config["max_challenge_psnr_drop"]),
    }
    if primary:
        checks["PSNR_or_LPIPS_gain"] = (
            delta["PSNR"] >= float(config["min_psnr_gain"])
            or delta["LPIPS"] <= -float(config["min_lpips_reduction"])
        )
    return {"delta": delta, "checks": checks, "passes": all(checks.values())}


def panels(output: Path, source_images: Path, runs: dict[str, Path], views: list[str]) -> tuple[list[str], list[dict]]:
    panel_names: list[str] = []
    closeups: list[dict] = []
    fixed_boxes = {
        "054.png": (350, 550, 850, 950),
        "159.png": (400, 450, 900, 850),
        "160.png": (400, 450, 900, 850),
    }
    for name in views:
        paths = [source_images / name, *[runs[ident] / "r1_validation_renders" / name for ident in IDS]]
        canvas = Image.new("RGB", (1320 * 4, 1025), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (path, title) in enumerate(zip(paths, ("Reference", *IDS))):
            image = Image.open(path).convert("RGB")
            if image.size != (1320, 989):
                raise ValueError(f"Image grid changed: {path}")
            canvas.paste(image, (index * 1320, 36))
            draw.text((index * 1320 + 12, 10), f"{name} | {title}", fill="black")
        panel = output / f"{Path(name).stem}_native_comparison.png"
        canvas.save(panel)
        panel_names.append(panel.name)
        for tile, (x, y) in enumerate(((0, 0), (660, 0), (0, 494), (660, 494))):
            crop = Image.new("RGB", (660 * 4, 531), "white")
            crop_draw = ImageDraw.Draw(crop)
            for index in range(4):
                crop.paste(canvas.crop((index * 1320 + x, y + 36, index * 1320 + x + 660, min(y + 531, 1025))), (index * 660, 36))
                crop_draw.text((index * 660 + 10, 10), ("Reference", *IDS)[index], fill="black")
            crop.save(output / f"{Path(name).stem}_tile{tile}.png")
        if name in fixed_boxes:
            box = fixed_boxes[name]
            closeup = Image.new("RGB", (500 * 4, 436), "white")
            closeup_draw = ImageDraw.Draw(closeup)
            for index, (path, title) in enumerate(zip(paths, ("Reference", *IDS))):
                closeup.paste(Image.open(path).convert("RGB").crop(box), (index * 500, 36))
                closeup_draw.text((index * 500 + 10, 10), title, fill="black")
            closeup_path = output / f"{Path(name).stem}_deck_closeup_native.png"
            closeup.save(closeup_path)
            closeups.append({"view": name, "file": closeup_path.name, "box_xyxy": box, "pixel_scale": 1, "sha256": sha256(closeup_path)})
    return panel_names, closeups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config_path = ROOT / "experiments/runs/RGB-101-ENVIRONMENT/config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = {
        "RGB-030-FULL": args.artifact_root / "r1/runs/RGB-030-FULL",
        "RGB-100-ENV-CTRL": args.artifact_root / "r2/rgb_101/runs/RGB-100-ENV-CTRL-full-r1",
        "RGB-101-ENVIRONMENT": args.artifact_root / "r2/rgb_101/runs/RGB-101-ENVIRONMENT-full-r1",
    }
    for ident in IDS[1:]:
        execution = json.loads((runs[ident] / "execution.json").read_text(encoding="utf-8"))
        if execution.get("status") != "completed":
            raise ValueError(f"Unfinished final run: {ident}")
        if sha256(runs[ident] / "chkpnt30000.pth") != execution["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint modified: {ident}")
        if sha256(runs[ident] / "environment.pth") != execution["environment_sha256"]:
            raise ValueError(f"Environment state modified: {ident}")
    values = {ident: metrics(run)[0] for ident, run in runs.items()}
    versus_control = gate(values[IDS[2]], values[IDS[1]], config["evaluation"], primary=True)
    versus_frozen = gate(values[IDS[2]], values[IDS[0]], config["evaluation"], primary=False)
    args.output_dir.mkdir(parents=True)
    panels_written, closeups = panels(args.output_dir, args.artifact_root / "rgb_r1_spatial_v1/scene_400_eval/images", runs, config["evaluation"]["visual_views"])
    result = {
        "experiment_id": "RGB-101-ENVIRONMENT",
        "status": "numeric_audit_complete_visual_review_pending",
        "config_sha256": sha256(config_path),
        "metrics": values,
        "versus_control": versus_control,
        "versus_frozen": versus_frozen,
        "numeric_gate_passes": bool(versus_control["passes"] and versus_frozen["passes"]),
        "visual_gate_passes": None,
        "promotion": "pending_visual_review",
        "panels": panels_written,
        "deck_closeups": closeups,
        "developer_only_not_blind_test": True,
        "semantic_evaluation_performed": False,
        "fallback": "RGB-030-FULL + SEM-161",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("metrics", "versus_control", "versus_frozen", "numeric_gate_passes")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
