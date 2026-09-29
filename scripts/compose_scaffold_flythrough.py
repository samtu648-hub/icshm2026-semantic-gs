#!/usr/bin/env python3
"""Create side-by-side frozen Scaffold RGB and SEM-001 overlay video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]], dtype=np.uint8
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    args = parser.parse_args()
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be in [0,1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty frame directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.renders / "submission_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("uses_reference_rgb_or_labels") is not False:
        raise ValueError("Render manifest does not prove the no-reference-data contract")

    header = 40
    frame_records = []
    for index, record in enumerate(manifest["outputs"]):
        rgb = np.asarray(Image.open(args.renders / record["rgb"]["path"]).convert("RGB"), dtype=np.uint8)
        semantic = np.asarray(Image.open(args.renders / record["semantic"]["path"]), dtype=np.uint8)
        if semantic.shape != rgb.shape[:2] or (semantic.size and int(semantic.max()) > 4):
            raise ValueError(f"Invalid semantic frame: {record['pose_name']}")
        colors = PALETTE[semantic]
        overlay = rgb.copy()
        foreground = semantic != 0
        alpha = args.overlay_alpha
        overlay[foreground] = np.rint((1.0 - alpha) * rgb[foreground] + alpha * colors[foreground]).astype(np.uint8)
        canvas = Image.new("RGB", (rgb.shape[1] * 2, rgb.shape[0] + header), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 14), "Frozen Scaffold-GS novel-view RGB", fill="white")
        draw.text(
            (rgb.shape[1] + 12, 14),
            "SEM-001 overlay: deck orange | stay_cable yellow | tower blue | foundation green",
            fill="white",
        )
        canvas.paste(Image.fromarray(rgb, mode="RGB"), (0, header))
        canvas.paste(Image.fromarray(overlay, mode="RGB"), (rgb.shape[1], header))
        output = args.output_dir / f"frame{index:04d}.png"
        canvas.save(output)
        frame_records.append({"index": index, "pose_name": record["pose_name"], "path": output.name})
    result = {
        "schema_version": 1,
        "source_submission_id": manifest["submission_id"],
        "frame_count": len(frame_records),
        "width": int(manifest["outputs"][0]["width"] * 2),
        "height": int(manifest["outputs"][0]["height"] + header),
        "overlay_alpha": args.overlay_alpha,
        "uses_reference_rgb_or_labels": False,
        "frames": frame_records,
    }
    (args.output_dir / "composition_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"frame_count": len(frame_records), "status": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
