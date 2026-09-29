#!/usr/bin/env python3
"""Display-only padding of sealed SEM326 panels; never decode GT or predictions."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(audit_dir, output_dir):
    audit_path = audit_dir / "result.json"
    audit = json.loads(audit_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in audit["panels"]:
        source = audit_dir / row["path"]
        if digest(source) != row["sha256"]:
            raise ValueError("Sealed panel differs")
        x0, y0, x1, y1 = row["crop_xyxy"]
        width, height = x1 - x0, y1 - y0
        cell_width, header = max(320, width + 16), 52
        with Image.open(source) as original:
            if original.size != (width * 3, (height + 32) * 2):
                raise ValueError("Unexpected original panel layout")
            display = Image.new("RGB", (cell_width * 3, (height + header) * 2), "white")
            draw = ImageDraw.Draw(display)
            titles = [row["name"] + " frozen RGB", "GT (reused developer)", "SEM161 baseline",
                      "SEM326 dense review", "Green: corrected / Red: harmed\nAmber: still wrong",
                      "Only 4-to-0; native pixels\nPadding only, no resampling"]
            for i, title in enumerate(titles):
                sx, sy = (i % 3) * width, (i // 3) * (height + 32) + 32
                tile = original.crop((sx, sy, sx + width, sy + height))
                dx, dy = (i % 3) * cell_width, (i // 3) * (height + header)
                left, top = dx + (cell_width - width) // 2, dy + header
                display.paste(tile, (left, top))
                draw.multiline_text((dx + 8, dy + 6), title, fill="black", spacing=4)
                if display.crop((left, top, left + width, top + height)).tobytes() != tile.tobytes():
                    raise ValueError("Display changed a scientific image tile")
            target = output_dir / row["name"]
            display.save(target)
        rows.append({"name": row["name"], "source_sha256": row["sha256"],
                     "output_sha256": digest(target), "crop_xyxy": row["crop_xyxy"],
                     "native_image_tiles_exact": True})
    result = {"experiment_id": "SEM-326-DENSE-FOUNDATION-REVIEW-FINAL",
              "status": "display_only_layout_verified", "audit_sha256": digest(audit_path),
              "script_sha256": digest(Path(__file__)), "new_gt_decodes": 0,
              "new_inference": 0, "resampling": False, "panels": rows}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.audit_dir.resolve(), args.output_dir.resolve())
    print(result["status"], len(result["panels"]))
