#!/usr/bin/env python3
"""Reflow sealed SEM314 diagnostic panels without changing any scientific pixels.

Presentation only: no labels, probabilities, checkpoints or metric code are read.
Original audit panels and their sealed source snapshot remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit_dir / "result.json"
    audit = json.loads(audit_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"experiment_id": audit["experiment_id"], "presentation_only": True,
              "audit_sha256": sha256(audit_path), "code_sha256": sha256(Path(__file__)),
              "resampling": "none; all five native crop arrays retained exactly", "panels": []}
    try:
        for rec in audit["panels"]:
            source = args.audit_dir / "panels" / rec["name"]
            if sha256(source) != rec["sha256"]:
                raise ValueError("Sealed source panel changed: " + rec["name"])
            old = Image.open(source).convert("RGB")
            x0, y0, x1, y1 = rec["crop_xyxy"]
            w, h = x1 - x0, y1 - y0
            if old.size != (5 * w, h + 44):
                raise ValueError("Unexpected sealed panel layout")
            size = min(36, max(16, round(w / 26)))
            font = ImageFont.truetype("DejaVuSans.ttf", size)
            tile_w, header, row_h = max(220, w) + 24, size * 4, h + size * 3
            new = Image.new("RGB", (3 * tile_w, header + 2 * row_h), "white")
            draw = ImageDraw.Draw(new)
            draw.text((12, 8), rec["name"] + " | SEM314 fixed developer diagnostic", font=font, fill="black")
            draw.text((12, size * 2), "Same crop, native pixels, no resizing", font=font, fill="black")
            placements = [(0, 0, "Frozen RGB"), (1, 0, "GT (reused developer)"),
                          (0, 1, "SEM161 baseline"), (1, 1, "SEM314 final review"),
                          (2, 0, "Changed pixels")]
            bodies = []
            for i, (col, row, title) in enumerate(placements):
                body = old.crop((i * w, 44, (i + 1) * w, 44 + h))
                left, top = col * tile_w + 12, header + row * row_h
                if draw.textlength(title, font=font) > tile_w - 24:
                    raise ValueError("Caption does not fit")
                draw.text((left, top), title, font=font, fill="black")
                box = (left, top + size * 2, left + w, top + size * 2 + h)
                new.paste(body, box[:2]); bodies.append((body, box))
            left, top = 2 * tile_w + 12, header + row_h
            for j, line in enumerate(("Change legend", "Green: corrected", "Red: harmed", "Amber: still wrong", "Black: unchanged")):
                draw.text((left, top + j * size * 2), line, font=font, fill="black")
            target = args.output_dir / rec["name"]
            new.save(target)
            with Image.open(target) as saved:
                for body, box in bodies:
                    if saved.crop(box).tobytes() != body.tobytes():
                        raise ValueError("Scientific pixels changed")
            result["panels"].append({"name": rec["name"], "path": target.name,
                                     "sha256": sha256(target), "source_sha256": rec["sha256"],
                                     "crop_xyxy": rec["crop_xyxy"], "five_native_arrays_exact": True})
        result["status"] = "passed_presentation_layout_and_native_pixel_roundtrip"
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print({"status": result["status"], "panels": len(result["panels"])})
    except Exception as exc:
        (args.output_dir / "failure.json").write_text(json.dumps({"error": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
