#!/usr/bin/env python3
"""Build a fixed same-position RGB150/RGB151 review page."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


CONTROL = ROOT / "outputs/quality/RGB-150/full-r1/evaluation"
CANDIDATE = ROOT / "outputs/quality/RGB-151/full-r2/evaluation-r3"
REFERENCE = ROOT / "outputs/quality/RGB-150/scene/images"
OUTPUT = ROOT / "outputs/quality/RGB-151/full-r2/review-r1"
VIEWS = (
    "057.png",
    "064.png",
    "108.png",
    "145.png",
    "159.png",
    "160.png",
    "174.png",
    "180.png",
    "188.png",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_rows(root: Path) -> dict[str, dict[str, float]]:
    rows = {}
    for partition in ("train", "holdout"):
        with (root / f"{partition}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["view"]] = {
                    key: float(row[key]) for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
                }
    return rows


def review_html(rows: list[dict[str, Any]]) -> str:
    embedded = json.dumps(rows, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RGB151 Long-Axis Split Review</title>
<style>
:root {{ color-scheme:dark; --bg:#101315; --panel:#1a1f22; --line:#394147; --text:#eef1f2; --muted:#aab3b8; --accent:#d9b75d; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
header {{ position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:16px; padding:14px 20px; background:#101315f2; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:20px; }} select {{ background:var(--panel); color:var(--text); border:1px solid var(--line); padding:7px 10px; }}
main {{ max-width:1560px; margin:auto; padding:18px; }} .summary,.metrics {{ color:var(--muted); }} .metrics b {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-top:16px; }} figure {{ margin:0; border:1px solid var(--line); background:var(--panel); }}
figcaption {{ padding:8px 10px; border-bottom:1px solid var(--line); color:var(--muted); }} img {{ display:block; width:100%; aspect-ratio:1320/989; object-fit:contain; }}
@media(max-width:850px) {{ header {{ align-items:flex-start; flex-direction:column; }} .grid {{ grid-template-columns:1fr; }} }}
</style></head><body>
<header><h1>RGB151 Long-Axis Split Review</h1><label>Fixed view <select id="view"></select></label></header>
<main><p class="summary">Reference, frozen RGB150 control, and RGB151 LAS candidate at identical official cameras. Difference is |candidate-control| amplified 4x; it is diagnostic only.</p><p class="metrics" id="metrics"></p><section class="grid" id="grid"></section></main>
<script>
const rows={embedded}; const select=document.querySelector('#view');
for(const row of rows){{const option=document.createElement('option'); option.value=row.name; option.textContent=row.name; select.appendChild(option);}}
function f(x,n=6){{return Number(x).toFixed(n);}}
function show(){{const row=rows.find(x=>x.name===select.value); const d=row.delta;
document.querySelector('#metrics').innerHTML=`Candidate-control: PSNR <b>${{d.PSNR_dB>=0?'+':''}}${{f(d.PSNR_dB)}} dB</b>, SSIM <b>${{d.SSIM>=0?'+':''}}${{f(d.SSIM)}}</b>, LPIPS <b>${{d.LPIPS_Alex>=0?'+':''}}${{f(d.LPIPS_Alex)}}</b>; RGB mean absolute change <b>${{f(row.mean_absolute_change,8)}}</b>.`;
const tiles=[['Reference',row.reference],['RGB150 control',row.control],['RGB151 LAS',row.candidate],['4x candidate-control difference',row.difference]];
document.querySelector('#grid').innerHTML=tiles.map(x=>`<figure><figcaption>${{x[0]}}</figcaption><img src="${{x[1]}}" alt="${{x[0]}} ${{row.name}}"></figure>`).join('');}}
select.addEventListener('change',show); show();
</script></body></html>"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Review output already exists: {OUTPUT}")
    control_result = read(CONTROL / "result.json")
    candidate_result = read(CANDIDATE / "result.json")
    if (
        control_result["status"] != "completed"
        or candidate_result["status"] != "completed"
    ):
        raise ValueError("Completed control and candidate evaluations are required")
    if not candidate_result["endpoint_unchanged"]:
        raise ValueError("RGB151 endpoint changed during evaluation")
    control_metrics = metric_rows(CONTROL)
    candidate_metrics = metric_rows(CANDIDATE)
    OUTPUT.mkdir(parents=True)
    for directory in ("reference", "control", "candidate", "difference"):
        (OUTPUT / directory).mkdir()

    rows = []
    for name in VIEWS:
        reference = np.asarray(
            Image.open(REFERENCE / name).convert("RGB"), dtype=np.int16
        )
        control = np.asarray(
            Image.open(CONTROL / "images" / name).convert("RGB"), dtype=np.int16
        )
        candidate = np.asarray(
            Image.open(CANDIDATE / "images" / name).convert("RGB"), dtype=np.int16
        )
        if reference.shape != control.shape or control.shape != candidate.shape:
            raise ValueError(f"Image shape mismatch for {name}")
        shutil.copyfile(REFERENCE / name, OUTPUT / "reference" / name)
        shutil.copyfile(CONTROL / "images" / name, OUTPUT / "control" / name)
        shutil.copyfile(CANDIDATE / "images" / name, OUTPUT / "candidate" / name)
        difference = np.clip(np.abs(candidate - control) * 4, 0, 255).astype(np.uint8)
        Image.fromarray(difference).save(OUTPUT / "difference" / name)
        delta = {
            key: candidate_metrics[name][key] - control_metrics[name][key]
            for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
        }
        rows.append(
            {
                "name": name,
                "control": f"control/{name}",
                "candidate": f"candidate/{name}",
                "reference": f"reference/{name}",
                "difference": f"difference/{name}",
                "delta": delta,
                "mean_absolute_change": float(
                    np.mean(np.abs(candidate - control)) / 255.0
                ),
            }
        )

    sheet_width = 1320
    thumb_width, thumb_height = 330, 247
    sheet = Image.new(  # type: ignore[arg-type]
        "RGB",
        (sheet_width, len(rows) * (thumb_height + 28)),
        (16, 19, 21),  # type: ignore[arg-type]
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        y = index * (thumb_height + 28)
        draw.text((4, y + 5), row["name"], fill=(238, 241, 242))
        for column, key in enumerate(
            ("reference", "control", "candidate", "difference")
        ):
            image = Image.open(OUTPUT / row[key]).convert("RGB")
            image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (column * thumb_width, y + 28))
    sheet.save(OUTPUT / "contact-sheet.jpg", quality=92)
    (OUTPUT / "index.html").write_text(review_html(rows), encoding="utf-8")
    result = {
        "schema_version": 1,
        "experiment_id": "RGB-151-STANDARD-ABSGRAD-LAS-DEV30",
        "status": "completed",
        "scope": "Fixed post-hoc same-camera visual review; no model or parameter selection",
        "control_result_sha256": sha256(CONTROL / "result.json"),
        "candidate_result_sha256": sha256(CANDIDATE / "result.json"),
        "views": rows,
        "contact_sheet_sha256": sha256(OUTPUT / "contact-sheet.jpg"),
        "index_sha256": sha256(OUTPUT / "index.html"),
    }
    write_new_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
