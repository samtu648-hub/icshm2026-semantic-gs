#!/usr/bin/env python3
"""Build fixed-view panels, browser, and synchronized trajectory video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json


SOURCES = (
    (
        "Photo",
        ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images",
    ),
    ("A control", ROOT / "outputs/quality/rgb147/control-r1/evaluation/images"),
    (
        "B region RGB",
        ROOT / "outputs/quality/rgb147/region-rgb-r1/evaluation/images",
    ),
    (
        "C region gradient",
        ROOT / "outputs/quality/rgb147/region-gradient-r1/evaluation/images",
    ),
)


def panel(name: str, box: tuple[int, int, int, int], destination: Path) -> None:
    width = 440
    height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
    canvas = Image.fromarray(
        np.full((height + 30, width * len(SOURCES), 3), 255, dtype=np.uint8)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, root) in enumerate(SOURCES):
        image = Image.open(root / name).convert("RGB").crop(box)
        canvas.paste(
            image.resize((width, height), Image.Resampling.LANCZOS),
            (index * width, 30),
        )
        draw.text((index * width + 5, 7), title, fill=(0, 0, 0))
    canvas.save(destination)


def build_video(destination: Path) -> None:
    width, height = 3960, 1024
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "12",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    trajectory_sources = SOURCES[1:]
    for index in range(144):
        name = f"frame{index:04d}.png"
        canvas = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
        draw = ImageDraw.Draw(canvas)
        for column, (title, root) in enumerate(trajectory_sources):
            canvas.paste(Image.open(root / name).convert("RGB"), (column * 1320, 35))
            draw.text(
                (column * 1320 + 10, 9),
                f"{title} | {name}",
                fill=(255, 255, 255),
            )
        process.stdin.write(np.asarray(canvas).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg exited {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/quality/rgb147/review-r1"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema_version": 1, "status": "running"}
    try:
        comparison_path = ROOT / "outputs/quality/rgb147/comparison-r1/result.json"
        comparison = json.loads(comparison_path.read_text())
        if comparison["status"] != "completed":
            raise ValueError("Comparison is incomplete")
        fixed = json.loads(
            (
                ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml"
            ).read_text()
        )
        panels = args.output / "panels"
        panels.mkdir()
        panel_records = []
        for name in fixed["views"]:
            destination = panels / f"{Path(name).stem}-full.png"
            panel(name, (0, 0, 1320, 989), destination)
            panel_records.append(
                {
                    "view": name,
                    "region": "full",
                    "box": [0, 0, 1320, 989],
                    "path": str(destination.relative_to(args.output)),
                }
            )
        for region, record in fixed["regions"].items():
            for index, raw_box in enumerate(record["boxes"]):
                box = tuple(raw_box)
                destination = panels / f"{region}-{index}.png"
                panel(record["view"], box, destination)
                panel_records.append(
                    {
                        "view": record["view"],
                        "region": region,
                        "box": raw_box,
                        "path": str(destination.relative_to(args.output)),
                    }
                )

        video = args.output / "trajectory-a-b-c.mp4"
        build_video(video)
        browser_data = {
            "views": [f"{index:03d}.png" for index in range(1, 401)],
            "fixed_views": fixed["views"],
            "regions": fixed["regions"],
            "sources": [
                {
                    "label": "Photo",
                    "root": "../../all_data/RGB-120-r2/scene/images",
                },
                {"label": "A control", "root": "../control-r1/evaluation/images"},
                {
                    "label": "B region RGB",
                    "root": "../region-rgb-r1/evaluation/images",
                },
                {
                    "label": "C region gradient",
                    "root": "../region-gradient-r1/evaluation/images",
                },
            ],
            "metrics": comparison["global_metrics"],
            "deltas": comparison["global_deltas"],
            "classes": comparison["class_rgb_mae_255"],
            "structure": comparison["common_dilated_structure_region"],
            "resources": comparison["resources"],
            "panels": panel_records,
        }
        (args.output / "data.json").write_text(
            json.dumps(browser_data, indent=2, allow_nan=False) + "\n"
        )
        html = r"""<!doctype html><meta charset="utf-8"><title>RGB147-149 GT-region supervision review</title>
<style>body{font:15px/1.55 system-ui;margin:26px;background:#f1f3f5;color:#17212b}h1{font-size:25px}h2{margin-top:32px}.note{max-width:1150px;color:#4a5662}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.card,.tile{background:white;padding:10px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}canvas,video,img{width:100%;display:block}.tools{position:sticky;top:0;background:white;padding:10px;z-index:2}select,button{padding:7px;margin:3px}code{font-size:12px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}a{color:#08677f}</style>
<h1>RGB147-149: semantic GT localizes real-photo supervision</h1><p class="note">A is the matched RGB140 control. B adds real-photo RGB L1 only inside the common dilated GT structure region. C adds native/4x real-photo gradient consistency and does not include B. All 400 scores are training-view fit; released labels are training diagnostics; the 144-frame trajectory has no GT.</p>
<div class="cards" id="cards"></div><div class="tools"><select id="view"></select><select id="preset"></select><button id="full">Full image</button><span id="box"></span></div><div class="grid" id="grid"></div>
<h2>Fixed nine and registered ROIs</h2><p class="note">Full views cover deck, towers, main cable and cable fan. Registered crops cover shore, near foreground, far occlusion, water ghost and sky. The calibration panels additionally show the GT structure mask and gradient-error maps.</p><p><a href="../calibration-r2/panels/159.png">Open calibration panel example</a> | <a href="data.json">Structured review data</a> | <a href="../comparison-r1/result.json">Complete paired metrics</a></p><div class="panels" id="panels"></div>
<h2>Same fixed 144-camera trajectory</h2><video controls preload="metadata" src="trajectory-a-b-c.mp4"></video>
<script>const D=__DATA__,V=document.querySelector('#view'),P=document.querySelector('#preset'),G=document.querySelector('#grid');let box=[0,0,1320,989],loaded=[],generation=0;const f=(x,n=5)=>Number(x).toFixed(n);cards.innerHTML=Object.entries(D.metrics).map(([k,m])=>`<div class=card><b>${k}</b><br>PSNR ${f(m.PSNR)} | SSIM ${f(m.SSIM)} | LPIPS ${f(m.LPIPS)}<br>points ${D.resources[k].final_gaussian_count.toLocaleString()} | train ${(D.resources[k].training_wall_seconds/60).toFixed(1)} min</div>`).join('');D.views.forEach(n=>V.add(new Option((D.fixed_views.includes(n)?'fixed | ':'')+n,n)));P.add(new Option('Registered ROI',''));Object.entries(D.regions).forEach(([k,r])=>P.add(new Option(k,k)));function draw(){document.querySelector('#box').textContent=box.map(Math.round).join(', ');for(const x of loaded){x.c.width=box[2]-box[0];x.c.height=box[3]-box[1];x.c.getContext('2d').drawImage(x.im,box[0],box[1],x.c.width,x.c.height,0,0,x.c.width,x.c.height)}}function load(){const token=++generation;loaded=[];G.replaceChildren();for(const s of D.sources){const d=document.createElement('div');d.className='tile';const a=document.createElement('a'),c=document.createElement('canvas');a.textContent=s.label+' | native PNG';a.href=s.root+'/'+V.value;a.target='_blank';d.append(a,c);G.append(d);const im=new Image();im.onload=()=>{if(token===generation){loaded.push({im,c});draw()}};im.src=a.href}}V.onchange=()=>{box=[0,0,1320,989];P.value='';load()};P.onchange=()=>{if(!P.value)return;const r=D.regions[P.value];V.value=r.view;box=r.boxes[0];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};D.panels.forEach(x=>{const d=document.createElement('div');d.className='tile';d.innerHTML=`<b>${x.view} | ${x.region}</b><br><code>${x.box}</code><a href="${x.path}" target=_blank><img src="${x.path}"></a>`;panels.append(d)});V.value='159.png';load();</script>""".replace(
            "__DATA__", json.dumps(browser_data, separators=(",", ":"))
        )
        (args.output / "index.html").write_text(html + "\n")
        result.update(
            status="completed",
            comparison_sha256=sha256(comparison_path),
            panel_count=len(panel_records),
            video={
                "path": video.name,
                "sha256": sha256(video),
                "frames": 144,
                "fps": 12,
                "columns": [source[0] for source in SOURCES[1:]],
                "ground_truth_available": False,
            },
            artifacts={
                "index.html": sha256(args.output / "index.html"),
                "data.json": sha256(args.output / "data.json"),
                **{
                    record["path"]: sha256(args.output / record["path"])
                    for record in panel_records
                },
            },
        )
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(args.output / "result.json", result)
        print(json.dumps({"status": result["status"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
