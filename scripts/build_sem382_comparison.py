#!/usr/bin/env python3
"""Build the SEM378/381/382-3000/382-6000 matching-location review."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json

EID = "SEM-382-DINOV3-VITB-SFP-LR1E4-ALL300"
BATCH = ROOT / "outputs/quality/SEM-382/r1"
PARENT = ROOT / "outputs/quality/SEM-381/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
OLD_BROWSER = ROOT / "outputs/quality/backbone_comparison/browser-r1"
PHOTO = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
PALETTE = np.asarray(
    [[30, 30, 30], [80, 160, 245], [245, 210, 65], [220, 80, 80], [170, 100, 190]],
    dtype=np.uint8,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    audit = read(BATCH / "audit.json")
    export = read(BATCH / "export/result.json")
    if audit["status"] != "completed" or len(export["image_hashes"]) != 544:
        raise ValueError("Completed audit and 544-view export required")
    output = BATCH / "comparison"
    output.mkdir(exist_ok=False)
    semantic_roots = {
        "SEM378": (BASE / "inference/predictions", output / "sem378"),
        "SEM381": (PARENT / "inference/predictions", output / "sem381"),
        "SEM382-3000": (
            BATCH / "inference-3000/predictions",
            output / "sem382-3000",
        ),
        "SEM382-6000": (
            BATCH / "inference-6000/predictions",
            output / "sem382-6000",
        ),
    }
    for _, destination in semantic_roots.values():
        destination.mkdir()
    gt_root = output / "gt"
    gt_root.mkdir()
    no_gt_changes = output / "no-gt-change-vs-sem381"
    no_gt_changes.mkdir()
    panels = output / "panels"
    panels.mkdir()

    for name in sorted(export["image_hashes"]):
        for source, destination in semantic_roots.values():
            prediction = np.asarray(Image.open(source / name))
            if prediction.shape != (989, 1320) or int(prediction.max()) >= 5:
                raise ValueError(f"Invalid prediction: {source / name}")
            Image.fromarray(PALETTE[prediction]).save(destination / name)
        if (LABELS / name).exists():
            target = np.asarray(Image.open(LABELS / name))
            Image.fromarray(PALETTE[target]).save(gt_root / name)
        else:
            old = np.asarray(Image.open(PARENT / "inference/predictions" / name))
            new = np.asarray(Image.open(BATCH / "inference-6000/predictions" / name))
            delta = np.zeros((*old.shape, 3), np.uint8)
            delta[old != new] = [60, 160, 210]
            delta[(old != 2) & (new == 2)] = [245, 210, 65]
            delta[(old == 2) & (new != 2)] = [190, 70, 210]
            Image.fromarray(delta).save(no_gt_changes / name)

    previous = {row["name"]: row for row in read(OLD_BROWSER / "data.json")["images"]}
    fixed = read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")[
        "diagnostic_views"
    ]
    trajectory = export["trajectory_names"]
    released = sorted(
        name for name in export["image_hashes"] if not name.startswith("frame")
    )
    names = fixed + [name for name in released if name not in fixed] + trajectory
    panel_arms = [
        ("RGB140", BATCH / "export/images", Image.Resampling.LANCZOS),
        ("SEM378", output / "sem378", Image.Resampling.NEAREST),
        ("SEM381", output / "sem381", Image.Resampling.NEAREST),
        ("SEM382-3000", output / "sem382-3000", Image.Resampling.NEAREST),
        ("SEM382-6000", output / "sem382-6000", Image.Resampling.NEAREST),
    ]

    def panel(name: str, box: list[int] | tuple[int, ...] | None = None) -> Image.Image:
        box = box or [0, 0, 1320, 989]
        width = 264
        height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
        header = 18
        canvas = Image.new(
            "RGB", (width * len(panel_arms), height + header), cast(Any, "white")
        )
        draw = ImageDraw.Draw(canvas)
        for column, (label, path, resample) in enumerate(panel_arms):
            draw.text((column * width + 4, 2), f"{name} {label}", fill="black")
            image = (
                Image.open(path / name)
                .convert("RGB")
                .crop(box)
                .resize((width, height), resample)
            )
            canvas.paste(image, (column * width, header))
        return canvas

    for name in fixed:
        item = previous[name]
        panel(name).save(panels / f"{Path(name).stem}-full.png")
        panel(name, item["box"]).save(panels / f"{Path(name).stem}-roi.png")
        panel(name, item["structure_box"]).save(
            panels / f"{Path(name).stem}-structure.png"
        )
    for index in (0, 36, 71, 72, 73, 75, 96, 97, 98, 108, 143):
        panel(trajectory[index]).save(panels / f"trajectory-{index:03d}.png")

    unlabeled_rows = []
    for name in released:
        if (LABELS / name).exists():
            continue
        old = np.asarray(Image.open(PARENT / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference-6000/predictions" / name))
        unlabeled_rows.append(
            {
                "name": name,
                "changed_pixels": int((old != new).sum()),
                "sem381_background_to_cable": int(((old == 0) & (new == 2)).sum()),
                "sem381_cable_to_background": int(((old == 2) & (new == 0)).sum()),
            }
        )
    if len(unlabeled_rows) != 100:
        raise ValueError("Expected exactly 100 released unlabeled views")
    review_names = list(
        dict.fromkeys(
            ["350.png", "390.png", "311.png"]
            + [
                row["name"]
                for row in sorted(
                    unlabeled_rows,
                    key=lambda row: row["changed_pixels"],
                    reverse=True,
                )[:6]
            ]
        )
    )
    for name in review_names:
        panel(name).save(panels / f"unlabeled-{Path(name).stem}.png")
    write_new_json(
        output / "unlabeled-proxy.json",
        {
            "status": "completed",
            "scope": "no GT; label changes only, not accuracy",
            "reference": "SEM381",
            "candidate": "SEM382-6000",
            "per_view": unlabeled_rows,
            "review_names": review_names,
        },
    )

    video = output / "trajectory-rgb140-sem378-sem381-sem382-3000-6000.mp4"
    sample_frame = panel(trajectory[0])
    video_width, video_height = sample_frame.size
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{video_width}x{video_height}",
        "-r",
        "12",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        for name in trajectory:
            process.stdin.write(panel(name).tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")

    sheets = output / "trajectory-sheets"
    sheets.mkdir()
    for start in range(0, 144, 6):
        sheet = Image.new("RGB", (video_width, 6 * video_height), cast(Any, "white"))
        for row, name in enumerate(trajectory[start : start + 6]):
            sheet.paste(panel(name), (0, row * video_height))
        sheet.save(sheets / f"{start:03d}-{start + 5:03d}.jpg", quality=94)
    (sheets / "index.html").write_text(
        '<html><meta charset="utf-8"><h1>全部144帧：SEM378 / SEM381 / SEM382</h1>'
        "<p>无GT；每行同帧，只作定性变化检查。</p>"
        + "".join(
            f'<img style="max-width:100%" src="{path.name}">'
            for path in sorted(sheets.glob("*.jpg"))
        )
    )

    items = []
    for name in names:
        item = {
            "name": name,
            "box": previous[name]["box"] if name in previous else [0, 0, 1320, 989],
            "structure_box": previous[name]["structure_box"]
            if name in previous
            else [0, 0, 1320, 989],
            "tiles": [],
        }
        if (PHOTO / name).exists():
            item["tiles"].append(
                {"label": "Real photo", "path": os.path.relpath(PHOTO / name, output)}
            )
        item["tiles"].append(
            {
                "label": "Fixed RGB140",
                "path": os.path.relpath(BATCH / "export/images" / name, output),
            }
        )
        if (LABELS / name).exists():
            item["tiles"].append(
                {"label": "GT", "path": os.path.relpath(gt_root / name, output)}
            )
        for label, (_, destination) in semantic_roots.items():
            item["tiles"].append(
                {"label": label, "path": os.path.relpath(destination / name, output)}
            )
        if (LABELS / name).exists():
            item["tiles"].append(
                {
                    "label": "SEM382-6000 vs SEM381: green fix / red harm / purple wrong-to-wrong",
                    "path": os.path.relpath(
                        BATCH / "error-changes-vs-sem381" / name, output
                    ),
                }
            )
            item["tiles"].append(
                {
                    "label": "6000 vs 3000: green fix / red harm / purple wrong-to-wrong",
                    "path": os.path.relpath(
                        BATCH / "error-changes-3000-to-6000" / name, output
                    ),
                }
            )
        else:
            item["tiles"].append(
                {
                    "label": "No-GT change vs SEM381: yellow added cable / purple removed cable / blue other",
                    "path": os.path.relpath(no_gt_changes / name, output),
                }
            )
        items.append(item)

    data = {
        "experiment_id": EID,
        "primary_checkpoint": 6000,
        "analysis_checkpoint": 3000,
        "metrics": audit["metrics"],
        "comparisons": audit["comparisons"],
        "boundary": audit["boundary"],
        "regions": audit["regions"],
        "trajectory_proxy": audit["trajectory_proxy"],
        "unlabeled_review_names": review_names,
        "class_names": list(audit["metrics"]["sem382_6000"]["per_class"]),
        "items": items,
        "fixed": fixed,
    }
    write_new_json(output / "data.json", data)
    payload = json.dumps(data, separators=(",", ":"))
    document = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEM382 SFP LR 对照</title>
<style>body{font:15px/1.55 system-ui;margin:24px;background:#f3f5f7;color:#18212a}h1{font-size:25px}.note{max-width:1180px;color:#4c5863}.controls{position:sticky;top:0;background:white;padding:10px;z-index:2}button,select{padding:7px;margin-right:8px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.tile{background:white}.tile p{margin:5px;font-size:12px}canvas{width:100%;display:block}video{width:100%;max-width:1320px}table{border-collapse:collapse;background:white}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style>
<h1>固定 RGB140：SEM378 / SEM381 / SEM382 SFP LR 1e-4</h1>
<p class="note">相同初始化、DINOv3、decoder、300标签、seed、6000采样、原生768裁块和768/512推理；唯一变量是全部SFP参数含norm的初始LR由1e-5变为1e-4。6000为事前主结果，3000仅分析进展。300张分数是训练拟合；100张无标签和144帧轨迹没有GT。</p><div id="scores"></div>
<p class="note">蓝 deck，黄 stay_cable，红 tower，紫 foundation，深灰 background。拖拽任意图同步裁剪；整片索标注不表示逐根几何恢复。</p>
<div class="controls"><select id="views"></select><button id="prev">上一图</button><button id="next">下一图</button><button id="full">完整图</button><button id="roi">困难区域</button><button id="structure">结构区域</button><span id="coords"></span></div><div class="grid" id="grid"></div>
<h2>同一官方坐标144帧轨迹</h2><p><a href="trajectory-sheets/index.html">全部逐帧对照</a></p><video controls preload="metadata" src="trajectory-rgb140-sem378-sem381-sem382-3000-6000.mp4"></video>
<p><a href="data.json">完整数据</a> · <a href="../audit.json">独立审计</a> · <a href="../verification.json">重载/RGB核验</a></p>
<script>const D=__DATA__,f=x=>x==null?'n/a':(100*x).toFixed(6),M=D.metrics,arms=['sem378','sem381','sem382_3000','sem382_6000'];scores.innerHTML='<table><tr><th>系统</th><th>mIoU %</th>'+D.class_names.map(x=>'<th>'+x+' IoU %</th>').join('')+'</tr>'+arms.map(a=>'<tr><td>'+a+'</td><td>'+f(M[a].mIoU_with_background)+'</td>'+D.class_names.map(c=>'<td>'+f(M[a].per_class[c].IoU)+'</td>').join('')+'</tr>').join('')+'</table>'+Object.values(D.comparisons).map(x=>'<p>'+x.new_arm+' - '+x.old_arm+': '+x.delta_mIoU_pp.toFixed(6)+' pp; improved/degraded/tied '+x.paired.improved+'/'+x.paired.degraded+'/'+x.paired.tied+'.</p>').join('');let i=0,box=null,drag=null;D.items.forEach((x,j)=>{const o=document.createElement('option');o.value=j;o.textContent=x.name;views.appendChild(o)});const draw=()=>{const x=D.items[i];views.value=i;grid.innerHTML='';coords.textContent=box?box.join(', '):'full';x.tiles.forEach(t=>{const d=document.createElement('div');d.className='tile';d.innerHTML='<p>'+t.label+'</p><canvas></canvas>';grid.appendChild(d);const c=d.querySelector('canvas'),im=new Image;im.onload=()=>{const b=box||[0,0,im.width,im.height];c.width=b[2]-b[0];c.height=b[3]-b[1];c.getContext('2d').drawImage(im,b[0],b[1],c.width,c.height,0,0,c.width,c.height)};im.src=t.path;c.onpointerdown=e=>{drag=[e.offsetX,e.offsetY,c.width,c.height]};c.onpointerup=e=>{if(!drag)return;const b=box||[0,0,1320,989],sx=(b[2]-b[0])/drag[2],sy=(b[3]-b[1])/drag[3],x0=Math.min(drag[0],e.offsetX),x1=Math.max(drag[0],e.offsetX),y0=Math.min(drag[1],e.offsetY),y1=Math.max(drag[1],e.offsetY);if(x1-x0>8&&y1-y0>8)box=[Math.round(b[0]+x0*sx),Math.round(b[1]+y0*sy),Math.round(b[0]+x1*sx),Math.round(b[1]+y1*sy)];drag=null;draw()}})};views.onchange=()=>{i=+views.value;box=null;draw()};prev.onclick=()=>{i=(i+D.items.length-1)%D.items.length;box=null;draw()};next.onclick=()=>{i=(i+1)%D.items.length;box=null;draw()};full.onclick=()=>{box=null;draw()};roi.onclick=()=>{box=D.items[i].box;draw()};structure.onclick=()=>{box=D.items[i].structure_box;draw()};draw();</script></html>""".replace(
        "__DATA__", payload
    )
    (output / "index.html").write_text(document)
    for item in items:
        for tile in item["tiles"]:
            if not (output / tile["path"]).is_file():
                raise FileNotFoundError(tile)
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
            text=True,
        )
    )
    stream = probe["streams"][0]
    if (
        int(stream["nb_frames"]) != 144
        or int(stream["width"]) != video_width
        or int(stream["height"]) != video_height
    ):
        raise ValueError("Trajectory video dimensions or frame count mismatch")
    write_new_json(
        output / "result.json",
        {
            "schema_version": 1,
            "status": "completed_artifacts_pending_visual_review",
            "experiment_id": EID,
            "frames": 144,
            "video_sha256": sha256(video),
            "image_resources_checked": sum(len(item["tiles"]) for item in items),
            "script_sha256": sha256(Path(__file__)),
            "ffprobe": probe,
        },
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
