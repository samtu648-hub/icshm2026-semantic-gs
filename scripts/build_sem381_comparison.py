#!/usr/bin/env python3
"""Build matching-location RGB140/SEM378/SEM381 review assets."""

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

EID = "SEM-381-DINOV3-VITB-SFP-ALL300"
BATCH = ROOT / "outputs/quality/SEM-381/r1"
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
    assert audit["status"] == "completed" and len(export["image_hashes"]) == 544
    output = BATCH / "comparison"
    output.mkdir(exist_ok=False)
    semantic = output / "sem381"
    semantic.mkdir()
    gt_root = output / "gt"
    gt_root.mkdir()
    changes = output / "no-gt-change"
    changes.mkdir()
    panels = output / "panels"
    panels.mkdir()

    for name in sorted(export["image_hashes"]):
        prediction = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        assert prediction.shape == (989, 1320) and int(prediction.max()) < 5
        Image.fromarray(PALETTE[prediction]).save(semantic / name)
        if (LABELS / name).exists():
            target = np.asarray(Image.open(LABELS / name))
            Image.fromarray(PALETTE[target]).save(gt_root / name)
        else:
            old = np.asarray(Image.open(BASE / "inference/predictions" / name))
            delta = np.zeros((*old.shape, 3), np.uint8)
            delta[old != prediction] = [60, 160, 210]
            delta[(old != 2) & (prediction == 2)] = [245, 210, 65]
            delta[(old == 2) & (prediction != 2)] = [190, 70, 210]
            Image.fromarray(delta).save(changes / name)

    previous = {row["name"]: row for row in read(OLD_BROWSER / "data.json")["images"]}
    fixed = read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")[
        "diagnostic_views"
    ]
    trajectory = export["trajectory_names"]
    released = sorted(
        name for name in export["image_hashes"] if not name.startswith("frame")
    )
    names = fixed + [name for name in released if name not in fixed] + trajectory
    arms = [
        ("RGB140", BATCH / "export/images", Image.Resampling.LANCZOS),
        ("SEM378", BASE / "comparison/sem378", Image.Resampling.NEAREST),
        ("SEM381", semantic, Image.Resampling.NEAREST),
    ]

    def panel(name: str, box: list[int] | tuple[int, ...] | None = None) -> Image.Image:
        box = box or [0, 0, 1320, 989]
        width = 440
        height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
        header = 18
        canvas = Image.new("RGB", (width * 3, height + header), cast(Any, "white"))
        draw = ImageDraw.Draw(canvas)
        for column, (label, path, resample) in enumerate(arms):
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

    video = output / "trajectory-rgb140-sem378-sem381.mp4"
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
        "1320x348",
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
        assert process.stdin is not None
        for name in trajectory:
            frame = panel(name)
            assert frame.size == (1320, 348)
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    assert process.wait() == 0

    sheets = output / "trajectory-sheets"
    sheets.mkdir()
    for start in range(0, 144, 6):
        sheet = Image.new("RGB", (1320, 6 * 348), cast(Any, "white"))
        for row, name in enumerate(trajectory[start : start + 6]):
            sheet.paste(panel(name), (0, row * 348))
        sheet.save(sheets / f"{start:03d}-{start + 5:03d}.jpg", quality=94)
    (sheets / "index.html").write_text(
        '<html><meta charset="utf-8"><h1>全部144帧：RGB140 / SEM378 / SEM381</h1>'
        "<p>无GT；每行同帧，只作定性变化检查。</p>"
        + "".join(
            f'<img style="max-width:100%" src="{path.name}">'
            for path in sorted(sheets.glob("*.jpg"))
        )
    )

    items = []
    for name in names:
        record = {
            "name": name,
            "box": previous[name]["box"] if name in previous else [0, 0, 1320, 989],
            "structure_box": previous[name]["structure_box"]
            if name in previous
            else [0, 0, 1320, 989],
            "tiles": [],
        }
        if (PHOTO / name).exists():
            record["tiles"].append(
                {"label": "真实照片", "path": os.path.relpath(PHOTO / name, output)}
            )
        record["tiles"].append(
            {
                "label": "固定RGB140",
                "path": os.path.relpath(BATCH / "export/images" / name, output),
            }
        )
        if (LABELS / name).exists():
            record["tiles"].append(
                {"label": "GT", "path": os.path.relpath(gt_root / name, output)}
            )
        record["tiles"].extend(
            [
                {
                    "label": "SEM378",
                    "path": os.path.relpath(BASE / "comparison/sem378" / name, output),
                },
                {
                    "label": "SEM381 DINOv3",
                    "path": os.path.relpath(semantic / name, output),
                },
            ]
        )
        if (LABELS / name).exists():
            record["tiles"].append(
                {
                    "label": "错误变化：绿修对 / 红改错 / 紫换错类",
                    "path": os.path.relpath(BATCH / "error-changes" / name, output),
                }
            )
        else:
            record["tiles"].append(
                {
                    "label": "无GT变化：黄新增索 / 紫删除索 / 蓝其它",
                    "path": os.path.relpath(changes / name, output),
                }
            )
        items.append(record)

    unlabeled = []
    for name in released:
        if (LABELS / name).exists():
            continue
        old = np.asarray(Image.open(BASE / "inference/predictions" / name))
        new = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        unlabeled.append(
            {
                "name": name,
                "changed_pixels": int((old != new).sum()),
                "old_background_to_cable": int(((old == 0) & (new == 2)).sum()),
                "old_cable_to_background": int(((old == 2) & (new == 0)).sum()),
            }
        )
    assert len(unlabeled) == 100
    hard = [
        name
        for name in ("350.png", "390.png", "311.png")
        if any(row["name"] == name for row in unlabeled)
    ]
    review_names = list(
        dict.fromkeys(
            hard
            + [
                row["name"]
                for row in sorted(
                    unlabeled, key=lambda row: row["changed_pixels"], reverse=True
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
            "per_view": unlabeled,
            "review_names": review_names,
        },
    )

    class_names = list(audit["metrics"]["sem381"]["per_class"])
    assert class_names == list(audit["metrics"]["sem378"]["per_class"])
    data = {
        "experiment_id": EID,
        "metrics": audit["metrics"],
        "delta_mIoU_pp": audit["delta_mIoU_pp"],
        "paired": {
            key: value for key, value in audit["paired"].items() if key != "per_view"
        },
        "pixel_changes": audit["pixel_changes"],
        "cable_changes": audit["cable_changes"],
        "boundary": audit["boundary"],
        "regions": audit["regions"],
        "trajectory_proxy": audit["trajectory_proxy"],
        "unlabeled_review_names": review_names,
        "class_names": class_names,
        "items": items,
        "fixed": fixed,
    }
    write_new_json(output / "data.json", data)
    payload = json.dumps(data, separators=(",", ":"))
    document = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEM381 DINOv3 对照</title>
<style>body{font:15px/1.55 system-ui;margin:24px;background:#f3f5f7;color:#18212a}h1{font-size:25px}.note{max-width:1100px;color:#4c5863}.controls{position:sticky;top:0;background:white;padding:10px;z-index:2}button,select{padding:7px;margin-right:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.tile{background:white}.tile p{margin:5px;font-size:12px}canvas{width:100%;display:block}video{width:100%;max-width:1320px}table{border-collapse:collapse;background:white}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<h1>固定 RGB140：SEM378 Swin-L vs SEM381 frozen DINOv3 ViT-B/16 + SFP</h1>
<p class="note">相同544张RGB140、300标签、seed、原生768裁块、原6000采样、Mask2Former decoder/loss和768/512推理。唯一方法包变量是冻结DINOv3特征及其必需SFP替换Swin-L。300张分数是训练拟合；100张无标签和144帧轨迹没有GT。</p><div id="scores"></div>
<p class="note">蓝 deck，黄 stay_cable，红 tower，紫 foundation，深灰 background。拖拽任意图同步裁剪；整片索标注不表示逐根几何恢复。</p>
<div class="controls"><select id="views"></select><button id="prev">上一图</button><button id="next">下一图</button><button id="full">完整图</button><button id="roi">困难区域</button><button id="structure">结构区域</button><span id="coords"></span></div><div class="grid" id="grid"></div>
<h2>同一官方坐标144帧轨迹</h2><p><a href="trajectory-sheets/index.html">全部逐帧对照</a></p><video controls preload="metadata" src="trajectory-rgb140-sem378-sem381.mp4"></video>
<p><a href="data.json">完整数据</a> · <a href="../audit.json">独立审计</a> · <a href="../verification.json">重载/RGB核验</a></p>
<script>const D=__DATA__,f=x=>(100*x).toFixed(6),M=D.metrics;scores.innerHTML='<table><tr><th>系统</th><th>mIoU %</th>'+D.class_names.map(x=>'<th>'+x+' IoU %</th>').join('')+'</tr>'+['sem378','sem381'].map(a=>'<tr><td>'+a+'</td><td>'+f(M[a].mIoU_with_background)+'</td>'+D.class_names.map(c=>'<td>'+f(M[a].per_class[c].IoU)+'</td>').join('')+'</tr>').join('')+'</table><p>SEM381 - SEM378: '+D.delta_mIoU_pp.toFixed(6)+' pp; improved/degraded/tied views '+D.paired.improved+'/'+D.paired.degraded+'/'+D.paired.tied+'.</p>';let i=0,box=null,drag=null;D.items.forEach((x,j)=>{const o=document.createElement('option');o.value=j;o.textContent=x.name;views.appendChild(o)});const draw=()=>{const x=D.items[i];views.value=i;grid.innerHTML='';coords.textContent=box?box.join(', '):'full';x.tiles.forEach(t=>{const d=document.createElement('div');d.className='tile';d.innerHTML='<p>'+t.label+'</p><canvas></canvas>';grid.appendChild(d);const c=d.querySelector('canvas'),im=new Image;im.onload=()=>{const b=box||[0,0,im.width,im.height];c.width=b[2]-b[0];c.height=b[3]-b[1];c.getContext('2d').drawImage(im,b[0],b[1],c.width,c.height,0,0,c.width,c.height)};im.src=t.path;c.onpointerdown=e=>{drag=[e.offsetX,e.offsetY,c.width,c.height]};c.onpointerup=e=>{if(!drag)return;const b=box||[0,0,1320,989],sx=(b[2]-b[0])/drag[2],sy=(b[3]-b[1])/drag[3],x0=Math.min(drag[0],e.offsetX),x1=Math.max(drag[0],e.offsetX),y0=Math.min(drag[1],e.offsetY),y1=Math.max(drag[1],e.offsetY);if(x1-x0>8&&y1-y0>8)box=[Math.round(b[0]+x0*sx),Math.round(b[1]+y0*sy),Math.round(b[0]+x1*sx),Math.round(b[1]+y1*sy)];drag=null;draw()}})};views.onchange=()=>{i=+views.value;box=null;draw()};prev.onclick=()=>{i=(i+D.items.length-1)%D.items.length;box=null;draw()};next.onclick=()=>{i=(i+1)%D.items.length;box=null;draw()};full.onclick=()=>{box=null;draw()};roi.onclick=()=>{box=D.items[i].box;draw()};structure.onclick=()=>{box=D.items[i].structure_box;draw()};draw();</script></html>""".replace(
        "__DATA__", payload
    )
    (output / "index.html").write_text(document)
    for item in items:
        for tile in item["tiles"]:
            assert (output / tile["path"]).is_file(), tile
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
            text=True,
        )
    )
    stream = probe["streams"][0]
    assert (
        int(stream["nb_frames"]) == 144
        and int(stream["width"]) == 1320
        and int(stream["height"]) == 348
    )
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
