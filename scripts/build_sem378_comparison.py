#!/usr/bin/env python3
"""Build RGB130/SEM371 versus RGB140/SEM378 local and trajectory comparison."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json

EID = "SEM-378-RGB140-ABSGRAD-ALL300"
BATCH = ROOT / "outputs/quality/SEM-378/r1"
BASELINE = ROOT / "outputs/quality/backbone_comparison/standard"
OLD_BROWSER = ROOT / "outputs/quality/backbone_comparison/browser-r1"
PHOTO = ROOT / "outputs/quality/all_data/RGB-120-r2/scene/images"
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
PALETTE = np.asarray([[30, 30, 30], [80, 160, 245], [245, 210, 65],
                      [220, 80, 80], [170, 100, 190]], dtype=np.uint8)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    audit = read(BATCH / "audit.json")
    export = read(BATCH / "export/result.json")
    assert audit["status"] == "completed" and len(export["image_hashes"]) == 544
    output = BATCH / "comparison"
    output.mkdir(exist_ok=False)
    semantic = output / "sem378"
    semantic.mkdir()
    panels = output / "panels"
    panels.mkdir()
    for name in sorted(export["image_hashes"]):
        prediction = np.asarray(Image.open(BATCH / "inference/predictions" / name))
        assert prediction.shape == (989, 1320) and int(prediction.max()) < 5
        Image.fromarray(PALETTE[prediction]).save(semantic / name)

    source = read(OLD_BROWSER / "data.json")
    previous = {row["name"]: row for row in source["images"]}
    fixed = read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")["diagnostic_views"]
    trajectory = export["trajectory_names"]
    names = fixed + [name for name in sorted(export["image_hashes"])
                     if name not in fixed and name not in trajectory] + trajectory
    arms = [
        ("RGB130 + SEM371", BASELINE / "export/images", OLD_BROWSER / "standard"),
        ("RGB140 + SEM378", BATCH / "export/images", semantic),
    ]

    def panel(name: str, box: list[int] | tuple[int, ...] | None = None) -> Image.Image:
        if box is None:
            box = [0, 0, 1320, 989]
        width = 660
        height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
        header = 24
        canvas = Image.new("RGB", (width * 2, (height + header) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (label, rgb_path, sem_path) in enumerate(arms):
            for row, (kind, path, resample) in enumerate((
                ("RGB", rgb_path, Image.Resampling.LANCZOS),
                ("semantic", sem_path, Image.Resampling.NEAREST),
            )):
                y = row * (height + header)
                draw.text((column * width + 5, y + 5), f"{label} · {kind}", fill="black")
                image = Image.open(path / name).convert("RGB").crop(box).resize((width, height), resample)
                canvas.paste(image, (column * width, y + header))
        return canvas

    for name in fixed:
        item = previous[name]
        panel(name).save(panels / f"{Path(name).stem}-full.png")
        panel(name, item["box"]).save(panels / f"{Path(name).stem}-roi.png")
        panel(name, item["structure_box"]).save(panels / f"{Path(name).stem}-structure.png")
    for index in (0, 36, 71, 72, 73, 75, 96, 108, 143):
        panel(trajectory[index]).save(panels / f"trajectory-{index:03d}.png")

    video = output / "trajectory-rgb130-sem371-rgb140-sem378.mp4"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "1320x1036", "-r", "12", "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-threads", "4", "-crf", "18",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for name in trajectory:
            frame = panel(name)
            assert frame.size == (1320, 1036)
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    assert process.wait() == 0

    items = []
    for name in names:
        record = {
            "name": name,
            "box": previous[name]["box"] if name in previous else [0, 0, 1320, 989],
            "structure_box": previous[name]["structure_box"] if name in previous else [0, 0, 1320, 989],
            "tiles": [
                {"label": "RGB130", "path": os.path.relpath(BASELINE / "export/images" / name, output)},
                {"label": "SEM371", "path": os.path.relpath(OLD_BROWSER / "standard" / name, output)},
                {"label": "RGB140", "path": os.path.relpath(BATCH / "export/images" / name, output)},
                {"label": "SEM378", "path": os.path.relpath(semantic / name, output)},
            ],
        }
        if (PHOTO / name).exists():
            record["tiles"].append({"label": "真实照片", "path": os.path.relpath(PHOTO / name, output)})
        if (LABELS / name).exists():
            gt = output / "gt" / name
            gt.parent.mkdir(exist_ok=True)
            Image.fromarray(PALETTE[np.asarray(Image.open(LABELS / name))]).save(gt)
            record["tiles"].append({"label": "GT", "path": os.path.relpath(gt, output)})
        items.append(record)

    data = {
        "experiment_id": EID,
        "metrics": audit["metrics"],
        "paired": audit["paired"],
        "regions": audit["regions"],
        "trajectory_proxy": audit["trajectory_proxy"],
        "class_names": list(audit["metrics"]["sem378"]["per_class"]),
        "items": items,
        "fixed": fixed,
    }
    write_new_json(output / "data.json", data)
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RGB140 × SEM378 匹配语义对照</title><style>body{font:15px/1.6 system-ui;margin:24px;background:#f5f6f8;color:#17212a}h1{font-size:25px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.controls{position:sticky;top:0;background:white;padding:10px;z-index:2}button,select{padding:7px;margin-right:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.tile{background:white}.tile p{margin:5px;font-size:12px}canvas{width:100%;display:block}video{width:100%}.note{max-width:1100px;color:#47515b}</style><h1>RGB130 + SEM371 vs RGB140 + SEM378</h1><p class="note">同一 Mask2Former Swin-L 初始化、300 张训练标签、768 裁块、6000 次成功更新、seed 20260805；唯一主要变量是 RGB 输入域。指标为训练视角拟合；144 帧轨迹无 GT，只作视觉检查。</p><div id="scores"></div><p class="note">蓝色 deck，黄色 stay_cable，红色 tower，紫色 foundation，深灰 background。拖拽任意图可同步裁剪。</p><div class="controls"><select id="views"></select><button id="full">完整图</button><button id="roi">困难区域</button><button id="structure">结构区域</button><span id="coords"></span></div><div class="grid" id="grid"></div><h2>同一官方坐标 144 帧轨迹</h2><video controls preload="metadata" src="trajectory-rgb130-sem371-rgb140-sem378.mp4"></video><p><a href="data.json">完整指标、区域与轨迹代理</a> · <a href="../audit.json">独立核验</a></p><script>const D=__DATA__;const f=x=>(x*100).toFixed(6);scores.innerHTML='<table><tr><th>系统</th><th>mIoU %</th>'+D.class_names.map(n=>'<th>'+n+' IoU %</th>').join('')+'</tr>'+[['sem371','RGB130 + SEM371'],['sem378','RGB140 + SEM378']].map(([k,l])=>'<tr><td>'+l+'</td><td>'+f(D.metrics[k].mIoU_with_background)+'</td>'+D.class_names.map(n=>'<td>'+f(D.metrics[k].per_class[n].IoU)+'</td>').join('')+'</tr>').join('')+'</table>';let index=0,box=[0,0,1320,989],loaded=[],version=0;D.items.forEach((x,i)=>views.add(new Option((D.fixed.includes(x.name)?'固定 · ':'')+x.name,i)));function draw(){coords.textContent=box.join(', ');for(const {im,c} of loaded){c.width=box[2]-box[0];c.height=box[3]-box[1];const x=c.getContext('2d');x.imageSmoothingEnabled=false;x.drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)}}function load(){const v=++version;loaded=[];grid.replaceChildren();for(const tile of D.items[index].tiles){const div=document.createElement('div');div.className='tile';const p=document.createElement('p');p.textContent=tile.label;const c=document.createElement('canvas');div.append(p,c);grid.append(div);const im=new Image();im.onload=()=>{if(v!==version)return;loaded.push({im,c});draw()};im.src=tile.path;let start=null;c.onpointerdown=e=>{start=[e.offsetX/c.clientWidth,e.offsetY/c.clientHeight];c.setPointerCapture(e.pointerId)};c.onpointerup=e=>{if(!start)return;const end=[Math.max(0,Math.min(1,e.offsetX/c.clientWidth)),Math.max(0,Math.min(1,e.offsetY/c.clientHeight))];if(Math.abs(end[0]-start[0])>.02&&Math.abs(end[1]-start[1])>.02){const [x,y,r,b]=box;box=[Math.round(x+Math.min(start[0],end[0])*(r-x)),Math.round(y+Math.min(start[1],end[1])*(b-y)),Math.round(x+Math.max(start[0],end[0])*(r-x)),Math.round(y+Math.max(start[1],end[1])*(b-y))];draw()}start=null}}}views.onchange=()=>{index=+views.value;box=[0,0,1320,989];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};roi.onclick=()=>{box=[...D.items[index].box];draw()};structure.onclick=()=>{box=[...D.items[index].structure_box];draw()};load();</script></html>'''.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (output / "index.html").write_text(document)
    for item in items:
        for tile in item["tiles"]:
            assert (output / tile["path"]).is_file(), tile
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)], text=True))
    stream = probe["streams"][0]
    assert int(stream["nb_frames"]) == 144 and int(stream["width"]) == 1320 and int(stream["height"]) == 1036
    write_new_json(output / "result.json", {
        "schema_version": 1, "status": "completed_artifacts_pending_visual_review",
        "experiment_id": EID, "frames": 144, "video_sha256": sha256(video),
        "image_resources_checked": sum(len(item["tiles"]) for item in items),
        "script_sha256": sha256(Path(__file__)), "ffprobe": probe,
    })
    print(output / "index.html")


if __name__ == "__main__":
    main()
