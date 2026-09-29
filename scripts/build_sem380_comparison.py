#!/usr/bin/env python3
"""Build the complete RGB140/SEM378 versus RGB148/SEM380 review."""

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

EID = "SEM-380-RGB148-GT-REGION-RGB-ALL300"
BATCH = ROOT / "outputs/quality/SEM-380/r2"
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
    semantic = output / "sem380"
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
            gt = np.asarray(Image.open(LABELS / name))
            Image.fromarray(PALETTE[gt]).save(gt_root / name)
        else:
            old = np.asarray(Image.open(BASE / "inference/predictions" / name))
            delta = np.zeros((*old.shape, 3), dtype=np.uint8)
            delta[old != prediction] = [60, 160, 210]
            delta[(old != 2) & (prediction == 2)] = [245, 210, 65]
            delta[(old == 2) & (prediction != 2)] = [190, 70, 210]
            Image.fromarray(delta).save(changes / name)

    source = read(OLD_BROWSER / "data.json")
    previous = {row["name"]: row for row in source["images"]}
    fixed = read(ROOT / "experiments/runs/SEM-370-ALL300/config.yaml")[
        "diagnostic_views"
    ]
    trajectory = export["trajectory_names"]
    released = sorted(
        name for name in export["image_hashes"] if not name.startswith("frame")
    )
    names = fixed + [name for name in released if name not in fixed] + trajectory
    arms = [
        ("RGB140 + SEM378", BASE / "export/images", BASE / "comparison/sem378"),
        ("RGB148 + SEM380", BATCH / "export/images", semantic),
    ]

    def panel(name: str, box: list[int] | tuple[int, ...] | None = None) -> Image.Image:
        box = box or [0, 0, 1320, 989]
        width = 660
        height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
        header = 24
        canvas = Image.new(
            "RGB", (width * 2, (height + header) * 2), cast(Any, "white")
        )
        draw = ImageDraw.Draw(canvas)
        for column, (label, rgb_path, semantic_path) in enumerate(arms):
            for row, (kind, path, resample) in enumerate(
                (
                    ("RGB", rgb_path, Image.Resampling.LANCZOS),
                    ("semantic", semantic_path, Image.Resampling.NEAREST),
                )
            ):
                y = row * (height + header)
                draw.text(
                    (column * width + 5, y + 5), f"{label} · {kind}", fill="black"
                )
                image = (
                    Image.open(path / name)
                    .convert("RGB")
                    .crop(box)
                    .resize((width, height), resample)
                )
                canvas.paste(image, (column * width, y + header))
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

    video = output / "trajectory-rgb140-sem378-rgb148-sem380.mp4"
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
        "1320x1036",
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
            assert frame.size == (1320, 1036)
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    assert process.wait() == 0

    sheets = output / "trajectory-sheets"
    sheets.mkdir()
    for start in range(0, 144, 6):
        sheet = Image.new("RGB", (1320, 6 * 348), cast(Any, "white"))
        draw = ImageDraw.Draw(sheet)
        for row, name in enumerate(trajectory[start : start + 6]):
            for column, (label, path) in enumerate(
                (
                    ("RGB148", BATCH / "export/images"),
                    ("SEM378", BASE / "comparison/sem378"),
                    ("SEM380", semantic),
                )
            ):
                draw.text(
                    (column * 440 + 4, row * 348 + 2), f"{name} {label}", fill="black"
                )
                image = (
                    Image.open(path / name)
                    .convert("RGB")
                    .resize((440, 330), Image.Resampling.NEAREST)
                )
                sheet.paste(image, (column * 440, row * 348 + 18))
        sheet.save(sheets / f"{start:03d}-{start + 5:03d}.jpg", quality=94)
    (sheets / "index.html").write_text(
        '<html><meta charset="utf-8"><h1>全部144帧：RGB148 / SEM378 / SEM380</h1>'
        "<p>无GT；每行同帧，仅作定性变化检查。</p>"
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
            "tiles": [
                {
                    "label": "RGB140",
                    "path": os.path.relpath(BASE / "export/images" / name, output),
                },
                {
                    "label": "RGB148",
                    "path": os.path.relpath(BATCH / "export/images" / name, output),
                },
                {
                    "label": "SEM378",
                    "path": os.path.relpath(BASE / "comparison/sem378" / name, output),
                },
                {"label": "SEM380", "path": os.path.relpath(semantic / name, output)},
            ],
        }
        if (PHOTO / name).exists():
            record["tiles"].insert(
                0, {"label": "真实照片", "path": os.path.relpath(PHOTO / name, output)}
            )
        if (LABELS / name).exists():
            record["tiles"].insert(
                3, {"label": "GT", "path": os.path.relpath(gt_root / name, output)}
            )
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

    unlabeled = audit["unlabeled_proxy"]["per_view"]
    review_names = list(
        dict.fromkeys(
            [
                row["name"]
                for row in sorted(
                    unlabeled, key=lambda row: row["changed_pixels"], reverse=True
                )[:4]
            ]
            + [unlabeled[index]["name"] for index in np.linspace(0, 99, 6, dtype=int)]
            + ["350.png"]
        )
    )
    for name in review_names:
        panel(name).save(panels / f"unlabeled-{Path(name).stem}.png")
    write_new_json(
        output / "unlabeled-proxy.json",
        {
            "status": "completed",
            "scope": "no GT, label changes only; not accuracy",
            "per_view": unlabeled,
            "review_names": review_names,
        },
    )

    data = {
        "experiment_id": EID,
        "metrics": audit["metrics"],
        "paired": audit["paired"],
        "stay_cable_binary_changes": audit["stay_cable_binary_changes"],
        "boundary": audit["boundary"],
        "regions": audit["regions"],
        "trajectory_proxy": audit["trajectory_proxy"],
        "system_cost": audit["system_cost"],
        "input_detail_proxy": audit["input_detail_proxy"],
        "unlabeled_review_names": review_names,
        "class_names": list(audit["metrics"]["rgb148_sem380"]["per_class"]),
        "items": items,
        "fixed": fixed,
    }
    write_new_json(output / "data.json", data)
    document = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>RGB148 × SEM380 完整系统对照</title><style>body{font:15px/1.6 system-ui;margin:24px;background:#f5f6f8;color:#17212a}h1{font-size:25px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}.controls{position:sticky;top:0;background:white;padding:10px;z-index:2}button,select{padding:7px;margin-right:8px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.tile{background:white}.tile p{margin:5px;font-size:12px}canvas{width:100%;display:block}video{width:100%}.note{max-width:1150px;color:#47515b}</style><h1>RGB140 + SEM378 vs RGB148 + SEM380</h1><p class="note">SEM380从同一ADE Swin-L预训练独立初始化；与SEM378保持300标签、原生768随机裁块、seed 20260805、6000成功更新、优化器/损失/推理一致，只更换冻结RGB输入。全300指标是训练拟合；100无标签视角与144轨迹无GT，只作定性变化检查。本页是完整系统比较，不单独归因区域RGB监督。</p><div id="scores"></div><p class="note">蓝deck，黄stay_cable，红tower，紫foundation，深灰background。错误变化：绿修对、红改错、紫为错误类别互换。拖拽任意图同步裁剪。</p><div class="controls"><select id="views"></select><button id="prev">上一图</button><button id="next">下一图</button><button id="full">完整图</button><button id="roi">困难区域</button><button id="structure">结构区域</button><span id="coords"></span></div><div class="grid" id="grid"></div><h2>同一官方坐标144帧</h2><p><a href="trajectory-sheets/index.html">全部144帧逐帧表</a>；重点检查96–98。轨迹无GT。</p><video controls preload="metadata" src="trajectory-rgb140-sem378-rgb148-sem380.mp4"></video><p><a href="data.json">完整指标与资源</a> · <a href="../audit.json">独立审计</a> · <a href="unlabeled-proxy.json">100无GT变化</a></p><script>const D=__DATA__;const f=x=>(100*x).toFixed(6);scores.innerHTML='<table><tr><th>系统</th><th>mIoU %</th>'+D.class_names.map(n=>'<th>'+n+' IoU %</th>').join('')+'</tr>'+[['rgb140_sem378','RGB140 + SEM378'],['rgb148_sem380','RGB148 + SEM380']].map(([k,l])=>'<tr><td>'+l+'</td><td>'+f(D.metrics[k].mIoU_with_background)+'</td>'+D.class_names.map(n=>'<td>'+f(D.metrics[k].per_class[n].IoU)+'</td>').join('')+'</tr>').join('')+'</table>';let index=0,box=[0,0,1320,989],loaded=[],version=0;D.items.forEach((x,i)=>views.add(new Option((D.fixed.includes(x.name)?'固定 · ':'')+x.name,i)));function draw(){coords.textContent=JSON.stringify(box);for(const x of loaded){x.c.width=box[2]-box[0];x.c.height=box[3]-box[1];x.c.getContext('2d').drawImage(x.im,box[0],box[1],x.c.width,x.c.height,0,0,x.c.width,x.c.height)}}function load(){const token=++version;loaded=[];grid.replaceChildren();const item=D.items[index];for(const t of item.tiles){const d=document.createElement('div');d.className='tile';d.innerHTML='<p>'+t.label+'</p><canvas></canvas>';grid.append(d);const im=new Image;const c=d.querySelector('canvas');im.onload=()=>{if(token===version){loaded.push({im,c});draw()}};im.src=t.path}views.value=index;box=[0,0,1320,989]}views.onchange=()=>{index=+views.value;load()};prev.onclick=()=>{index=(index+D.items.length-1)%D.items.length;load()};next.onclick=()=>{index=(index+1)%D.items.length;load()};full.onclick=()=>{box=[0,0,1320,989];draw()};roi.onclick=()=>{box=D.items[index].box;draw()};structure.onclick=()=>{box=D.items[index].structure_box;draw()};grid.addEventListener('mousedown',e=>{if(e.target.tagName!=='CANVAS')return;const c=e.target,r=c.getBoundingClientRect(),sx=box[0]+(e.clientX-r.left)/r.width*(box[2]-box[0]),sy=box[1]+(e.clientY-r.top)/r.height*(box[3]-box[1]);const move=q=>{let ex=box[0]+(q.clientX-r.left)/r.width*(box[2]-box[0]),ey=box[1]+(q.clientY-r.top)/r.height*(box[3]-box[1]);box=[Math.round(Math.min(sx,ex)),Math.round(Math.min(sy,ey)),Math.round(Math.max(sx,ex)),Math.round(Math.max(sy,ey))];draw()};const up=()=>{removeEventListener('mousemove',move);removeEventListener('mouseup',up)};addEventListener('mousemove',move);addEventListener('mouseup',up)});load();</script>""".replace(
        "__DATA__", json.dumps(data, ensure_ascii=False)
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
    assert int(stream["nb_frames"]) == 144
    assert (int(stream["width"]), int(stream["height"])) == (1320, 1036)
    write_new_json(
        output / "result.json",
        {
            "schema_version": 1,
            "status": "completed_artifacts_pending_visual_review",
            "experiment_id": EID,
            "frames": 144,
            "video_sha256": sha256(video),
            "image_resources_checked": sum(len(item["tiles"]) for item in items),
            "review_names": review_names,
            "script_sha256": sha256(Path(__file__)),
            "ffprobe": probe,
        },
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
