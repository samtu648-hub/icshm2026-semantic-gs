#!/usr/bin/env python3
"""Build the SEM378/382/383 synchronized comparison browser."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json

EID = "SEM-383-DINOV3-VITB-SPATIAL-SFP-ALL300"
BATCH = ROOT / "outputs/quality/SEM-383/r2"
CONTROL = ROOT / "outputs/quality/SEM-382/r1"
BASE = ROOT / "outputs/quality/SEM-378/r1"
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
    diagnostic = read(BATCH / "input-diagnostic-candidate/result.json")
    if audit["status"] != "completed" or diagnostic["status"] != "completed":
        raise ValueError("Completed audit and candidate input diagnostic required")
    output = BATCH / "comparison"
    output.mkdir(exist_ok=False)
    semantic_roots = {
        "SEM378": (BASE / "inference/predictions", output / "sem378"),
        "SEM382": (CONTROL / "inference-6000/predictions", output / "sem382"),
        "SEM383-3000": (BATCH / "inference-3000/predictions", output / "sem383-3000"),
        "SEM383-6000": (BATCH / "inference-6000/predictions", output / "sem383-6000"),
    }
    for _, destination in semantic_roots.values():
        destination.mkdir()
    gt_root = output / "gt"
    gt_root.mkdir()
    no_gt_changes = output / "no-gt-change-vs-sem382"
    no_gt_changes.mkdir()

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
            old = np.asarray(Image.open(CONTROL / "inference-6000/predictions" / name))
            new = np.asarray(Image.open(BATCH / "inference-6000/predictions" / name))
            delta = np.zeros((*old.shape, 3), np.uint8)
            delta[old != new] = [60, 160, 210]
            delta[(old != 2) & (new == 2)] = [245, 210, 65]
            delta[(old == 2) & (new != 2)] = [190, 70, 210]
            Image.fromarray(delta).save(no_gt_changes / name)

    fixed = ["159.png", "160.png", "146.png", "057.png", "064.png"]
    diagnostic_names = ["311.png", "350.png", "356.png"]
    trajectory_focus = [export["trajectory_names"][index] for index in (96, 97, 98)]
    released = sorted(
        name for name in export["image_hashes"] if not name.startswith("frame")
    )
    names = fixed + diagnostic_names + trajectory_focus
    names += [name for name in released if name not in names]
    names += [name for name in export["trajectory_names"] if name not in names]
    items = []
    for name in names:
        item = {"name": name, "tiles": []}
        if (PHOTO / name).exists():
            item["tiles"].append(
                {
                    "label": "Prepared photo",
                    "path": os.path.relpath(PHOTO / name, output),
                }
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
                    "label": "SEM383 vs SEM382: green fix / red harm / purple wrong-to-wrong",
                    "path": os.path.relpath(
                        BATCH / "error-changes-vs-sem382" / name, output
                    ),
                }
            )
        else:
            item["tiles"].append(
                {
                    "label": "No-GT change vs SEM382: yellow added cable / purple removed cable / blue other",
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
        "trajectory_proxy": audit["trajectory_proxy"],
        "input_diagnostic": diagnostic["per_view"],
        "class_names": list(audit["metrics"]["sem383_6000"]["per_class"]),
        "items": items,
        "focus": fixed + diagnostic_names + trajectory_focus,
    }
    write_new_json(output / "data.json", data)
    payload = json.dumps(data, separators=(",", ":"))
    document = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>SEM383 spatial adapter</title>
<style>body{font:15px/1.55 system-ui;margin:24px;background:#eef2f4;color:#17212a}h1{font-size:25px}.note{max-width:1180px;color:#4c5863}.controls{position:sticky;top:0;background:white;padding:10px;z-index:2}button,select{padding:7px;margin-right:8px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.tile{background:white}.tile p{margin:5px;font-size:12px}canvas{width:100%;display:block}table{border-collapse:collapse;background:white}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style>
<h1>冻结 RGB140：SEM378 / SEM382 / SEM383 spatial-prior SFP</h1>
<p class="note">SEM383仅新增514,048参数的deformable-free RGB spatial-prior residual；DINOv3冻结，SFP/decoder LR、300标签、seed、6000采样、768/512协议与SEM382一致。300张分数是训练拟合；100无标签和144轨迹无GT。该实现不是官方DINOv3 Adapter或完整ViT-Adapter。</p><div id="scores"></div>
<p class="note">蓝deck，黄stay_cable，红tower，紫foundation，深灰background。拖拽任意图同步裁剪；宽索包络不表示逐根几何恢复。</p>
<div class="controls"><select id="views"></select><button id="prev">上一图</button><button id="next">下一图</button><button id="full">完整图</button><span id="coords"></span></div><div class="grid" id="grid"></div>
<p><a href="data.json">完整数据</a> · <a href="../audit.json">独立审计</a> · <a href="../verification.json">冻结核验</a> · <a href="../input-diagnostic-candidate/result.json">照片/渲染诊断</a></p>
<script>const D=__DATA__,f=x=>x==null?'n/a':(100*x).toFixed(6),M=D.metrics,arms=['sem378','sem382','sem383_3000','sem383_6000'];scores.innerHTML='<table><tr><th>系统</th><th>mIoU %</th>'+D.class_names.map(x=>'<th>'+x+' IoU %</th>').join('')+'</tr>'+arms.map(a=>'<tr><td>'+a+'</td><td>'+f(M[a].mIoU_with_background)+'</td>'+D.class_names.map(c=>'<td>'+f(M[a].per_class[c].IoU)+'</td>').join('')+'</tr>').join('')+'</table>'+Object.values(D.comparisons).map(x=>'<p>'+x.new_arm+' - '+x.old_arm+': '+x.delta_mIoU_pp.toFixed(6)+' pp; improved/degraded/tied '+x.paired.improved+'/'+x.paired.degraded+'/'+x.paired.tied+'.</p>').join('');let i=0,box=null,drag=null;D.items.forEach((x,j)=>{const o=document.createElement('option');o.value=j;o.textContent=x.name;views.appendChild(o)});const draw=()=>{const x=D.items[i];views.value=i;grid.innerHTML='';coords.textContent=box?box.join(', '):'full';x.tiles.forEach(t=>{const d=document.createElement('div');d.className='tile';d.innerHTML='<p>'+t.label+'</p><canvas></canvas>';grid.appendChild(d);const c=d.querySelector('canvas'),im=new Image;im.onload=()=>{const b=box||[0,0,im.width,im.height];c.width=b[2]-b[0];c.height=b[3]-b[1];c.getContext('2d').drawImage(im,b[0],b[1],c.width,c.height,0,0,c.width,c.height)};im.src=t.path;c.onpointerdown=e=>{drag=[e.offsetX,e.offsetY,c.width,c.height]};c.onpointerup=e=>{if(!drag)return;const b=box||[0,0,1320,989],sx=(b[2]-b[0])/drag[2],sy=(b[3]-b[1])/drag[3],x0=Math.min(drag[0],e.offsetX),x1=Math.max(drag[0],e.offsetX),y0=Math.min(drag[1],e.offsetY),y1=Math.max(drag[1],e.offsetY);if(x1-x0>8&&y1-y0>8)box=[Math.round(b[0]+x0*sx),Math.round(b[1]+y0*sy),Math.round(b[0]+x1*sx),Math.round(b[1]+y1*sy)];drag=null;draw()}})};views.onchange=()=>{i=+views.value;box=null;draw()};prev.onclick=()=>{i=(i+D.items.length-1)%D.items.length;box=null;draw()};next.onclick=()=>{i=(i+1)%D.items.length;box=null;draw()};full.onclick=()=>{box=null;draw()};draw();</script></html>""".replace(
        "__DATA__", payload
    )
    (output / "index.html").write_text(document)
    for item in items:
        for tile in item["tiles"]:
            if not (output / tile["path"]).is_file():
                raise FileNotFoundError(tile)
    write_new_json(
        output / "result.json",
        {
            "schema_version": 1,
            "status": "completed_artifacts_pending_visual_review",
            "experiment_id": EID,
            "views": len(items),
            "trajectory_frames": len(export["trajectory_names"]),
            "image_resources_checked": sum(len(item["tiles"]) for item in items),
            "data_sha256": sha256(output / "data.json"),
            "index_sha256": sha256(output / "index.html"),
            "script_sha256": sha256(Path(__file__)),
        },
    )
    print(output / "index.html")


if __name__ == "__main__":
    main()
