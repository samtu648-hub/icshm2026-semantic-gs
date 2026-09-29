#!/usr/bin/env python3
"""Recover RGB137 CPU deliverables from a completed render/metric evaluation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, write_new_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attempt", default="evaluation-recovery-r1")
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / "evaluation"
    out = run / args.attempt
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    source_result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    fixed_cfg = json.loads(
        (ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml").read_text(encoding="utf-8")
    )
    train = json.loads((ROOT / "outputs/quality/all_data/RGB-120-r2/split.json").read_text())["rgb_train"]
    trajectory = [f"frame{i:04d}.png" for i in range(144)]
    frames = source / "browser/frames"
    if source_result.get("status") != "failed_preserved" or "metrics" not in source_result:
        raise RuntimeError("source evaluation is not the expected post-metric preserved failure")
    if len(list((source / "images").glob("*.png"))) != 544 or len(list(frames.glob("*.png"))) != 144:
        raise RuntimeError("source render or trajectory-frame set is incomplete")
    browser = out / "browser"
    browser.mkdir()
    shutil.copytree(source / "browser/panels", browser / "panels")
    video = browser / "trajectory-rgb130-vs-rgb137.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-framerate", "12",
            "-i", str(frames / "frame%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-threads", "4", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ],
        check=True,
    )
    tiles = [
        {"label": "真实照片", "path": "../../../../all_data/RGB-120-r2/scene/images"},
        {"label": "RGB130：303视角初始化", "path": "../../../../backbone_comparison/standard/export/images"},
        {"label": "RGB137：400视角初始化", "path": "../../evaluation/images"},
    ]
    data = {
        "metrics": {"rgb130": source_result["baseline_metrics"], "rgb137": source_result["metrics"]},
        "paired": source_result["paired"],
        "regions": source_result["regions"],
        "fixed": fixed_cfg["views"],
        "train": train,
        "names": fixed_cfg["views"] + [n for n in train if n not in fixed_cfg["views"]] + trajectory,
        "tiles": tiles,
    }
    write_new_json(browser / "data.json", data)
    html = '''<!doctype html><meta charset="utf-8"><title>RGB137 全400初始化对照</title><style>body{font:15px/1.6 system-ui;margin:26px;background:#f4f5f7;color:#18202b}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.tile{background:#fff;padding:9px}canvas,video{width:100%}button,select{padding:7px;margin:4px}.metric{display:inline-block;background:#fff;padding:10px 18px;margin:6px}a{color:#075daf}</style><h1>RGB137：303视角初始化 vs 全400视角初始化</h1><p>两者均为Standard 3DGS、全400张RGB、30k、seed0；唯一主要变量为初始化来源。数值是训练视角拟合，144帧轨迹无GT，仅供视觉检查。</p><div id=m></div><select id=v></select><button id=full>完整图</button><button id=shore>岸线/近景</button><button id=sky>天空</button><span id=b></span><div class=grid id=g></div><h2>同一144帧轨迹</h2><video controls src=trajectory-rgb130-vs-rgb137.mp4></video><p><a href=data.json>完整指标与区域统计</a></p><script>const D=__D__;const F=(x,n=6)=>Number(x).toFixed(n);document.querySelector('#m').innerHTML=Object.entries(D.metrics).map(([k,x])=>`<div class=metric><b>${k}</b><br>PSNR ${F(x.PSNR)}<br>SSIM ${F(x.SSIM)}<br>LPIPS ${F(x.LPIPS)}</div>`).join('');const s=document.querySelector('#v');D.names.forEach(n=>s.add(new Option((D.fixed.includes(n)?'固定 · ':'')+n,n)));let box=[0,0,1320,989],items=[],version=0;function draw(){b.textContent=box.join(', ');for(const [im,c] of items){c.width=box[2]-box[0];c.height=box[3]-box[1];c.getContext('2d').drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)}}function load(){version++;const z=version;items=[];g.replaceChildren();const show=D.train.includes(s.value)?D.tiles:D.tiles.slice(1);for(const t of show){const d=document.createElement('div');d.className='tile';d.append(t.label);const c=document.createElement('canvas'),a=document.createElement('a');a.href=t.path+'/'+s.value;a.textContent=' 打开原生PNG';d.append(c,a);g.append(d);const im=new Image();im.onload=()=>{if(z==version){items.push([im,c]);draw()}};im.src=a.href}}s.onchange=()=>{box=[0,0,1320,989];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};shore.onclick=()=>{box=[0,620,1320,989];draw()};sky.onclick=()=>{box=[0,0,1320,380];draw()};load();</script>'''.replace("__D__", json.dumps(data, ensure_ascii=False))
    (browser / "index.html").write_text(html, encoding="utf-8")
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,duration", "-of", "json", str(video)],
            text=True,
        )
    )
    result = {
        "experiment_id": "RGB-137-STANDARD-ALL400-INIT400",
        "status": "completed_recovered_deliverables",
        "source_evaluation": "evaluation",
        "source_result_sha256": sha256(source / "result.json"),
        "source_failure": source_result["error"],
        "reused_without_rerender": {"candidate_images": 544, "trajectory_frames": 144, "panels": len(list((browser / "panels").glob("*.png")))},
        "metrics": source_result["metrics"],
        "baseline_metrics": source_result["baseline_metrics"],
        "paired": source_result["paired"],
        "regions": source_result["regions"],
        "class_rgb_mae_255": source_result["class_rgb_mae_255"],
        "fresh_fixed9_exact": source_result["fresh_fixed9_exact"],
        "final_gaussian_count": source_result["final_gaussian_count"],
        "model_mib": source_result["model_mib"],
        "browser": "browser/index.html",
        "video": "browser/trajectory-rgb130-vs-rgb137.mp4",
        "video_sha256": sha256(video),
        "video_probe": probe,
        "code_sha256": sha256(Path(__file__)),
        "seconds": time.monotonic() - started,
    }
    write_new_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "video_probe": probe}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
