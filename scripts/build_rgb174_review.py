#!/usr/bin/env python3
"""Build the RGB170/RGB174 trajectory video and compact interactive review."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run = ROOT / "outputs/quality/RGB-174/full-r1"
    evaluation = json.loads((run / "evaluation-r1/result.json").read_text())
    trajectory = json.loads((run / "trajectory-r1/result.json").read_text())
    output = run / "review-r1"
    output.mkdir(parents=True, exist_ok=False)
    video = output / "rgb170-vs-rgb174-trajectory.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-n",
        "-framerate", "12", "-i", str(ROOT / "outputs/quality/RGB-170/full-r1/trajectory-r1/50000/frame%04d.png"),
        "-framerate", "12", "-i", str(run / "trajectory-r1/50000/frame%04d.png"),
        "-filter_complex_threads", "2", "-filter_complex", "[0:v][1:v]hstack=inputs=2,pad=iw:ceil(ih/2)*2[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], check=True)
    probe_frames = (36, 48, 60)
    probe = Image.new("RGB", (1320 * 2, (989 + 34) * len(probe_frames)), "white")
    draw = ImageDraw.Draw(probe)
    for row, frame in enumerate(probe_frames):
        y = row * (989 + 34)
        for column, (label, root) in enumerate((
            ("RGB170 50k", ROOT / "outputs/quality/RGB-170/full-r1/trajectory-r1/50000"),
            ("RGB174 50k", run / "trajectory-r1/50000"),
        )):
            x = column * 1320
            draw.text((x + 8, y + 8), f"frame{frame:04d} · {label}", fill="black")
            probe.paste(Image.open(root / f"frame{frame:04d}.png").convert("RGB"), (x, y + 34))
    probe.save(output / "trajectory-fog-probe-0036-0048-0060.jpg", quality=94)
    candidate = "RGB174_50000"
    baseline = "RGB170_50000"
    per_view = evaluation["per_view"]
    ranking = sorted(per_view, key=lambda row: abs(row[candidate]["PSNR"] - row[baseline]["PSNR"]), reverse=True)
    names = list(dict.fromkeys(
        ["054.png", "061.png", "081.png", "088.png", "140.png", "145.png", "159.png", "160.png", "174.png"]
        + [f"{value:03d}.png" for value in range(235, 246)]
        + [f"{value:03d}.png" for value in range(385, 393)]
        + [row["name"] for row in ranking[:20]]
    ))
    data = {
        "names": names,
        "all400": evaluation["all400"],
        "delta": evaluation["candidate_minus_rgb170"],
        "trajectory": trajectory,
        "tiles": [
            {"label": "photo", "path": "../../../all_data/RGB-120-r2/scene/images"},
            {"label": "RGB166 50k", "path": "../../../RGB-166/full-r1/evaluation-r1/images/50000"},
            {"label": "RGB170 50k", "path": "../../../RGB-170/full-r1/evaluation-r1/images/50000"},
            {"label": "RGB174 50k", "path": "../evaluation-r1/images/50000"},
        ],
    }
    document = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RGB174桥梁保护EAS复核</title><style>body{font:15px/1.55 system-ui;margin:24px;background:#f4f5f7;color:#18202b}h1{font-size:25px}.note{max-width:1100px;color:#46515d}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.tile{background:white;padding:7px}.tile p{margin:3px}canvas,video,img{width:100%;display:block}.controls{position:sticky;top:0;background:white;padding:9px;z-index:2}button,select{padding:7px;margin-right:7px}pre{background:white;padding:10px;overflow:auto}</style><h1>RGB174：桥梁硬保护、背景可靠性EAS</h1><p class="note">照片 / RGB166 / RGB170 / RGB174同位置。全400数字是训练视角拟合；144帧轨迹无GT，只用于观察大桥连续性、白雾/黄雾和跳变，不能称未知视角准确率。</p><pre id="metrics"></pre><div class="controls"><select id="views"></select><button id="full">完整图</button><button id="water">水面/近景</button><button id="sky">天空/远景</button><span id="coords"></span></div><div class="grid" id="grid"></div><h2>固定144帧轨迹</h2><p>左RGB170，右RGB174，12fps。</p><video controls preload="metadata" src="rgb170-vs-rgb174-trajectory.mp4"></video><h2>用户指出雾块附近的固定帧</h2><p>0036、0048、0060；每行左RGB170、右RGB174。无GT，只作同位置视觉判断。</p><img src="trajectory-fog-probe-0036-0048-0060.jpg"><script>const D=__DATA__;metrics.textContent=JSON.stringify({all400:D.all400,delta_vs_RGB170:D.delta},null,2);D.names.forEach(n=>views.add(new Option(n,n)));let box=[0,0,1320,989],loaded=[],token=0;function draw(){coords.textContent=box.join(', ');for(const x of loaded){x.c.width=box[2]-box[0];x.c.height=box[3]-box[1];x.c.getContext('2d').drawImage(x.im,box[0],box[1],x.c.width,x.c.height,0,0,x.c.width,x.c.height)}}function load(){const t=++token;loaded=[];grid.replaceChildren();for(const item of D.tiles){const d=document.createElement('div');d.className='tile';d.innerHTML='<p>'+item.label+'</p><canvas></canvas>';grid.append(d);const c=d.querySelector('canvas'),im=new Image;im.onload=()=>{if(t===token){loaded.push({c,im});draw()}};im.src=item.path+'/'+views.value}}views.onchange=()=>{box=[0,0,1320,989];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};water.onclick=()=>{box=[0,320,1320,989];draw()};sky.onclick=()=>{box=[0,0,1320,420];draw()};views.value=D.names[0];load();</script></html>'''.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    (output / "index.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
