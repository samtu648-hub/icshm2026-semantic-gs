#!/usr/bin/env python3
"""Evaluate RGB140 against frozen RGB130, with RGB139 as a secondary comparator."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run.resolve()
    output = run_root / "evaluation"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    base = ROOT / "outputs/quality/backbone_comparison"
    parent = ROOT / "outputs/quality/all_data/RGB-120-r2"
    rgb139_eval = ROOT / "outputs/quality/rgb139/full-r1/evaluation"
    config = ROOT / "experiments/runs/RGB-140-STANDARD-ABSGRAD-ALL400/config.yaml"
    fixed = json.loads((ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml").read_text())
    crops = json.loads((ROOT / "experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml").read_text())["crops"]
    train = json.loads((parent / "split.json").read_text())["rgb_train"]
    baseline_eval_path = base / "standard/rgb-evaluation/result.json"
    baseline_eval = json.loads(baseline_eval_path.read_text())
    baseline_export = json.loads((base / "standard/export/result.json").read_text())
    rgb139_result_path = rgb139_eval / "result.json"
    rgb139 = json.loads(rgb139_result_path.read_text())
    training = json.loads((run_root / "result.json").read_text())
    result = {
        "schema_version": 1, "experiment_id": "RGB-140-STANDARD-ABSGRAD-ALL400",
        "status": "starting", "seed": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config), "run_result_sha256": sha256(run_root / "result.json"),
        "baseline_eval_sha256": sha256(baseline_eval_path), "rgb139_eval_sha256": sha256(rgb139_result_path),
        "code_sha256": sha256(Path(__file__)),
        "quality_scope": "all400 training-view fit; fixed trajectory has no GT and is visual-only",
    }
    try:
        if training["status"] != "completed_all400":
            raise ValueError("RGB140 training run is not complete")
        verify_hashes(run_root, training["artifacts"])
        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"
            ], text=True)
            result["gpu_processes_before"] = apps
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError("Other GPU compute process active")
            source = args.artifact_root / "r1/third_party/standard-3dgs"
            sys.path[:0] = [str(source), str(source / "submodules/diff-gaussian-rasterization"),
                            str(source / "submodules/simple-knn")]
            import numpy as np
            import torch
            from PIL import Image, ImageDraw
            from plyfile import PlyData
            from gaussian_renderer import GaussianModel, render
            from arguments import PipelineParams
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import make_native_view
            from scripts.evaluate_rgb import psnr, ssim, build_lpips_model, lpips_score

            torch.set_num_threads(4)
            torch.manual_seed(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats()
            ply = run_root / "model/point_cloud/iteration_30000/point_cloud.ply"
            model = GaussianModel(3)
            model.load_ply(str(ply))
            params = PipelineParams(argparse.ArgumentParser())
            pipeline = params.extract(argparse.Namespace(convert_SHs_python=False,
                                                           compute_cov3D_python=False,
                                                           debug=False, antialiasing=False))
            background = torch.zeros(3, device="cuda")
            cameras = read_camera_requests(parent / "scene/sparse/0")
            trajectory_root = args.artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
            trajectory = read_camera_requests(trajectory_root)
            requests = {**cameras, **{name: replace(cam, model="PINHOLE", radial_k=0.)
                                     for name, cam in trajectory.items()}}
            if sorted(cameras) != sorted(train) or sorted(trajectory) != baseline_export["trajectory_names"]:
                raise ValueError("Camera universe differs from the fixed protocol")
            result["trajectory_metadata_sha256"] = {
                name: sha256(trajectory_root / name) for name in ("images.txt", "cameras.txt")}
            if result["trajectory_metadata_sha256"] != baseline_export["trajectory_metadata_sha256"]:
                raise ValueError("Trajectory metadata changed")

            def render8(active_model, camera):
                array = render(make_native_view(camera), active_model, pipeline, background)["render"]
                return np.rint(array.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            (output / "images").mkdir()
            hashes, times, rows, regions = {}, [], [], []
            lpips_model, _, device = build_lpips_model("cuda")
            baseline_rows = {row["name"]: row for row in baseline_eval["per_view"]}
            rgb139_rows = {row["name"]: row for row in rgb139["per_view"]}
            classes = ["background", "deck", "stay_cable", "tower", "foundation"]
            class_errors = {name: [] for name in classes}
            with torch.inference_mode():
                for index, (name, camera) in enumerate(sorted(requests.items())):
                    torch.cuda.synchronize(); tick = time.monotonic()
                    image = render8(model, camera)
                    torch.cuda.synchronize(); times.append(time.monotonic() - tick)
                    Image.fromarray(image).save(output / "images" / name)
                    hashes[name] = sha256(output / "images" / name)
                    if name in cameras:
                        photo = np.asarray(Image.open(parent / "scene/images" / name).convert("RGB"))
                        ref, pred = photo.astype(np.float32) / 255, image.astype(np.float32) / 255
                        row = {"name": name, "PSNR": psnr(ref, pred), "SSIM": ssim(ref, pred),
                               "LPIPS": lpips_score(ref, pred, lpips_model, torch, device)}
                        row["delta_vs_rgb130"] = {key: row[key] - baseline_rows[name][key]
                                                  for key in ("PSNR", "SSIM", "LPIPS")}
                        row["delta_vs_rgb139"] = {key: row[key] - rgb139_rows[name][key]
                                                  for key in ("PSNR", "SSIM", "LPIPS")}
                        rows.append(row)
                        old130 = np.asarray(Image.open(base / "standard/export/images" / name))
                        old139 = np.asarray(Image.open(rgb139_eval / "images" / name))
                        label = ROOT / "outputs/quality/all_data/SEM-370-r1/labels" / name
                        if label.exists():
                            semantic = np.asarray(Image.open(label))
                            for class_id, class_name in enumerate(classes):
                                mask = semantic == class_id
                                if mask.any():
                                    class_errors[class_name].append((
                                        float(np.abs(image.astype(float) - photo)[mask].mean()),
                                        float(np.abs(old130.astype(float) - photo)[mask].mean()),
                                        float(np.abs(old139.astype(float) - photo)[mask].mean())))
                        for region_name, region in fixed["regions"].items():
                            if region["view"] != name:
                                continue
                            mask = np.zeros(image.shape[:2], bool)
                            for x0, y0, x1, y1 in region["boxes"]:
                                mask[y0:y1, x0:x1] = True
                            regions.append({"region": region_name, "view": name, "pixels": int(mask.sum()),
                                "rgb130_mae_255": float(np.abs(old130.astype(float) - photo)[mask].mean()),
                                "rgb139_mae_255": float(np.abs(old139.astype(float) - photo)[mask].mean()),
                                "rgb140_mae_255": float(np.abs(image.astype(float) - photo)[mask].mean())})
                    if index % 50 == 0:
                        print(json.dumps({"rendered": index + 1, "total": len(requests)}), flush=True)
            fresh = GaussianModel(3); fresh.load_ply(str(ply))
            with torch.inference_mode():
                for name in fixed["views"]:
                    if not np.array_equal(render8(fresh, cameras[name]), np.asarray(Image.open(output / "images" / name))):
                        raise RuntimeError("Fresh reload render changed")
            metrics = {key: float(np.mean([row[key] for row in rows])) for key in ("PSNR", "SSIM", "LPIPS")}
            paired = {}
            for comparator in ("rgb130", "rgb139"):
                delta_key = f"delta_vs_{comparator}"; paired[comparator] = {}
                for metric in ("PSNR", "SSIM", "LPIPS"):
                    values = [row[delta_key][metric] for row in rows]
                    paired[comparator][metric] = {"mean_delta": float(np.mean(values)),
                        "improved": sum(value > 0 for value in values) if metric != "LPIPS" else sum(value < 0 for value in values),
                        "degraded": sum(value < 0 for value in values) if metric != "LPIPS" else sum(value > 0 for value in values)}
            result.update(status="completed", metrics=metrics, baseline_metrics=baseline_eval["metrics"],
                rgb139_metrics=rgb139["metrics"], paired=paired, per_view=rows, regions=regions,
                class_rgb_mae_255={name: {"rgb140": float(np.mean([x[0] for x in values])),
                                                 "rgb130": float(np.mean([x[1] for x in values])),
                                                 "rgb139": float(np.mean([x[2] for x in values]))}
                                   for name, values in class_errors.items()},
                image_hashes=hashes, train_count=len(rows), trajectory_count=len(trajectory),
                fresh_fixed9_exact=True, render_median_ms=float(np.median(times[1:]) * 1000),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                final_gaussian_count=int(len(PlyData.read(ply)["vertex"])), model_mib=ply.stat().st_size / 2**20)
            del model, fresh, lpips_model
            torch.cuda.empty_cache()

        # CPU-only browser deliverables after releasing the GPU lock.
        visual = output / "browser"; visual.mkdir()
        panels = visual / "panels"; panels.mkdir()
        tiles = [("真实照片", parent / "scene/images"),
                 ("RGB130：原生增密", base / "standard/export/images"),
                 ("RGB139：可靠性初始化", rgb139_eval / "images"),
                 ("RGB140：绝对梯度分裂", output / "images")]
        for name in fixed["views"]:
            default_crop = ([0, 620, 1320, 989] if name in ("159.png", "160.png") else
                            [0, 0, 1320, 380] if name == "174.png" else [0, 0, 1320, 989])
            for suffix, box in (("full", (0, 0, 1320, 989)),
                                ("crop", tuple(crops.get(Path(name).stem, default_crop)))):
                width = 550; height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
                canvas = Image.new("RGB", (width * 4, height + 28), "white"); draw = ImageDraw.Draw(canvas)
                for index, (label, path) in enumerate(tiles):
                    draw.text((index * width + 6, 7), label, fill="black")
                    canvas.paste(Image.open(path / name).crop(box).resize((width, height)), (index * width, 28))
                canvas.save(panels / f"{Path(name).stem}-{suffix}.png")
        frames = visual / "frames"; frames.mkdir()
        for index in range(144):
            name = f"frame{index:04d}.png"
            canvas = Image.new("RGB", (3960, 1018), "white"); draw = ImageDraw.Draw(canvas)
            for column, (label, path) in enumerate(tiles[1:]):
                draw.text((column * 1320 + 7, 7), label, fill="black")
                canvas.paste(Image.open(path / name), (column * 1320, 28))
            canvas.save(frames / name)
        video = visual / "trajectory-rgb130-rgb139-rgb140.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-framerate", "12",
                        "-i", str(frames / "frame%04d.png"), "-c:v", "libx264", "-threads", "4",
                        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(video)], check=True)
        data = {"metrics": {"rgb130": result["baseline_metrics"], "rgb139": result["rgb139_metrics"],
                            "rgb140": result["metrics"]}, "paired": result["paired"],
                "regions": result["regions"], "fixed": fixed["views"], "train": train,
                "names": fixed["views"] + [name for name in train if name not in fixed["views"]] + sorted(trajectory),
                "tiles": [{"label": label, "path": os.path.relpath(path, visual)} for label, path in tiles]}
        write_new_json(visual / "data.json", data)
        html = '''<!doctype html><meta charset="utf-8"><title>RGB140 绝对梯度分裂对照</title><style>body{font:15px/1.6 system-ui;margin:26px;background:#f4f5f7;color:#18202b}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.tile{background:#fff;padding:8px}canvas,video{width:100%}button,select{padding:7px;margin:4px}.metric{display:inline-block;background:#fff;padding:10px 18px;margin:6px}a{color:#075daf}</style><h1>RGB140：Standard 绝对梯度分裂</h1><p>RGB130 与 RGB140 使用同一初始化、全400 RGB、30k、seed0，唯一主要变量是大高斯分裂统计；RGB139 仅作既有次级参照。训练图指标只代表拟合；144帧轨迹无GT。</p><div id=m></div><select id=v></select><button id=full>完整图</button><button id=shore>岸线/近景</button><button id=sky>天空</button><span id=b></span><div class=grid id=g></div><h2>同一144帧轨迹</h2><video controls src=trajectory-rgb130-rgb139-rgb140.mp4></video><p><a href=data.json>完整指标与区域统计</a></p><script>const D=__D__;const F=(x,n=6)=>Number(x).toFixed(n);m.innerHTML=Object.entries(D.metrics).map(([k,x])=>`<div class=metric><b>${k}</b><br>PSNR ${F(x.PSNR)}<br>SSIM ${F(x.SSIM)}<br>LPIPS ${F(x.LPIPS)}</div>`).join('');D.names.forEach(n=>v.add(new Option((D.fixed.includes(n)?'固定 · ':'')+n,n)));let box=[0,0,1320,989],items=[],version=0;function draw(){b.textContent=box.join(', ');for(const [im,c] of items){c.width=box[2]-box[0];c.height=box[3]-box[1];c.getContext('2d').drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)}}function load(){version++;const z=version;items=[];g.replaceChildren();const show=D.train.includes(v.value)?D.tiles:D.tiles.slice(1);for(const t of show){const d=document.createElement('div');d.className='tile';d.append(t.label);const c=document.createElement('canvas'),a=document.createElement('a');a.href=t.path+'/'+v.value;a.textContent=' 打开原生PNG';d.append(c,a);g.append(d);const im=new Image();im.onload=()=>{if(z==version){items.push([im,c]);draw()}};im.src=a.href}}v.onchange=()=>{box=[0,0,1320,989];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};shore.onclick=()=>{box=[0,620,1320,989];draw()};sky.onclick=()=>{box=[0,0,1320,380];draw()};load();</script>'''.replace("__D__", json.dumps(data, ensure_ascii=False))
        (visual / "index.html").write_text(html)
        result.update(browser="browser/index.html", video="browser/trajectory-rgb130-rgb139-rgb140.mp4",
                      video_sha256=sha256(video))
    except BaseException as exc:
        result.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
