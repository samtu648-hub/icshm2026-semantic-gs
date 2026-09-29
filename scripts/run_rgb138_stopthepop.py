#!/usr/bin/env python3
"""Render frozen RGB130 with matched StopThePop global and hierarchical sorting."""
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    source, artifact_root, out = args.source.resolve(), args.artifact_root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    config = ROOT / "experiments/runs/RGB-138-STANDARD-SORT-DIAG/config.yaml"
    base = ROOT / "outputs/quality/backbone_comparison"
    parent = ROOT / "outputs/quality/all_data/RGB-120-r2"
    ply = base / "standard-full-r1/model/point_cloud/iteration_30000/point_cloud.ply"
    baseline_result_path = base / "standard-full-r1/result.json"
    baseline_eval_path = base / "standard/rgb-evaluation/result.json"
    baseline_export_path = base / "standard/export/result.json"
    baseline_result = json.loads(baseline_result_path.read_text())
    baseline_eval = json.loads(baseline_eval_path.read_text())
    baseline_export = json.loads(baseline_export_path.read_text())
    record = {
        "schema_version": 1,
        "experiment_id": "RGB-138-STANDARD-SORT-DIAG",
        "status": "starting",
        "seed": 0,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(config),
        "code_sha256": sha256(Path(__file__)),
        "baseline_result_sha256": sha256(baseline_result_path),
        "baseline_eval_sha256": sha256(baseline_eval_path),
        "baseline_export_sha256": sha256(baseline_export_path),
        "input_ply_sha256": sha256(ply),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_diff": subprocess.check_output(["git", "diff", "--", "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py"], cwd=source, text=True),
        "quality_scope": "render-only attribution; all400 metrics are training-view fit and trajectory has no GT",
    }
    try:
        if baseline_result.get("status") != "completed_all400":
            raise RuntimeError("RGB130 baseline is not complete")
        verify_hashes(base / "standard-full-r1", baseline_result["artifacts"])
        sys.path[:0] = [str(source), str(source / "submodules/diff-gaussian-rasterization"),
                        str(artifact_root / "r1/third_party/standard-3dgs/submodules/simple-knn")]
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from arguments import PipelineParams
        from diff_gaussian_rasterization import ExtendedSettings
        from gaussian_renderer import GaussianModel, render
        from scripts.evaluate_rgb import build_lpips_model, lpips_score, psnr, ssim
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        control_settings = ExtendedSettings()
        hier_settings = ExtendedSettings.from_dict({
            "sort_settings": {"sort_mode": 3, "sort_order": 3,
                              "queue_sizes": {"per_pixel": 4, "tile_2x2": 8, "tile_4x4": 64}},
            "culling_settings": {"rect_bounding": False, "tight_opacity_bounding": False,
                                 "tile_based_culling": False, "hierarchical_4x4_culling": False},
            "load_balancing": False,
            "proper_ewa_scaling": False,
        })
        record["renderer_settings"] = {"global": control_settings.to_dict(), "hierarchical": hier_settings.to_dict()}
        lock_path = ROOT / "outputs/quality/gpu.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                            "--format=csv,noheader"], text=True)
            record["gpu_processes_before"] = apps
            if any(line.strip() and "ToDesk" not in line for line in apps.splitlines()):
                raise RuntimeError("Other GPU compute process active")
            torch.cuda.reset_peak_memory_stats()
            model = GaussianModel(3)
            model.load_ply(str(ply))
            parser = argparse.ArgumentParser(add_help=False)
            pipe_params = PipelineParams(parser)
            pipe = pipe_params.extract(argparse.Namespace(convert_SHs_python=False,
                                                           compute_cov3D_python=False, debug=False))
            background = torch.zeros(3, device="cuda")
            cameras = read_camera_requests(parent / "scene/sparse/0")
            trajectory_root = artifact_root / "results/frozen_scaffold_sem001/flythrough_v1/cameras"
            trajectory = read_camera_requests(trajectory_root)
            requests = {**cameras, **{name: replace(camera, model="PINHOLE", radial_k=0.)
                                     for name, camera in trajectory.items()}}
            train = json.loads((parent / "split.json").read_text())["rgb_train"]
            if sorted(cameras) != sorted(train) or sorted(trajectory) != baseline_export["trajectory_names"]:
                raise RuntimeError("Camera universe differs from RGB130")
            trajectory_hashes = {name: sha256(trajectory_root / name) for name in ("images.txt", "cameras.txt")}
            if trajectory_hashes != baseline_export["trajectory_metadata_sha256"]:
                raise RuntimeError("Trajectory metadata differs from RGB130")
            record["trajectory_metadata_sha256"] = trajectory_hashes
            for dirname in ("global", "hierarchical"):
                (out / dirname).mkdir()

            def render8(camera, settings):
                tensor = render(make_native_view(camera), model, pipe, background,
                                splat_args=settings)["render"]
                return np.rint(tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            times = {"global": [], "hierarchical": []}
            image_hashes = {"global": {}, "hierarchical": {}}
            with torch.inference_mode():
                for index, (name, camera) in enumerate(sorted(requests.items())):
                    for label, settings in (("global", control_settings), ("hierarchical", hier_settings)):
                        torch.cuda.synchronize()
                        t0 = time.monotonic()
                        image = render8(camera, settings)
                        torch.cuda.synchronize()
                        times[label].append(time.monotonic() - t0)
                        path = out / label / name
                        Image.fromarray(image).save(path)
                        image_hashes[label][name] = sha256(path)
                    if index % 50 == 0:
                        print(json.dumps({"rendered_camera_pairs": index + 1, "total": len(requests)}), flush=True)
            original_ply_hash_after = sha256(ply)
            if original_ply_hash_after != record["input_ply_sha256"]:
                raise RuntimeError("Frozen RGB130 PLY changed")
            fresh = GaussianModel(3)
            fresh.load_ply(str(ply))
            old_model, model = model, fresh
            fixed_config = json.loads((ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml").read_text())
            with torch.inference_mode():
                for name in fixed_config["views"]:
                    for label, settings in (("global", control_settings), ("hierarchical", hier_settings)):
                        expected = np.asarray(Image.open(out / label / name))
                        if not np.array_equal(render8(cameras[name], settings), expected):
                            raise RuntimeError(f"Fresh reload mismatch: {label}/{name}")
            del old_model, model, fresh
            lpips_model, _, lpips_device = build_lpips_model("cuda")
            baseline_rows = {row["name"]: row for row in baseline_eval["per_view"]}
            rows = []
            original_delta = []
            for name in sorted(cameras):
                photo = np.asarray(Image.open(parent / "scene/images" / name).convert("RGB"))
                original = np.asarray(Image.open(base / "standard/export/images" / name).convert("RGB"))
                row = {"name": name}
                for label in ("global", "hierarchical"):
                    image = np.asarray(Image.open(out / label / name).convert("RGB"))
                    ref = photo.astype(np.float32) / 255
                    pred = image.astype(np.float32) / 255
                    row[label] = {"PSNR": psnr(ref, pred), "SSIM": ssim(ref, pred),
                                  "LPIPS": lpips_score(ref, pred, lpips_model, torch, lpips_device)}
                row["hierarchical_minus_global"] = {key: row["hierarchical"][key] - row["global"][key]
                                                     for key in ("PSNR", "SSIM", "LPIPS")}
                diff = np.abs(np.asarray(Image.open(out / "global" / name), dtype=np.int16) - original.astype(np.int16))
                original_delta.append(diff)
                row["global_vs_saved_original_mae_255"] = float(diff.mean())
                row["saved_original_metrics"] = {key: baseline_rows[name][key] for key in ("PSNR", "SSIM", "LPIPS")}
                rows.append(row)
            metrics = {label: {key: float(np.mean([row[label][key] for row in rows]))
                               for key in ("PSNR", "SSIM", "LPIPS")}
                       for label in ("global", "hierarchical")}
            paired = {key: {"mean_delta": float(np.mean([row["hierarchical_minus_global"][key] for row in rows])),
                            "improved": int(sum(row["hierarchical_minus_global"][key] > 0 for row in rows)),
                            "degraded": int(sum(row["hierarchical_minus_global"][key] < 0 for row in rows))}
                      for key in ("PSNR", "SSIM", "LPIPS")}
            paired["LPIPS"]["improved"] = int(sum(row["hierarchical_minus_global"]["LPIPS"] < 0 for row in rows))
            paired["LPIPS"]["degraded"] = int(sum(row["hierarchical_minus_global"]["LPIPS"] > 0 for row in rows))
            record.update(
                metrics=metrics,
                saved_rgb130_metrics=baseline_eval["metrics"],
                paired=paired,
                per_view=rows,
                image_hashes=image_hashes,
                global_vs_saved_original={
                    "mean_abs_error_255": float(np.mean([x.mean() for x in original_delta])),
                    "max_abs_error_255": int(max(x.max() for x in original_delta)),
                    "exact_images": int(sum(x.max() == 0 for x in original_delta)),
                },
                train_count=len(cameras), trajectory_count=len(trajectory), fresh_fixed9_exact=True,
                render_median_ms={key: float(np.median(values[1:]) * 1000) for key, values in times.items()},
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
            )
            del lpips_model
            torch.cuda.empty_cache()

        # All visual composition is CPU-only and happens after releasing the GPU lock.
        fixed_config = json.loads((ROOT / "experiments/runs/RGB-132-THREE-BACKBONE-DIAG/config.yaml").read_text())
        crops = json.loads((ROOT / "experiments/runs/QUAL-004-CURRENT-BOTTLENECK/config.yaml").read_text())["crops"]
        browser = out / "browser"
        panels = browser / "panels"
        frames = browser / "frames"
        panels.mkdir(parents=True)
        frames.mkdir()
        tiles = [("真实照片", parent / "scene/images"),
                 ("RGB130原保存渲染", base / "standard/export/images"),
                 ("StopThePop全局排序控制", out / "global"),
                 ("StopThePop层次排序", out / "hierarchical")]
        for name in fixed_config["views"]:
            boxes = [("full", (0, 0, 1320, 989)),
                     ("crop", tuple(crops.get(Path(name).stem,
                                               [0, 620, 1320, 989] if name in ("159.png", "160.png")
                                               else [0, 0, 1320, 380] if name == "174.png"
                                               else [0, 0, 1320, 989])))]
            for suffix, box in boxes:
                width = 550
                height = round(width * (box[3] - box[1]) / (box[2] - box[0]))
                canvas = Image.new("RGB", (width * 4, height + 28), "white")
                draw = ImageDraw.Draw(canvas)
                for index, (label, directory) in enumerate(tiles):
                    draw.text((index * width + 6, 7), label, fill="black")
                    canvas.paste(Image.open(directory / name).crop(box).resize((width, height)), (index * width, 28))
                canvas.save(panels / f"{Path(name).stem}-{suffix}.png")
        trajectory_names = sorted(trajectory)
        temporal = {"global": [], "hierarchical": [], "hierarchical_minus_global_delta_change": []}
        previous = {}
        for index, name in enumerate(trajectory_names):
            current = {label: np.asarray(Image.open(out / label / name), dtype=np.int16)
                       for label in ("global", "hierarchical")}
            if previous:
                for label in ("global", "hierarchical"):
                    temporal[label].append(float(np.abs(current[label] - previous[label]).mean()))
                now_delta = current["hierarchical"] - current["global"]
                old_delta = previous["hierarchical"] - previous["global"]
                temporal["hierarchical_minus_global_delta_change"].append(float(np.abs(now_delta - old_delta).mean()))
            previous = current
            canvas = Image.new("RGB", (2640, 1018), "white")
            draw = ImageDraw.Draw(canvas)
            for column, label in enumerate(("global", "hierarchical")):
                title = "StopThePop全局排序控制" if label == "global" else "StopThePop层次排序"
                draw.text((column * 1320 + 7, 7), title, fill="black")
                canvas.paste(Image.open(out / label / name), (column * 1320, 28))
            canvas.save(frames / name)
        record["trajectory_unwarped_proxy"] = {
            key: {"mean_mae_255": float(np.mean(values)), "p95_mae_255": float(np.quantile(values, .95))}
            for key, values in temporal.items()
        }
        video = browser / "trajectory-global-vs-hierarchical.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-framerate", "12",
                        "-i", str(frames / "frame%04d.png"), "-c:v", "libx264", "-threads", "4",
                        "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(video)], check=True)
        data = {"metrics": record["metrics"], "saved_rgb130_metrics": record["saved_rgb130_metrics"],
                "paired": record["paired"], "global_vs_saved_original": record["global_vs_saved_original"],
                "trajectory_unwarped_proxy": record["trajectory_unwarped_proxy"],
                "fixed": fixed_config["views"], "train": sorted(cameras),
                "names": fixed_config["views"] + [name for name in sorted(cameras) if name not in fixed_config["views"]]
                         + trajectory_names,
                "tiles": [{"label": label, "path": os.path.relpath(directory, browser)} for label, directory in tiles]}
        write_new_json(browser / "data.json", data)
        html = '''<!doctype html><meta charset="utf-8"><title>RGB138 StopThePop排序归因</title><style>body{font:15px/1.6 system-ui;margin:26px;background:#f4f5f7;color:#18202b}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.tile{background:#fff;padding:8px}canvas,video{width:100%}button,select{padding:7px;margin:4px}.metric{display:inline-block;background:#fff;padding:10px 18px;margin:6px}a{color:#075daf}</style><h1>RGB138：冻结RGB130的排序渲染归因</h1><p>模型、相机和高斯参数完全冻结；StopThePop全局控制与层次排序只改变排序。400图是训练视角拟合，144帧无GT。</p><div id=m></div><select id=v></select><button id=full>完整图</button><button id=shore>岸线</button><button id=sky>天空</button><span id=b></span><div class=grid id=g></div><h2>同一144帧轨迹</h2><video controls src=trajectory-global-vs-hierarchical.mp4></video><p><a href=data.json>结构化结果</a></p><script>const D=__D__;const F=(x,n=6)=>Number(x).toFixed(n);m.innerHTML=Object.entries(D.metrics).map(([k,x])=>`<div class=metric><b>${k}</b><br>PSNR ${F(x.PSNR)}<br>SSIM ${F(x.SSIM)}<br>LPIPS ${F(x.LPIPS)}</div>`).join('');D.names.forEach(n=>v.add(new Option((D.fixed.includes(n)?'固定 · ':'')+n,n)));let box=[0,0,1320,989],items=[],version=0;function draw(){b.textContent=box.join(', ');for(const [im,c] of items){c.width=box[2]-box[0];c.height=box[3]-box[1];c.getContext('2d').drawImage(im,box[0],box[1],c.width,c.height,0,0,c.width,c.height)}}function load(){version++;const z=version;items=[];g.replaceChildren();const show=D.train.includes(v.value)?D.tiles:D.tiles.slice(1);for(const t of show){const d=document.createElement('div');d.className='tile';d.append(t.label);const c=document.createElement('canvas'),a=document.createElement('a');a.href=t.path+'/'+v.value;a.textContent=' 原生PNG';d.append(c,a);g.append(d);const im=new Image();im.onload=()=>{if(z==version){items.push([im,c]);draw()}};im.src=a.href}}v.onchange=()=>{box=[0,0,1320,989];load()};full.onclick=()=>{box=[0,0,1320,989];draw()};shore.onclick=()=>{box=[0,620,1320,989];draw()};sky.onclick=()=>{box=[0,0,1320,380];draw()};load()</script>'''.replace("__D__", json.dumps(data, ensure_ascii=False))
        (browser / "index.html").write_text(html)
        record.update(status="completed", browser="browser/index.html", video=str(video.relative_to(out)),
                      video_sha256=sha256(video))
    except BaseException as exc:
        record.update(status="failed_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        write_new_json(out / "result.json", record)


if __name__ == "__main__":
    main()
