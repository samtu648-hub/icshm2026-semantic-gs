#!/usr/bin/env python3
"""Build SEM439 hard-view review and complete no-GT videos."""

from __future__ import annotations

import json
import subprocess
import sys
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
EID = "SEM-439-RGB170-DECK-ABSENCE-FALLBACK"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def semantic(path: Path) -> Image.Image:
    label = np.asarray(Image.open(path), dtype=np.uint8)
    return Image.fromarray(PALETTE[label], mode="RGB")


def panel(rgb_path: Path, native_path: Path, candidate_path: Path, size=(528, 396)) -> bytes:
    images = [
        Image.open(rgb_path).convert("RGB").resize(size, Image.Resampling.LANCZOS),
        semantic(native_path).resize(size, Image.Resampling.NEAREST),
        semantic(candidate_path).resize(size, Image.Resampling.NEAREST),
    ]
    canvas = Image.new("RGB", (size[0] * 3, size[1] + 30), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(zip(images, ("RGB170", "SEM437 native", "SEM439 fallback"), strict=True)):
        x = index * size[0]
        canvas.paste(image, (x, 30))
        draw.text((x + 8, 8), title, fill="white")
    return np.asarray(canvas, dtype=np.uint8).tobytes()


def video(names, rgb_root: Path, native_root: Path, candidate_root: Path, target: Path):
    width, height = 528 * 3, 396 + 30
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", "12", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for name in names:
            process.stdin.write(panel(rgb_root / name, native_root / name, candidate_root / name))
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {target}")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(target)], text=True))
    stream = probe["streams"][0]
    if int(stream["nb_frames"]) != len(names) or int(stream["width"]) != width or int(stream["height"]) != height:
        raise ValueError(f"Video verification failed: {target}")
    return {"path": str(target.relative_to(ROOT)), "frames": len(names), "width": width, "height": height}


def main() -> None:
    config = read(CONFIG)
    output = ROOT / config["output_path"]
    result = read(output / "result.json")
    export_path = ROOT / config["parent_export_result"]
    export = read(export_path)
    rgb_root = export_path.parent / "images"
    native_root = ROOT / config["native_prediction_root"]
    candidate_root = output / "predictions"
    review = output / "review"
    if review.exists():
        raise FileExistsError(review)
    assets = review / "assets"
    videos = review / "videos"
    assets.mkdir(parents=True)
    videos.mkdir()
    cards = []
    for index in range(344, 359):
        name = f"{index:03d}.png"
        figures = []
        sources = (
            (rgb_root / name, f"{index:03d}-rgb.jpg", "RGB170", "rgb"),
            (native_root / name, f"{index:03d}-native.png", "SEM437 native", "sem"),
            (ROOT / config["fallback_prediction_root"] / name, f"{index:03d}-global1p5.png", "global-1.5x", "sem"),
            (candidate_root / name, f"{index:03d}-sem439.png", "SEM439", "sem"),
        )
        for source, filename, label, kind in sources:
            target = assets / filename
            image = Image.open(source).convert("RGB") if kind == "rgb" else semantic(source)
            image.thumbnail((440, 330), Image.Resampling.LANCZOS if kind == "rgb" else Image.Resampling.NEAREST)
            if kind == "rgb":
                image.save(target, quality=90)
            else:
                image.save(target)
            figures.append(f'<figure><a href="assets/{escape(filename)}"><img src="assets/{escape(filename)}"></a><figcaption>{escape(label)}</figcaption></figure>')
        row = result["rows"][name]
        cards.append(f'<article><h2>{index:03d}</h2><div class="grid">{"".join(figures)}</div><p>trigger={str(row["triggered"]).lower()} · added deck={row["added_deck_pixels"]} · changed={100*row["changed_fraction"]:.4f}%</p></article>')
    video_records = {
        "unlabeled": video(export["unlabeled"], rgb_root, native_root, candidate_root, videos / "unlabeled301_400_rgb_native_sem439.mp4"),
        "trajectory": video(export["trajectory"], rgb_root, native_root, candidate_root, videos / "trajectory144_rgb_native_sem439.mp4"),
    }
    html = f'''<!doctype html><meta charset="utf-8"><title>SEM439 deck absence fallback</title><style>
body{{margin:0;background:#101318;color:#edf2f7;font:14px system-ui}}header{{position:sticky;top:0;background:#101318f2;padding:15px;z-index:2}}main{{padding:12px}}article{{border:1px solid #34404d;border-radius:8px;padding:10px;margin:12px 0;background:#171c23}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}figure{{margin:0}}img,video{{width:100%;display:block;background:#000}}figcaption{{color:#aab4c2}}.warn{{color:#ffd166}}@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style><header><h1>SEM439：zero-deck异常回退</h1><p class="warn">规则在查看344–358后设计；301–400和轨迹无GT，只能视觉审阅，不能写成准确率提升。</p><p>仅355、356触发；357及其余542帧保持SEM437 native。红=deck，黄=stay_cable，蓝=tower，橄榄=foundation。</p></header><main><h2>301–400完整视频</h2><video controls preload="metadata" src="videos/unlabeled301_400_rgb_native_sem439.mp4"></video><h2>144帧轨迹（SEM439与native逐像素相同）</h2><video controls preload="metadata" src="videos/trajectory144_rgb_native_sem439.mp4"></video>{''.join(cards)}</main>'''
    (review / "index.html").write_text(html, encoding="utf-8")
    (review / "result.json").write_text(json.dumps({"schema_version": 1, "status": "completed", "videos": video_records, "trigger_names": result["triggers"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
