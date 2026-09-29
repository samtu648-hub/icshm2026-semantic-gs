#!/usr/bin/env python3
"""Build a no-GT visual review for released images 301--400.

The page deliberately labels every panel as qualitative-only. Missing panels are
reported rather than silently dropped because RGB152 and RGB154 have different
COLMAP registration coverage.
"""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/quality/SEM-434/r2/review/unlabeled_long_training"
ASSETS = OUT / "assets"
PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)
SYSTEMS = (
    {
        "key": "v1",
        "label": "v1 · RGB152 / SEM434 · 256张标注训练",
        "root": ROOT / "outputs/quality/SEM-434/r2",
        "control": "6000",
        "long": "8000",
    },
    {
        "key": "v2",
        "label": "v2 · RGB154 / SEM435 · 255张标注训练",
        "root": ROOT / "outputs/quality/SEM-435/r2",
        "control": "6000",
        "long": "12000",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thumbnail(image: Image.Image, width: int = 360) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_rgb(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    thumbnail(Image.open(source).convert("RGB")).save(destination, quality=88, optimize=True)


def read_labels(source: Path) -> np.ndarray:
    labels = np.asarray(Image.open(source), dtype=np.uint8)
    if labels.ndim != 2 or int(labels.max(initial=0)) >= len(PALETTE):
        raise ValueError(f"invalid semantic labels: {source}")
    return labels


def write_semantic(labels: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    thumbnail(Image.fromarray(PALETTE[labels], mode="RGB")).save(destination, optimize=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    manifest: dict[str, object] = {
        "scope": "released images 301--400; qualitative only; no ground truth",
        "palette": PALETTE.tolist(),
        "class_names": ["background", "deck", "stay_cable", "tower", "foundation"],
        "systems": [],
        "views": {},
    }

    for system in SYSTEMS:
        manifest["systems"].append(
            {
                "key": system["key"],
                "label": system["label"],
                "control": system["control"],
                "long": system["long"],
            }
        )

    for number in range(301, 401):
        name = f"{number:03d}.png"
        panels: list[str] = []
        view_record: dict[str, object] = {}
        for system in SYSTEMS:
            key = str(system["key"])
            root = Path(system["root"])
            rgb_source = root / "export/images" / name
            control_source = root / "inference/predictions" / str(system["control"]) / name
            long_source = root / "inference/predictions" / str(system["long"]) / name
            available = rgb_source.is_file() and control_source.is_file() and long_source.is_file()
            if not available:
                panels.append(
                    f'<section class="system missing"><h3>{escape(str(system["label"]))}</h3>'
                    '<div class="missing-box">该重建未注册此视角，未生成预测</div></section>'
                )
                view_record[key] = {"available": False}
                continue

            rgb_asset = ASSETS / f"{number:03d}-{key}-rgb.jpg"
            control_asset = ASSETS / f"{number:03d}-{key}-{system['control']}.png"
            long_asset = ASSETS / f"{number:03d}-{key}-{system['long']}.png"
            write_rgb(rgb_source, rgb_asset)
            control_labels = read_labels(control_source)
            long_labels = read_labels(long_source)
            if control_labels.shape != long_labels.shape:
                raise ValueError(f"shape mismatch for {key} {name}")
            write_semantic(control_labels, control_asset)
            write_semantic(long_labels, long_asset)
            changed = int(np.count_nonzero(control_labels != long_labels))
            total = int(control_labels.size)
            panels.append(
                f'<section class="system"><h3>{escape(str(system["label"]))}</h3>'
                '<div class="panels">'
                f'<figure><img loading="lazy" src="assets/{rgb_asset.name}"><figcaption>冻结RGB渲染</figcaption></figure>'
                f'<figure><img loading="lazy" src="assets/{control_asset.name}"><figcaption>{system["control"]}步控制</figcaption></figure>'
                f'<figure><img loading="lazy" src="assets/{long_asset.name}"><figcaption>{system["long"]}步长训</figcaption></figure>'
                '</div>'
                f'<p class="change">端点变化：{changed:,} / {total:,} 像素（{100.0 * changed / total:.4f}%）；无GT，不代表改善。</p>'
                '</section>'
            )
            view_record[key] = {
                "available": True,
                "changed_pixels": changed,
                "total_pixels": total,
                "changed_percent": 100.0 * changed / total,
                "rgb_sha256": sha256(rgb_source),
                "control_prediction_sha256": sha256(control_source),
                "long_prediction_sha256": sha256(long_source),
            }
        cards.append(f'<article class="card"><h2>{name}</h2>{"".join(panels)}</article>')
        manifest["views"][name] = view_record

    legend = "".join(
        f'<span><i style="background:rgb({r},{g},{b})"></i>{idx} {escape(name)}</span>'
        for idx, ((r, g, b), name) in enumerate(
            zip(PALETTE.tolist(), manifest["class_names"], strict=True)
        )
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mask2Former长训 · 无标签301–400视觉审阅</title>
<style>
:root{{--bg:#101318;--panel:#171c23;--line:#303945;--text:#edf2f7;--muted:#a8b3c2;--warn:#ffd166}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:18px 24px;background:#101318f2;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 6px;font-size:22px}} header p{{margin:3px 0;color:var(--muted)}} .warning{{color:var(--warn);font-weight:700}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}} .legend span{{display:flex;align-items:center;gap:5px}} .legend i{{width:15px;height:15px;border:1px solid #fff5;display:inline-block}}
main{{padding:18px;display:grid;gap:18px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}
h2{{margin:0 0 10px}} h3{{margin:10px 0 7px;font-size:16px}} .system+ .system{{margin-top:14px;padding-top:10px;border-top:1px solid var(--line)}}
.panels{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px}} figure{{margin:0}} img{{display:block;width:100%;height:auto;background:#000;border-radius:5px}} figcaption{{padding-top:4px;color:var(--muted)}}
.change{{margin:5px 0 0;color:var(--muted)}} .missing-box{{height:180px;border:1px dashed #6d7785;border-radius:5px;display:grid;place-items:center;color:var(--warn)}}
@media(max-width:850px){{.panels{{grid-template-columns:1fr}} header{{position:static}}}}
</style></head><body><header><h1>Mask2Former长训：无标签301–400视觉审阅</h1>
<p class="warning">这些图没有语义真值；页面只能用于观察变化，不能判断准确率或挑选端点。</p>
<p>v1展示6000→8000，v2展示6000→12000。冻结RGB渲染仅作为模型输入参照；缺失视角表示对应COLMAP重建未注册。</p>
<div class="legend">{legend}</div></header><main>{''.join(cards)}</main></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT / "index.html"), "views": 100}, ensure_ascii=False))


if __name__ == "__main__":
    main()
