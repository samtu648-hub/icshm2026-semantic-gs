#!/usr/bin/env python3
"""Build qualitative RGB170/SEM378/SEM436 review pages.

Images 301--400 and the 144 camera trajectory have no semantic ground truth.
The generated pages therefore expose every long-training endpoint and label all
pixel-change statistics as descriptive rather than accuracy measurements.
"""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/quality/SEM-436/r2"
OUT = RUN / "review"
ASSETS = OUT / "assets"
OLD = ROOT / "outputs/quality/SEM-378/r1/inference/predictions"
PALETTE = np.asarray(
    [[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]],
    dtype=np.uint8,
)
CLASS_NAMES = ["background", "deck", "stay_cable", "tower", "foundation"]
ENDPOINTS = (6000, 8000, 10000, 12000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thumbnail(image: Image.Image, width: int = 320) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.NEAREST)


def read_labels(path: Path) -> np.ndarray:
    labels = np.asarray(Image.open(path), dtype=np.uint8)
    if labels.ndim != 2 or int(labels.max(initial=0)) >= len(PALETTE):
        raise ValueError(f"invalid semantic prediction: {path}")
    return labels


def write_rgb(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    width = 320
    height = max(1, round(image.height * width / image.width))
    image.resize((width, height), Image.Resampling.LANCZOS).save(
        destination, quality=88, optimize=True
    )


def write_semantic(labels: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    thumbnail(Image.fromarray(PALETTE[labels], mode="RGB")).save(destination, optimize=True)


def occupancy(labels: np.ndarray) -> dict[str, float]:
    total = labels.size
    return {
        name: float(np.count_nonzero(labels == class_id) / total)
        for class_id, name in enumerate(CLASS_NAMES)
    }


def card(name: str, group: str, records: dict[str, Any]) -> str:
    panels = [
        (f"../assets/{group}-{name}-rgb.jpg", "RGB170 冻结渲染"),
        (f"../assets/{group}-{name}-sem378.png", "SEM378 · RGB140 · 6000步"),
    ]
    panels.extend(
        (f"../assets/{group}-{name}-{step}.png", f"SEM436 · RGB170 · {step}步")
        for step in ENDPOINTS
    )
    figures = "".join(
        f'<figure><a href="{escape(path)}" target="_blank"><img loading="lazy" src="{escape(path)}"></a>'
        f"<figcaption>{escape(label)}</figcaption></figure>"
        for path, label in panels
    )
    changes = " · ".join(
        f"{step}: {records['endpoints'][str(step)]['changed_from_6000_percent']:.3f}%"
        for step in ENDPOINTS[1:]
    )
    return (
        f'<article class="card"><h2>{escape(name)}</h2><div class="panels">{figures}</div>'
        f'<p>相对同轨迹6000步的标签变化像素：{changes}。无GT，不代表改善。</p></article>'
    )


def write_page(path: Path, title: str, scope: str, cards: list[str]) -> None:
    legend = "".join(
        f'<span><i style="background:rgb({r},{g},{b})"></i>{index} {escape(name)}</span>'
        for index, ((r, g, b), name) in enumerate(zip(PALETTE.tolist(), CLASS_NAMES, strict=True))
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>
:root{{--bg:#101318;--panel:#171c23;--line:#303945;--text:#edf2f7;--muted:#a8b3c2;--warn:#ffd166}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;padding:16px 22px;background:#101318f2;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 5px;font-size:21px}}header p{{margin:3px 0;color:var(--muted)}}.warning{{color:var(--warn);font-weight:700}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}}.legend span{{display:flex;align-items:center;gap:5px}}.legend i{{width:14px;height:14px;border:1px solid #fff5;display:inline-block}}
main{{padding:16px;display:grid;gap:16px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px}}
h2{{margin:0 0 8px;font-size:17px}}.panels{{display:grid;grid-template-columns:repeat(3,minmax(210px,1fr));gap:9px}}figure{{margin:0}}img{{display:block;width:100%;height:auto;background:#000;border-radius:4px}}figcaption{{padding-top:3px;color:var(--muted)}}.card p{{margin:7px 0 0;color:var(--muted)}}
@media(min-width:1500px){{.panels{{grid-template-columns:repeat(6,minmax(180px,1fr))}}}}@media(max-width:760px){{.panels{{grid-template-columns:1fr}}header{{position:static}}}}
</style></head><body><header><h1>{escape(title)}</h1><p class="warning">无语义真值：只能比较输出变化和视觉完整性，不能据此声称准确率提高。</p>
<p>{escape(scope)}。SEM378与SEM436同时改变了RGB重建和训练时长；SEM436的四端点才是严格同轨迹时长对比。</p>
<div class="legend">{legend}</div></header><main>{''.join(cards)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def build_group(group: str, names: list[str], scope: str) -> dict[str, Any]:
    records: dict[str, Any] = {}
    cards: list[str] = []
    for name in names:
        rgb_source = RUN / "export/images" / name
        old_source = OLD / name
        sources = {step: RUN / "inference/predictions" / str(step) / name for step in ENDPOINTS}
        required = [rgb_source, old_source, *sources.values()]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing review inputs for {name}: {missing}")
        stem = Path(name).stem
        write_rgb(rgb_source, ASSETS / f"{group}-{stem}-rgb.jpg")
        old_labels = read_labels(old_source)
        write_semantic(old_labels, ASSETS / f"{group}-{stem}-sem378.png")
        baseline = read_labels(sources[6000])
        endpoint_records: dict[str, Any] = {}
        for step, source in sources.items():
            labels = read_labels(source)
            if labels.shape != baseline.shape or labels.shape != old_labels.shape:
                raise ValueError(f"prediction shape mismatch: {name} endpoint {step}")
            write_semantic(labels, ASSETS / f"{group}-{stem}-{step}.png")
            endpoint_records[str(step)] = {
                "prediction_sha256": sha256(source),
                "changed_from_6000_pixels": int(np.count_nonzero(labels != baseline)),
                "changed_from_6000_percent": float(100 * np.count_nonzero(labels != baseline) / labels.size),
                "occupancy": occupancy(labels),
            }
        records[name] = {
            "rgb_sha256": sha256(rgb_source),
            "sem378_prediction_sha256": sha256(old_source),
            "sem378_occupancy": occupancy(old_labels),
            "endpoints": endpoint_records,
        }
        cards.append(card(stem, group, records[name]))
    write_page(OUT / group / "index.html", f"SEM436 {scope}视觉审阅", scope, cards)
    return records


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    unlabeled = [f"{index:03d}.png" for index in range(301, 401)]
    trajectory = [f"frame{index:04d}.png" for index in range(144)]
    manifest = {
        "schema_version": 1,
        "experiment_id": "SEM-436-RGB170-MASK2FORMER-LONG12K-ALL300",
        "warning": "no semantic ground truth; qualitative/descriptive review only",
        "class_names": CLASS_NAMES,
        "palette": PALETTE.tolist(),
        "unlabeled_301_400": build_group("unlabeled", unlabeled, "无标签301–400"),
        "trajectory_144": build_group("trajectory", trajectory, "无标签144帧轨迹"),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"unlabeled": str(OUT / "unlabeled/index.html"), "trajectory": str(OUT / "trajectory/index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
