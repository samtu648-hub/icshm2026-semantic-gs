#!/usr/bin/env python3
"""Compare audited SEM437 endpoints with SEM436 and build no-GT reviews."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "outputs/quality/SEM-436/r2"
CANDIDATE = ROOT / "outputs/quality/SEM-437/r2"
OUT = CANDIDATE / "review"
ASSETS = OUT / "assets"
ENDPOINTS = (6000, 8000, 10000, 12000)
CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")
PALETTE = np.asarray([[0, 0, 0], [230, 86, 37], [255, 210, 0], [38, 139, 210], [133, 153, 0]], dtype=np.uint8)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def labels(path: Path):
    value = np.asarray(Image.open(path), dtype=np.uint8)
    if value.ndim != 2 or value.max(initial=0) >= 5:
        raise ValueError(path)
    return value


def metric_row(record):
    metrics = record["metrics"]
    return {
        "mIoU_with_background": metrics["mIoU_with_background"],
        "mIoU_without_background": metrics["mIoU_without_background"],
        "per_class_iou": {name: metrics["per_class"][name]["IoU"] for name in CLASSES},
        "stay_cable_precision": metrics["per_class"]["stay_cable"]["precision"],
        "stay_cable_recall": metrics["per_class"]["stay_cable"]["recall"],
        "stay_cable_fp": metrics["per_class"]["stay_cable"]["fp"],
        "stay_cable_fn": metrics["per_class"]["stay_cable"]["fn"],
        "boundary_f1": record["boundary_metrics"]["boundary_f1"],
    }


def numeric_delta(candidate, control):
    result = {}
    for key, value in candidate.items():
        if isinstance(value, dict):
            result[key] = numeric_delta(value, control[key])
        elif isinstance(value, (int, float)):
            result[key] = value - control[key]
    return result


def save_rgb(source: Path, target: Path):
    image = Image.open(source).convert("RGB")
    image.thumbnail((360, 270), Image.Resampling.LANCZOS)
    image.save(target, quality=88)


def save_sem(source: Path, target: Path):
    image = Image.fromarray(PALETTE[labels(source)], mode="RGB")
    image.thumbnail((360, 270), Image.Resampling.NEAREST)
    image.save(target)


def build_page(group: str, names: list[str], title: str):
    group_dir = OUT / group
    group_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    records = {}
    for name in names:
        stem = Path(name).stem
        rgb_asset = ASSETS / f"{group}-{stem}-rgb.jpg"
        save_rgb(CANDIDATE / "export/images" / name, rgb_asset)
        control_source = CONTROL / "inference/predictions/12000" / name
        control_asset = ASSETS / f"{group}-{stem}-sem436-12000.png"
        save_sem(control_source, control_asset)
        control_label = labels(control_source)
        panels = [(rgb_asset, "RGB170"), (control_asset, "SEM436 12k控制")]
        endpoint_record = {}
        for step in ENDPOINTS:
            source = CANDIDATE / f"inference/predictions/{step}" / name
            asset = ASSETS / f"{group}-{stem}-sem437-{step}.png"
            save_sem(source, asset)
            value = labels(source)
            endpoint_record[str(step)] = {
                "changed_from_sem436_12000_fraction": float(np.mean(value != control_label)),
                "class_pixels": np.bincount(value.ravel(), minlength=5).tolist(),
            }
            panels.append((asset, f"SEM437 {step//1000}k"))
        records[name] = endpoint_record
        figures = "".join(
            f'<figure><a href="../assets/{escape(path.name)}"><img loading="lazy" src="../assets/{escape(path.name)}"></a><figcaption>{escape(label)}</figcaption></figure>'
            for path, label in panels
        )
        change = 100 * endpoint_record["12000"]["changed_from_sem436_12000_fraction"]
        cards.append(f'<article><h2>{escape(stem)}</h2><div class="grid">{figures}</div><p>SEM437-12k相对SEM436-12k变化 {change:.3f}%；无GT，不代表改善。</p></article>')
    legend = " ".join(f'<span><i style="background:rgb({r},{g},{b})"></i>{i} {escape(name)}</span>' for i, ((r,g,b),name) in enumerate(zip(PALETTE.tolist(), CLASSES, strict=True)))
    html = f'''<!doctype html><meta charset="utf-8"><title>{escape(title)}</title><style>
body{{margin:0;background:#101318;color:#edf2f7;font:14px system-ui}}header{{position:sticky;top:0;background:#101318f2;padding:15px;z-index:2}}main{{padding:12px}}article{{border:1px solid #34404d;border-radius:8px;padding:10px;margin:12px 0;background:#171c23}}h1,h2,p{{margin:5px 0}}.warn{{color:#ffd166}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}figure{{margin:0}}img{{width:100%;display:block;background:#000}}figcaption{{color:#aab4c2}}span{{margin-right:12px}}i{{display:inline-block;width:12px;height:12px;margin-right:4px}}@media(min-width:1500px){{.grid{{grid-template-columns:repeat(6,1fr)}}}}
</style><header><h1>{escape(title)}</h1><p class="warn">301–400与轨迹均无GT；页面只比较结构完整性和输出变化，不能声称准确率。</p><p>{legend}</p></header><main>{''.join(cards)}</main>'''
    (group_dir / "index.html").write_text(html, encoding="utf-8")
    return records


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    control = read(CONTROL / "inference/result.json")
    candidate = read(CANDIDATE / "inference/result.json")
    endpoint_rows = {}
    for step in ENDPOINTS:
        control_row = metric_row(control["endpoints"][str(step)])
        candidate_row = metric_row(candidate["endpoints"][str(step)])
        cviews = {row["name"]: row for row in control["endpoints"][str(step)]["per_view"]}
        wins = sum(row["mIoU_without_background"] > cviews[row["name"]]["mIoU_without_background"] for row in candidate["endpoints"][str(step)]["per_view"])
        endpoint_rows[str(step)] = {"control": control_row, "candidate": candidate_row, "delta": numeric_delta(candidate_row, control_row), "foreground_miou_view_wins": wins, "view_count": 300}
    hard = build_page("hard344_358", [f"{i:03d}.png" for i in range(344, 359)], "SEM437 344–358困难视角")
    unlabeled = build_page("unlabeled", [f"{i:03d}.png" for i in range(301, 401)], "SEM437 301–400完整审阅")
    trajectory = build_page("trajectory", [f"frame{i:04d}.png" for i in range(144)], "SEM437 144帧轨迹审阅")
    payload = {"schema_version": 1, "status": "completed", "experiment_id": "SEM-437-RGB170-MASK2FORMER-CONTEXT50-LONG12K-ALL300", "quality_scope": "all300 training fit; no GT for 301-400 or trajectory", "endpoints_vs_sem436": endpoint_rows, "hard344_358": hard, "unlabeled": unlabeled, "trajectory": trajectory}
    (CANDIDATE / "comparison-vs-sem436.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
