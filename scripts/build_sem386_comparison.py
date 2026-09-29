#!/usr/bin/env python3
"""Audit sealed SEM386 against SEM384 and build the native Dev30 review page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.semantic_render_domain import SEMANTIC_PALETTE

CONFIG = ROOT / "experiments/runs/SEM-386-RGB150-DEV270-CONTEXT50-MEMORY/config.yaml"
CLASS_NAMES = ("background", "deck", "stay_cable", "tower", "foundation")


def read(path):
    return json.loads(path.read_text())


def confusion(target, prediction):
    if target.shape != prediction.shape or target.max() > 4 or prediction.max() > 4:
        raise ValueError("Invalid native class mask")
    return np.bincount(
        5 * target.astype(np.int64).ravel() + prediction.ravel(), minlength=25
    ).reshape(5, 5)


def scalar_delta(a, b):
    return None if a is None or b is None else 100 * (b - a)


def aggregate_scores(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    union = matrix.sum(0) + matrix.sum(1) - np.diag(matrix)
    iou = np.divide(
        np.diag(matrix),
        union,
        out=np.full(5, np.nan, dtype=np.float64),
        where=union > 0,
    )
    return {
        "five_class_miou": float(np.nanmean(iou)),
        "foreground_miou": float(np.nanmean(iou[1:])),
        "stay_cable_iou": float(iou[2]),
    }


def boundary_counts(target, prediction):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    cable = target == 2
    gt_boundary = cable ^ binary_erosion(cable)
    predicted_cable = prediction == 2
    predicted_boundary = predicted_cable ^ binary_erosion(predicted_cable)
    gt_distance = np.asarray(distance_transform_edt(~gt_boundary))
    predicted_distance = np.asarray(distance_transform_edt(~predicted_boundary))
    return np.asarray(
        [
            np.sum(predicted_boundary & (gt_distance <= 2)),
            np.sum(predicted_boundary),
            np.sum(gt_boundary & (predicted_distance <= 2)),
            np.sum(gt_boundary),
        ],
        dtype=np.int64,
    )


def boundary_f1(counts):
    matched_prediction, predicted, matched_target, target = counts
    precision = matched_prediction / predicted if predicted else np.nan
    recall = matched_target / target if target else np.nan
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else np.nan
    )


def paired_bootstrap(records, seed, draws=10000):
    if len(records) != 30:
        raise ValueError("Paired bootstrap requires exactly 30 holdout views")
    rng = np.random.default_rng(seed)
    deltas = {
        "five_class_miou_pp": np.empty(draws),
        "foreground_miou_pp": np.empty(draws),
        "stay_cable_iou_pp": np.empty(draws),
        "stay_cable_boundary_f1_pp": np.empty(draws),
    }
    for draw in range(draws):
        selected = rng.integers(0, len(records), len(records))
        baseline_confusion = sum(
            (records[index][0] for index in selected), np.zeros((5, 5), np.int64)
        )
        candidate_confusion = sum(
            (records[index][1] for index in selected), np.zeros((5, 5), np.int64)
        )
        baseline_scores = aggregate_scores(baseline_confusion)
        candidate_scores = aggregate_scores(candidate_confusion)
        for key in ("five_class_miou", "foreground_miou", "stay_cable_iou"):
            deltas[f"{key}_pp"][draw] = 100 * (
                candidate_scores[key] - baseline_scores[key]
            )
        baseline_boundary = sum(
            (records[index][2] for index in selected), np.zeros(4, np.int64)
        )
        candidate_boundary = sum(
            (records[index][3] for index in selected), np.zeros(4, np.int64)
        )
        deltas["stay_cable_boundary_f1_pp"][draw] = 100 * (
            boundary_f1(candidate_boundary) - boundary_f1(baseline_boundary)
        )
    return {
        "method": "paired_view_bootstrap_with_replacement",
        "views": len(records),
        "draws": draws,
        "seed": seed,
        "interval": "percentile_95",
        "metrics": {
            key: {
                "lower_pp": float(np.nanpercentile(values, 2.5)),
                "median_pp": float(np.nanpercentile(values, 50.0)),
                "upper_pp": float(np.nanpercentile(values, 97.5)),
                "probability_strictly_positive": float(np.nanmean(values > 0)),
            }
            for key, values in deltas.items()
        },
    }


def contact_sheets(output, rows):
    hashes = {}
    holdout = [row for row in rows if row["role"] == "holdout"]
    for group in ("regular-local", "oblique-local", "distant-local"):
        group_rows = [row for row in holdout if row["group"] == group]
        if len(group_rows) != 10:
            raise ValueError(f"Expected 10 holdout views for {group}")
        tile_width, tile_height, caption_height = 264, 198, 18
        sheet = Image.new(
            "RGB",
            (tile_width * 5, 30 + len(group_rows) * (tile_height + caption_height)),
        )
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, sheet.width, sheet.height), fill=(245, 244, 237))
        draw.text(
            (8, 8),
            f"{group}: RGB | GT | control | candidate | change",
            fill=(23, 59, 56),
        )
        for row_index, row in enumerate(group_rows):
            y = 30 + row_index * (tile_height + caption_height)
            for column, (label, directory) in enumerate(
                (
                    ("RGB", "rgb"),
                    ("GT", "gt"),
                    ("control", "baseline"),
                    ("candidate", "candidate"),
                    ("change", "change"),
                )
            ):
                image = Image.open(output / directory / row["name"]).convert("RGB")
                image = image.resize(
                    (tile_width, tile_height), Image.Resampling.LANCZOS
                )
                x = column * tile_width
                sheet.paste(image, (x, y))
                draw.text(
                    (x + 3, y + tile_height + 2),
                    f"{row['name']} {label}",
                    fill=(23, 59, 56),
                )
        path = output / f"review-{group}.jpg"
        sheet.save(path, quality=92)
        hashes[path.name] = sha256(path)
    return hashes


def main(config_path: Path = CONFIG):
    config = read(config_path)
    batch = ROOT / config["output_path"]
    parent = ROOT / config["parent_semantic_run"]
    candidate = read(batch / "inference/result.json")
    baseline = read(parent / "inference/result.json")
    training = read(batch / "mask2former/result.json")
    if (
        candidate["status"] != "completed"
        or baseline["status"] != "completed"
        or training["status"] != "completed_sealed_6000"
    ):
        raise ValueError("Only sealed, fully evaluated experiments may be compared")
    if candidate["split_sha256"] != baseline["split_sha256"]:
        raise ValueError("Different evaluation splits")
    verify_hashes(batch / "inference/predictions", candidate["prediction_hashes"])
    verify_hashes(parent / "inference/predictions", baseline["prediction_hashes"])
    output = batch / "comparison"
    output.mkdir(exist_ok=False)
    for name in ["baseline", "candidate", "gt", "change"]:
        (output / name).mkdir()
    rgb = ROOT / config["rgb_evaluation"] / "images"
    (output / "rgb").symlink_to(os.path.relpath(rgb, output), target_is_directory=True)
    labels = ROOT / config["label_root"]
    baseline_views = {row["name"]: row for row in baseline["per_view"]}
    candidate_views = {row["name"]: row for row in candidate["per_view"]}
    if baseline_views.keys() != candidate_views.keys():
        raise ValueError("Different labeled evaluation images")
    rows, audit_count = [], 0
    bootstrap_records = []
    error_transitions = {
        key: {
            "corrected_by_target_class": np.zeros(5, dtype=np.int64),
            "damaged_by_target_class": np.zeros(5, dtype=np.int64),
            "wrong_to_wrong_prediction_matrix": np.zeros((5, 5), dtype=np.int64),
        }
        for key in (
            "train",
            "holdout",
            "regular-local",
            "oblique-local",
            "distant-local",
        )
    }
    counts = {
        role: dict(
            corrected_pixels=0,
            damaged_pixels=0,
            improved_views=0,
            worse_views=0,
            tied_views=0,
        )
        for role in ["train", "holdout"]
    }
    for name in sorted(candidate["prediction_hashes"]):
        a = np.asarray(
            Image.open(parent / "inference/predictions" / name), dtype=np.uint8
        )
        b = np.asarray(
            Image.open(batch / "inference/predictions" / name), dtype=np.uint8
        )
        Image.fromarray(SEMANTIC_PALETTE[a]).save(output / "baseline" / name)
        Image.fromarray(SEMANTIC_PALETTE[b]).save(output / "candidate" / name)
        row = dict(name=name, role="unlabeled", disagreement_pixels=int((a != b).sum()))
        change = np.full((*a.shape, 3), 28, dtype=np.uint8)
        if name in candidate_views:
            x, y = baseline_views[name], candidate_views[name]
            if x["role"] != y["role"] or x["group"] != y["group"]:
                raise ValueError("Role or group drift")
            target = np.asarray(Image.open(labels / name), dtype=np.uint8)
            for prediction, record in [(a, x), (b, y)]:
                np.testing.assert_array_equal(
                    confusion(target, prediction), record["confusion"]
                )
                audit_count += 1
            Image.fromarray(SEMANTIC_PALETTE[target]).save(output / "gt" / name)
            corrected, damaged = (
                (a != target) & (b == target),
                (a == target) & (b != target),
            )
            keys = [y["role"]] + ([y["group"]] if y["role"] == "holdout" else [])
            wrong_swap = (a != target) & (b != target) & (a != b)
            for key in keys:
                transition = error_transitions[key]
                transition["corrected_by_target_class"] += np.bincount(
                    target[corrected], minlength=5
                )
                transition["damaged_by_target_class"] += np.bincount(
                    target[damaged], minlength=5
                )
                transition["wrong_to_wrong_prediction_matrix"] += np.bincount(
                    5 * a[wrong_swap].astype(np.int64) + b[wrong_swap], minlength=25
                ).reshape(5, 5)
            if y["role"] == "holdout":
                bootstrap_records.append(
                    (
                        np.asarray(x["confusion"], dtype=np.int64),
                        np.asarray(y["confusion"], dtype=np.int64),
                        boundary_counts(target, a),
                        boundary_counts(target, b),
                    )
                )
            change[corrected] = [28, 170, 109]
            change[damaged] = [229, 72, 69]
            change[(a != target) & (b != target)] = [184, 144, 40]
            delta = scalar_delta(x["mIoU_with_background"], y["mIoU_with_background"])
            if delta is None:
                raise ValueError("Labeled view mIoU delta is undefined")
            row.update(
                role=y["role"],
                group=y["group"],
                baseline_miou=x["mIoU_with_background"],
                candidate_miou=y["mIoU_with_background"],
                delta_pp=delta,
                corrected_pixels=int(corrected.sum()),
                damaged_pixels=int(damaged.sum()),
                per_class_delta_pp={
                    k: scalar_delta(x["per_class"][k]["IoU"], v["IoU"])
                    for k, v in y["per_class"].items()
                },
            )
            count = counts[y["role"]]
            count["corrected_pixels"] += row["corrected_pixels"]
            count["damaged_pixels"] += row["damaged_pixels"]
            count[
                "improved_views"
                if delta > 0
                else "worse_views"
                if delta < 0
                else "tied_views"
            ] += 1
        else:
            change[a != b] = [100, 140, 220]
        Image.fromarray(change).save(output / "change" / name)
        rows.append(row)
    comparisons = {}
    for group, b in candidate["metrics"].items():
        a = baseline["metrics"][group]
        comparisons[group] = dict(
            baseline_five_class_miou=a["mIoU_with_background"],
            candidate_five_class_miou=b["mIoU_with_background"],
            five_class_delta_pp=scalar_delta(
                a["mIoU_with_background"], b["mIoU_with_background"]
            ),
            baseline_foreground_miou=a["mIoU_without_background"],
            candidate_foreground_miou=b["mIoU_without_background"],
            foreground_delta_pp=scalar_delta(
                a["mIoU_without_background"], b["mIoU_without_background"]
            ),
            per_class_delta_pp={
                k: scalar_delta(a["per_class"][k]["IoU"], v["IoU"])
                for k, v in b["per_class"].items()
            },
            cable_fp_delta=b["per_class"]["stay_cable"]["fp"]
            - a["per_class"]["stay_cable"]["fp"],
            cable_fn_delta=b["per_class"]["stay_cable"]["fn"]
            - a["per_class"]["stay_cable"]["fn"],
            baseline_stay_cable=a["per_class"]["stay_cable"],
            candidate_stay_cable=b["per_class"]["stay_cable"],
            baseline_boundary=baseline["boundary_metrics"][group],
            candidate_boundary=candidate["boundary_metrics"][group],
            boundary_f1_delta_pp=scalar_delta(
                baseline["boundary_metrics"][group]["boundary_f1"],
                candidate["boundary_metrics"][group]["boundary_f1"],
            ),
        )
    transition_result = {
        key: {
            "class_order": list(CLASS_NAMES),
            "corrected_by_target_class": dict(
                zip(CLASS_NAMES, value["corrected_by_target_class"].tolist())
            ),
            "damaged_by_target_class": dict(
                zip(CLASS_NAMES, value["damaged_by_target_class"].tolist())
            ),
            "wrong_to_wrong_prediction_matrix": value[
                "wrong_to_wrong_prediction_matrix"
            ].tolist(),
        }
        for key, value in error_transitions.items()
    }
    bootstrap = paired_bootstrap(bootstrap_records, config["seed"])
    contact_sheet_hashes = contact_sheets(output, rows)
    baseline_training = read(parent / "mask2former/result.json")
    execution = {
        "baseline": {
            "training_seconds": baseline_training["seconds"],
            "training_peak_allocated_mib": baseline_training["peak_allocated_mib"],
            "training_peak_reserved_mib": baseline_training["peak_reserved_mib"],
            "inference_seconds": baseline["seconds"],
            "inference_median_seconds": baseline["inference_median_seconds"],
            "inference_peak_allocated_mib": baseline["peak_allocated_mib"],
            "inference_peak_reserved_mib": baseline["peak_reserved_mib"],
            "model_3000_bytes": (parent / "mask2former/model-3000.pt").stat().st_size,
            "model_6000_bytes": (parent / "mask2former/model-6000.pt").stat().st_size,
        },
        "candidate": {
            "training_seconds": training["seconds"],
            "training_peak_allocated_mib": training["peak_allocated_mib"],
            "training_peak_reserved_mib": training["peak_reserved_mib"],
            "inference_seconds": candidate["seconds"],
            "inference_median_seconds": candidate["inference_median_seconds"],
            "inference_peak_allocated_mib": candidate["peak_allocated_mib"],
            "inference_peak_reserved_mib": candidate["peak_reserved_mib"],
            "model_3000_bytes": (batch / "mask2former/model-3000.pt").stat().st_size,
            "model_6000_bytes": (batch / "mask2former/model-6000.pt").stat().st_size,
        },
    }
    result = dict(
        experiment_id=config["experiment_id"],
        status="completed",
        git_commit=training["git_commit"],
        seed=config["seed"],
        config_sha256=sha256(config_path),
        split_sha256=candidate["split_sha256"],
        baseline_result_sha256=sha256(parent / "inference/result.json"),
        candidate_result_sha256=sha256(batch / "inference/result.json"),
        independent_confusion_checks=audit_count,
        counts=counts,
        comparisons=comparisons,
        error_transitions=transition_result,
        paired_holdout_bootstrap=bootstrap,
        execution=execution,
        contact_sheet_hashes=contact_sheet_hashes,
        per_view=rows,
        interpretation="Single fixed local-composition split/seed, not blind. Unlabeled disagreements are not accuracy.",
    )
    write_new_json(output / "result.json", result)
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>__CANDIDATE__ / __BASELINE__ · Dev30</title>
<style>body{margin:24px;background:#f5f4ed;color:#173b38;font:16px system-ui}select{font:inherit;padding:8px}header{position:sticky;top:0;background:#f5f4ed;padding:10px 0}h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;display:block}figcaption{padding:6px}p{line-height:1.6}#summary{font-weight:600}#detail{font-variant-numeric:tabular-nums}@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}}</style>
<header><h1>完整画面上下文对照：__BASELINE__ → __CANDIDATE__</h1><select id="view"></select></header>
<p>固定 Dev30：270 张标签训练，30 张留出；单次划分、单 seed，非盲测。RGB150、模型与推理不变；两臂运行时策略由各自配置和结果记录约束。</p>
<p id="summary"></p><p id="detail"></p><div class="grid" id="images"></div>
<p>变化图：绿色＝修对，红色＝损伤，黄色＝仍然错误；无标签图仅以蓝色标记方案分歧，不能据此判定准确率。放大请点击图片查看原分辨率。</p>
<script>const DATA=__DATA__;const f=x=>x==null?'—':x.toFixed(3);const h=DATA.comparisons.holdout;
document.querySelector('#summary').textContent=`留出五类 mIoU：${f(100*h.baseline_five_class_miou)}% → ${f(100*h.candidate_five_class_miou)}%（${f(h.five_class_delta_pp)} pp）；前景：${f(100*h.baseline_foreground_miou)}% → ${f(100*h.candidate_foreground_miou)}%。`;
const rows=[...DATA.per_view].sort((a,b)=>(a.role==='holdout'?0:a.role==='unlabeled'?2:1)-(b.role==='holdout'?0:b.role==='unlabeled'?2:1)||a.name.localeCompare(b.name));
const select=document.querySelector('#view');for(const r of rows){const o=document.createElement('option');o.value=r.name;o.textContent=`${r.name} · ${r.role}${r.group?' · '+r.group:''}`;select.append(o)}
function draw(){const r=rows.find(x=>x.name===select.value);document.querySelector('#detail').textContent=r.role==='unlabeled'?'无真值：仅主观比较。':`本图五类 mIoU ${f(100*r.baseline_miou)}% → ${f(100*r.candidate_miou)}%（${f(r.delta_pp)} pp）；修对 ${r.corrected_pixels} / 损伤 ${r.damaged_pixels} 像素。`;
const tiles=[['RGB150','rgb'],...(r.role==='unlabeled'?[]:[['真值','gt']]),['__BASELINE__','baseline'],['__CANDIDATE__','candidate'],['变化','change']];document.querySelector('#images').innerHTML=tiles.map(([title,path])=>`<figure><figcaption>${title}</figcaption><a href="${path}/${r.name}" target="_blank"><img src="${path}/${r.name}" alt="${title}"></a></figure>`).join('')};select.onchange=draw;draw();</script></html>"""
    html = (
        html.replace("__DATA__", json.dumps(result, ensure_ascii=False))
        .replace("__BASELINE__", baseline["experiment_id"])
        .replace("__CANDIDATE__", candidate["experiment_id"])
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "comparisons": comparisons, "counts": counts},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    main(parser.parse_args().config.resolve())
