#!/usr/bin/env python3
"""Build the fixed dev30 RGB/semantic review and crop-coverage diagnostic."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, write_new_json
from src.semantic_render_domain import SEMANTIC_PALETTE


SEM384 = ROOT / "outputs/quality/SEM-384/r1"
RGB150 = ROOT / "outputs/quality/RGB-150/full-r1/evaluation"
REFERENCE = ROOT / "outputs/quality/RGB-150/scene/images"
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
SEM378_COMPARISON = ROOT / "outputs/quality/SEM-378/r1/comparison"
SEM379_AUDIT = ROOT / "outputs/quality/SEM-379/r1/sampling-audit.json"
SPLIT = ROOT / "experiments/splits/dev30_local_composition_v1.json"
OUTPUT = SEM384 / "review-r2"
SPECIAL = ("311.png", "350.png", "356.png")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def colorize(labels: np.ndarray) -> np.ndarray:
    if labels.dtype != np.uint8 or labels.ndim != 2 or labels.max() > 4:
        raise ValueError("Expected a native uint8 five-class label image")
    return SEMANTIC_PALETTE[labels]


def save_label(source: Path, target: Path) -> np.ndarray:
    labels = np.asarray(Image.open(source), dtype=np.uint8)
    Image.fromarray(colorize(labels)).save(target)
    return labels


def crop_coverage(samples: list[dict[str, Any]], crop: int = 768) -> np.ndarray:
    difference = np.zeros((990, 1321), dtype=np.int32)
    for sample in samples:
        top, left = int(sample["top"]), int(sample["left"])
        difference[top, left] += 1
        difference[top + crop, left] -= 1
        difference[top, left + crop] -= 1
        difference[top + crop, left + crop] += 1
    return difference.cumsum(0).cumsum(1)[:-1, :-1]


def review_html(data: dict[str, Any]) -> str:
    embedded = json.dumps(data, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dev30 Full-Chain Review</title>
<style>
:root {{ color-scheme: dark; --bg:#111315; --panel:#1a1e21; --line:#343a40; --text:#edf0f2; --muted:#aab2b9; --accent:#e2b74f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
header {{ position:sticky; top:0; z-index:3; display:flex; gap:18px; align-items:center; padding:14px 20px; background:#111315f2; border-bottom:1px solid var(--line); }}
h1 {{ margin:0; font-size:20px; }} h2 {{ margin:28px 0 10px; font-size:18px; }}
select {{ min-width:180px; background:var(--panel); color:var(--text); border:1px solid var(--line); padding:7px 10px; }}
main {{ max-width:1560px; margin:auto; padding:18px; }}
.summary {{ color:var(--muted); }} .summary strong {{ color:var(--text); }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.tile {{ margin:0; background:var(--panel); border:1px solid var(--line); }}
.tile figcaption {{ padding:8px 10px; color:var(--muted); border-bottom:1px solid var(--line); }}
.tile img {{ display:block; width:100%; aspect-ratio:1320/989; object-fit:contain; image-rendering:auto; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:14px; padding:10px 0 16px; color:var(--muted); }}
.metrics b {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
.diagnostic {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
.diagnostic article {{ border-top:2px solid var(--line); padding-top:10px; }}
.diagnostic img {{ width:100%; display:block; margin:7px 0; }}
.diagnostic p {{ color:var(--muted); }}
.legend {{ color:var(--muted); font-size:12px; }}
@media (max-width:900px) {{ .grid,.diagnostic {{ grid-template-columns:1fr; }} header {{ align-items:flex-start; flex-direction:column; gap:8px; }} }}
</style>
</head>
<body>
<header><h1>Dev30 Full-Chain Review</h1><label>Holdout view <select id="view"></select></label><span id="group"></span></header>
<main>
<p class="summary">Frozen local-composition development holdout. Metrics are reported after RGB-150 and the 6000-step SEM-384 checkpoint were sealed. This is not a blind or strictly spatial test.</p>
<div class="metrics" id="metrics"></div>
<section class="grid" id="tiles"></section>
<p class="legend">Semantic colors: background charcoal, deck blue, stay_cable yellow, tower red, foundation purple. Error map: cable FN yellow, cable FP purple, other class error red, correct pixels charcoal.</p>
<h2>Unlabeled Coverage Probes</h2>
<p class="summary">311/350/356 are RGB-training but unlabeled semantic views. They have no GT accuracy claim and received zero supervised crop draws. The images are reused from the frozen SEM378 comparison inventory.</p>
<section class="diagnostic" id="diagnostic"></section>
</main>
<script>
const DATA={embedded};
const select=document.querySelector('#view');
for (const row of DATA.holdout) {{ const option=document.createElement('option'); option.value=row.name; option.textContent=row.name; select.appendChild(option); }}
function fixed(value, digits=4) {{ return Number(value).toFixed(digits); }}
function show() {{
 const row=DATA.holdout.find(item=>item.name===select.value);
 document.querySelector('#group').textContent=row.group;
 document.querySelector('#metrics').innerHTML=`RGB PSNR <b>${{fixed(row.rgb.PSNR_dB,2)}} dB</b> SSIM <b>${{fixed(row.rgb.SSIM)}}</b> LPIPS <b>${{fixed(row.rgb.LPIPS_Alex)}}</b> Semantic foreground mIoU <b>${{fixed(row.semantic.mIoU_without_background)}}</b> stay_cable IoU <b>${{fixed(row.semantic.per_class.stay_cable.IoU)}}</b>`;
 const labels=['Reference PINHOLE RGB','RGB-150 render','Ground truth','SEM-384 prediction','Semantic error'];
 const keys=['reference','rgb150','ground_truth','prediction','error'];
 document.querySelector('#tiles').innerHTML=keys.map((key,i)=>`<figure class="tile"><figcaption>${{labels[i]}}</figcaption><img src="${{row[key]}}" alt="${{labels[i]}} ${{row.name}}"></figure>`).join('');
}}
select.addEventListener('change',show); show();
document.querySelector('#diagnostic').innerHTML=DATA.special.map(row=>`<article><h3>${{row.name}}</h3><p>Reference</p><img src="${{row.reference}}" alt="reference ${{row.name}}"><p>RGB-150</p><img src="${{row.rgb150}}" alt="RGB150 ${{row.name}}"><p>SEM-378 full-data reference</p><img src="${{row.sem378}}" alt="SEM378 ${{row.name}}"><p>SEM-384 dev270</p><img src="${{row.sem384}}" alt="SEM384 ${{row.name}}"><p>${{row.observation}}</p><p>RGB: ${{fixed(row.rgb.PSNR_dB,2)}} dB / SSIM ${{fixed(row.rgb.SSIM)}} / LPIPS ${{fixed(row.rgb.LPIPS_Alex)}}. SEM378-to-SEM384 disagreement: ${{fixed(100*row.sem378_sem384_disagreement,3)}}%.</p></article>`).join('');
</script>
</body></html>
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Review output already exists: {OUTPUT}")
    split = read(SPLIT)
    semantic = read(SEM384 / "inference/result.json")
    rgb = read(RGB150 / "result.json")
    plan = read(SEM384 / "sampling-plan.json")
    sem379 = read(SEM379_AUDIT)
    if semantic["status"] != "completed" or rgb["status"] != "completed":
        raise ValueError("Completed RGB-150 and SEM-384 evaluations are required")
    OUTPUT.mkdir(parents=True)
    for directory in (
        "reference",
        "rgb150",
        "ground-truth",
        "prediction",
        "error",
        "special",
    ):
        (OUTPUT / directory).mkdir()

    rgb_rows: dict[str, dict[str, float]] = {}
    for partition in ("train", "holdout"):
        with (RGB150 / f"{partition}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rgb_rows[row["view"]] = {
                    key: float(row[key]) for key in ("PSNR_dB", "SSIM", "LPIPS_Alex")
                }
    semantic_rows = {row["name"]: row for row in semantic["per_view"]}
    group_by_name = {
        name: group["group_id"] for group in split["groups"] for name in group["images"]
    }
    holdout = []
    for name in sorted(split["primary_fold"]["semantic_validation_labeled"]):
        shutil.copyfile(REFERENCE / name, OUTPUT / "reference" / name)
        shutil.copyfile(RGB150 / "images" / name, OUTPUT / "rgb150" / name)
        target = save_label(LABELS / name, OUTPUT / "ground-truth" / name)
        prediction = save_label(
            SEM384 / "inference/predictions" / name, OUTPUT / "prediction" / name
        )
        error = np.full((*target.shape, 3), [30, 30, 30], dtype=np.uint8)
        wrong = target != prediction
        error[wrong] = [220, 80, 80]
        error[(target == 2) & (prediction != 2)] = [245, 210, 65]
        error[(target != 2) & (prediction == 2)] = [170, 100, 190]
        Image.fromarray(error).save(OUTPUT / "error" / name)
        holdout.append(
            {
                "name": name,
                "group": group_by_name[name],
                "rgb": rgb_rows[name],
                "semantic": semantic_rows[name],
                "reference": f"reference/{name}",
                "rgb150": f"rgb150/{name}",
                "ground_truth": f"ground-truth/{name}",
                "prediction": f"prediction/{name}",
                "error": f"error/{name}",
            }
        )

    samples_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in plan["samples"]:
        samples_by_name[sample["name"]].append(sample)
    class_pixels = np.zeros(5, dtype=np.int64)
    class_exposures = np.zeros(5, dtype=np.int64)
    class_unseen = np.zeros(5, dtype=np.int64)
    unseen_fractions = []
    mean_exposures = []
    for name in plan["train_names"]:
        exposure = crop_coverage(samples_by_name[name])
        labels = np.asarray(Image.open(LABELS / name), dtype=np.uint8)
        unseen_fractions.append(float(np.mean(exposure == 0)))
        mean_exposures.append(float(np.mean(exposure)))
        for class_id in range(5):
            mask = labels == class_id
            class_pixels[class_id] += int(mask.sum())
            class_exposures[class_id] += int(exposure[mask].sum())
            class_unseen[class_id] += int(np.sum(mask & (exposure == 0)))

    observations = {
        "311.png": "A complete dominant tower and deck are visible, but RGB-150 smooths vehicles and fine hanger contrast. The geometry remains globally available; correctness of the unlabeled semantic mask is not measurable.",
        "350.png": "The bridge is distant, the deck is a thin full-width band, and only the right tower is complete. A 768 crop cannot include the whole 1320-pixel span, so local evidence can be detached from tower context.",
        "356.png": "The distant deck and cables cross the frame without a complete dominant tower. RGB-150 retains coarse alignment but weakens fine vertical detail; this is the strongest global-context probe of the three.",
    }
    special = []
    for name in SPECIAL:
        shutil.copyfile(
            ROOT / f"outputs/quality/all_data/RGB-120-r2/scene/images/{name}",
            OUTPUT / "special" / f"reference-{name}",
        )
        shutil.copyfile(RGB150 / "images" / name, OUTPUT / "special" / f"rgb150-{name}")
        current = save_label(
            SEM384 / "inference/predictions" / name,
            OUTPUT / "special" / f"sem384-{name}",
        )
        previous = np.asarray(
            Image.open(
                ROOT / "outputs/quality/SEM-378/r1/inference/predictions" / name
            ),
            dtype=np.uint8,
        )
        shutil.copyfile(
            SEM378_COMPARISON / "sem378" / name,
            OUTPUT / "special" / f"sem378-{name}",
        )
        special.append(
            {
                "name": name,
                "semantic_training_draws": 0,
                "has_semantic_ground_truth": False,
                "rgb": rgb_rows[name],
                "sem384_class_fraction": [
                    float(np.mean(current == i)) for i in range(5)
                ],
                "sem378_sem384_disagreement": float(np.mean(current != previous)),
                "sem378_asset_sha256": sha256(SEM378_COMPARISON / "sem378" / name),
                "sem378_raw_prediction_sha256": sha256(
                    ROOT / "outputs/quality/SEM-378/r1/inference/predictions" / name
                ),
                "observation": observations[name],
                "reference": f"special/reference-{name}",
                "rgb150": f"special/rgb150-{name}",
                "sem384": f"special/sem384-{name}",
                "sem378": f"special/sem378-{name}",
            }
        )

    class_names = ("background", "deck", "stay_cable", "tower", "foundation")
    coverage = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": "SEM-384-RGB150-DEV270",
        "scope": "Post-hoc coverage diagnostic; no model or split selection",
        "sampling_plan_sha256": sha256(SEM384 / "sampling-plan.json"),
        "crop": {
            "image_width": 1320,
            "image_height": 989,
            "crop_width": 768,
            "crop_height": 768,
            "single_crop_image_area_fraction": 768 * 768 / (1320 * 989),
            "single_crop_width_fraction": 768 / 1320,
            "single_crop_height_fraction": 768 / 989,
            "full_width_context_possible": False,
            "full_height_context_possible": False,
        },
        "actual_plan": {
            "steps": len(plan["samples"]),
            "training_views": len(plan["train_names"]),
            "minimum_draws_per_view": plan["minimum_samples_per_view"],
            "maximum_draws_per_view": plan["maximum_samples_per_view"],
            "view_unseen_pixel_fraction_quantiles": np.quantile(
                unseen_fractions, [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            "view_mean_pixel_exposure_quantiles": np.quantile(
                mean_exposures, [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            "class_coverage": {
                name: {
                    "pixels": int(class_pixels[index]),
                    "total_exposures": int(class_exposures[index]),
                    "mean_exposure": float(
                        class_exposures[index] / class_pixels[index]
                    ),
                    "never_seen_pixels": int(class_unseen[index]),
                    "never_seen_fraction": float(
                        class_unseen[index] / class_pixels[index]
                    ),
                }
                for index, name in enumerate(class_names)
            },
        },
        "sem379_reference": {
            "sampling_audit_sha256": sha256(SEM379_AUDIT),
            "guided_draws": sem379["guided_draws"],
            "uniform_draws": sem379["uniform_draws"],
            "zero_cable_crops_baseline": sem379["zero_cable_crops"]["baseline"],
            "zero_cable_crops_candidate": sem379["zero_cable_crops"]["candidate"],
            "stay_cable_exposure_change_fraction": sem379["class_pixel_exposure"][
                "candidate"
            ][2]
            / sem379["class_pixel_exposure"]["baseline"][2]
            - 1,
            "foundation_exposure_change_fraction": sem379["class_pixel_exposure"][
                "candidate"
            ][4]
            / sem379["class_pixel_exposure"]["baseline"][4]
            - 1,
            "verified_interpretation": "SEM379 increased stay_cable pixel exposure but did not improve recall; crop exposure alone is not a demonstrated causal fix.",
        },
        "unlabeled_probes": special,
        "findings": [
            "Every native 768 crop omits at least 552 horizontal and 221 vertical pixels, so it cannot contain the full released frame.",
            "The three probes are absent from semantic supervision; they test qualitative transfer only.",
            "350 and 356 combine a thin distant deck with incomplete tower context, while RGB-150 also attenuates fine hanger detail.",
            "SEM379 showed that adding local stay_cable exposure can trade precision and recall rather than resolve missing global context.",
        ],
        "next_experiments_maximum_two": [
            "One matched semantic context arm: keep the frozen view/flip sequence and 6000 updates, replacing a fixed 50 percent of native crops with aspect-preserving full-frame 768 inputs.",
            "Only after the context arm, one matched RGB detail arm: change a single pre-registered high-frequency preservation term and retrain the same semantic recipe to separate RGB blur from semantic context limits.",
        ],
    }
    write_new_json(OUTPUT / "coverage-diagnostic.json", coverage)
    page_data = {
        "experiment_id": "SEM-384-RGB150-DEV270",
        "holdout": holdout,
        "special": special,
        "summary": {
            "rgb": rgb["metrics"]["holdout"],
            "semantic": semantic["metrics"]["holdout"],
            "boundary": semantic["boundary_metrics"]["holdout"],
        },
    }
    (OUTPUT / "index.html").write_text(review_html(page_data), encoding="utf-8")
    write_new_json(OUTPUT / "data.json", page_data)
    write_new_json(
        OUTPUT / "result.json",
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": "SEM-384-RGB150-DEV270",
            "holdout_views": len(holdout),
            "unlabeled_probes": list(SPECIAL),
            "source_hashes": {
                "rgb_evaluation": sha256(RGB150 / "result.json"),
                "semantic_inference": sha256(SEM384 / "inference/result.json"),
                "sampling_plan": sha256(SEM384 / "sampling-plan.json"),
                "sem379_sampling_audit": sha256(SEM379_AUDIT),
                "split": sha256(SPLIT),
            },
            "artifacts": {
                "index.html": sha256(OUTPUT / "index.html"),
                "data.json": sha256(OUTPUT / "data.json"),
                "coverage-diagnostic.json": sha256(OUTPUT / "coverage-diagnostic.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
