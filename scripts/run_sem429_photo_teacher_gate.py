#!/usr/bin/env python3
"""Run SEM429 frozen photo-teacher validation and public-unlabeled export."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.photo_teacher_gate import PhotoTeacherGate, apply_photo_teacher_gate
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_semantic_audit import independent_confusion, metrics


EID = "SEM-429-PHOTO-TEACHER-GATE-DIAG"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"
CODE = (
    "scripts/run_sem429_photo_teacher_gate.py",
    "src/photo_teacher_gate.py",
    "src/quality_mask2former.py",
    "src/quality_semantic_audit.py",
)
PALETTE = np.asarray(
    [[30, 30, 30], [80, 160, 245], [245, 210, 65], [220, 80, 80], [170, 100, 190]],
    dtype=np.uint8,
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_runtime(artifact_root: Path):
    runtime = ROOT / read(CONFIG)["runtime_root"]
    sys.path[:0] = [
        str(runtime / "runtime"),
        str(runtime / "vendor/detectron2"),
        str(runtime / "vendor/Mask2Former"),
        str(runtime / "vendor/Mask2Former/mask2former/modeling/pixel_decoder/ops"),
    ]
    cuda = artifact_root / "r1/toolchain/cuda-12.8"
    os.environ.update(
        CUDA_HOME=str(cuda),
        PATH=str(cuda / "bin") + os.pathsep + os.environ.get("PATH", ""),
        LD_LIBRARY_PATH=str(cuda / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
        TORCH_HOME=str(artifact_root / "r1/weights/torch"),
        WANDB_MODE="disabled",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    import torch

    return runtime, torch


def one_hot(prediction: np.ndarray) -> np.ndarray:
    if prediction.dtype != np.uint8 or prediction.ndim != 2 or prediction.max() > 4:
        raise ValueError("Saved prediction must use native class IDs 0..4")
    return np.eye(5, dtype=np.float32)[prediction].transpose(2, 0, 1)


def load_rgb(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if value.shape != (989, 1320, 3):
        raise ValueError(f"Native registered RGB required: {path}")
    return value


def save_id_and_visual(output: Path, name: str, value: np.ndarray, visual: bool = True) -> None:
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8)).save(output / name)
    if visual:
        visual_output = output.parent / (output.name + "-visual")
        visual_output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(PALETTE[value]).save(visual_output / name)


def infer_partition(model, names: list[str], spec: dict[str, Any], output: Path, native_predict) -> list[dict[str, Any]]:
    rows = []
    render_root = ROOT / spec["render_root"]
    photo_root = ROOT / spec["photo_root"]
    saved_root = ROOT / spec["saved_predictions"]
    for index, name in enumerate(names):
        render_rgb = load_rgb(render_root / name)
        photo_rgb = load_rgb(photo_root / name)
        saved_prediction = np.asarray(Image.open(saved_root / name), dtype=np.uint8)
        if saved_prediction.shape != (989, 1320):
            raise ValueError(f"Saved prediction grid changed: {name}")
        photo_probability = native_predict(model, photo_rgb)
        flipped_probability = native_predict(model, np.ascontiguousarray(photo_rgb[:, ::-1]))[:, :, ::-1].copy()
        gated = apply_photo_teacher_gate(
            one_hot(saved_prediction), photo_probability, flipped_probability
        )
        if not np.array_equal(gated["render_prediction"], saved_prediction):
            raise ValueError(f"Saved render prediction was not preserved: {name}")
        save_id_and_visual(output / "render", name, saved_prediction)
        save_id_and_visual(output / "photo-teacher", name, gated["photo_prediction"])
        save_id_and_visual(output / "candidate", name, gated["candidate_prediction"])
        save_id_and_visual(output / "accepted-mask", name, gated["accepted_mask"].astype(np.uint8), visual=False)
        rows.append(
            {
                "name": name,
                "counts": gated["counts"],
                "photo_sha256": sha256(photo_root / name),
                "render_rgb_sha256": sha256(render_root / name),
                "saved_prediction_sha256": sha256(saved_root / name),
                "teacher_prediction_sha256": sha256(output / "photo-teacher" / name),
                "candidate_prediction_sha256": sha256(output / "candidate" / name),
                "accepted_mask_sha256": sha256(output / "accepted-mask" / name),
            }
        )
        if index % 10 == 0:
            print(json.dumps({"partition": output.name, "done": index + 1, "total": len(names)}), flush=True)
    return rows


def load_model(runtime: Path, torch, spec: dict[str, Any], config: dict[str, Any]):
    from src.quality_mask2former import config_for_model, load_official_model

    checkpoint = ROOT / spec["checkpoint"]
    record = read(ROOT / spec["checkpoint_result"])
    expected = record["artifacts"]["model-6000.pt"]
    if sha256(checkpoint) != expected:
        raise ValueError("Frozen checkpoint hash mismatch")
    model_config = config_for_model(
        runtime / "vendor/Mask2Former", config["upstream_config"], False
    )
    model, _ = load_official_model(
        model_config, runtime / "weights/model_final_6b4a3a.pkl"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    model.cuda().eval().requires_grad_(False)
    return model, {"path": spec["checkpoint"], "sha256": expected}


def evaluate_validation(config: dict[str, Any], validation_names: list[str], output: Path) -> dict[str, Any]:
    labels = ROOT / config["validation"]["label_root"]
    baseline_total = np.zeros((5, 5), dtype=np.int64)
    candidate_total = np.zeros((5, 5), dtype=np.int64)
    corrected = harmed = neutral = changed = 0
    per_view = []
    for name in validation_names:
        target = np.asarray(Image.open(labels / name), dtype=np.uint8)
        baseline = np.asarray(Image.open(output / "validation/render" / name), dtype=np.uint8)
        candidate = np.asarray(Image.open(output / "validation/candidate" / name), dtype=np.uint8)
        mask = baseline != candidate
        base_ok = baseline == target
        candidate_ok = candidate == target
        view_corrected = int(np.sum(mask & ~base_ok & candidate_ok))
        view_harmed = int(np.sum(mask & base_ok & ~candidate_ok))
        view_neutral = int(np.sum(mask & ~base_ok & ~candidate_ok))
        corrected += view_corrected
        harmed += view_harmed
        neutral += view_neutral
        changed += int(mask.sum())
        baseline_cm = independent_confusion(target, baseline)
        candidate_cm = independent_confusion(target, candidate)
        baseline_total += baseline_cm
        candidate_total += candidate_cm
        per_view.append(
            {
                "name": name,
                "changed": int(mask.sum()),
                "corrected": view_corrected,
                "harmed": view_harmed,
                "neutral_wrong_to_wrong": view_neutral,
                "baseline": metrics(baseline_cm),
                "candidate": metrics(candidate_cm),
            }
        )
    baseline_metrics = metrics(baseline_total)
    candidate_metrics = metrics(candidate_total)
    changed_precision = corrected / changed if changed else 0.0
    delta = 100 * (
        candidate_metrics["mIoU_with_background"]
        - baseline_metrics["mIoU_with_background"]
    )
    gates = config["promotion_gates"]
    protected_cable = all(
        np.array_equal(
            np.asarray(Image.open(output / "validation/render" / name)) == 2,
            np.asarray(Image.open(output / "validation/candidate" / name)) == 2,
        )
        for name in validation_names
    )
    protected_foundation = all(
        np.array_equal(
            np.asarray(Image.open(output / "validation/render" / name)) == 4,
            np.asarray(Image.open(output / "validation/candidate" / name)) == 4,
        )
        for name in validation_names
    )
    gate_results = {
        "minimum_changed_precision": changed_precision >= gates["minimum_changed_precision"],
        "minimum_five_class_miou_delta_pp": delta >= gates["minimum_five_class_miou_delta_pp"],
        "positive_net_corrections": corrected > harmed,
        "stay_cable_prediction_identity": protected_cable,
        "foundation_prediction_identity": protected_foundation,
    }
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "five_class_miou_delta_pp": delta,
        "foreground_miou_delta_pp": 100 * (
            candidate_metrics["mIoU_without_background"]
            - baseline_metrics["mIoU_without_background"]
        ),
        "changed_pixels": changed,
        "changed_fraction": changed / (len(validation_names) * 989 * 1320),
        "corrected_pixels": corrected,
        "harmed_pixels": harmed,
        "neutral_wrong_to_wrong": neutral,
        "net_corrections": corrected - harmed,
        "changed_precision": changed_precision,
        "gates": gate_results,
        "training_authorized": all(gate_results.values()),
        "per_view": per_view,
    }


def build_review(config: dict[str, Any], output: Path, validation_names: list[str], public_names: list[str]) -> None:
    review = output / "review"
    review.mkdir(exist_ok=False)
    payload = {"validation": validation_names, "public100": public_names}
    root = "../../../../../"
    body = f"""<!doctype html><meta charset='utf-8'><title>SEM429 photo teacher review</title>
<style>body{{font:14px system-ui;background:#11151b;color:#e8edf3;margin:0}}header{{position:sticky;top:0;background:#11151bee;padding:12px;z-index:2}}button,select{{background:#202733;color:#fff;border:1px solid #4d5b70;padding:7px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:10px}}figure{{margin:0;background:#1b222c;padding:6px}}img{{width:100%;display:block}}figcaption{{padding:5px}}.note{{color:#aebbd0}}</style>
<header><b>SEM429 照片教师门控审阅</b> <select id='part'><option value='public100'>301–400 无标签</option><option value='validation'>空间确认30图</option></select> <button id='prev'>上一张</button> <select id='name'></select> <button id='next'>下一张</button><div class='note'>候选只允许 render background → flip稳定的大块 deck/tower；stay_cable/foundation预测保持不变。</div></header>
<main class='grid'><figure><figcaption>真实照片（仅教师输入）</figcaption><img id='photo'></figure><figure><figcaption>重建RGB（测试可用输入）</figcaption><img id='renderRgb'></figure><figure><figcaption>接受区域（白）</figcaption><img id='mask'></figure><figure><figcaption>原render语义</figcaption><img id='base'></figure><figure><figcaption>photo teacher语义</figcaption><img id='teacher'></figure><figure><figcaption>门控候选语义</figcaption><img id='candidate'></figure></main>
<script>const data={json.dumps(payload)};let i=0;const part=document.querySelector('#part'),sel=document.querySelector('#name');function fill(){{sel.innerHTML=data[part.value].map(x=>`<option>${{x}}</option>`).join('');i=0;show()}}function show(){{const names=data[part.value];i=(i+names.length)%names.length;sel.selectedIndex=i;const n=names[i],p=part.value;document.querySelector('#photo').src='{root}'+(p==='public100'?'outputs/quality/all_data/RGB-120-r2/scene/images/':'outputs/quality/RGB-152/scene/images/')+n;document.querySelector('#renderRgb').src='{root}'+(p==='public100'?'outputs/quality/rgb140/full-r2/evaluation/images/':'outputs/quality/RGB-152/full-r2/evaluation/images/')+n;document.querySelector('#mask').src='../'+p+'/accepted-mask/'+n;document.querySelector('#base').src='../'+p+'/render-visual/'+n;document.querySelector('#teacher').src='../'+p+'/photo-teacher-visual/'+n;document.querySelector('#candidate').src='../'+p+'/candidate-visual/'+n}}part.onchange=fill;sel.onchange=()=>{{i=sel.selectedIndex;show()}};document.querySelector('#prev').onclick=()=>{{i--;show()}};document.querySelector('#next').onclick=()=>{{i++;show()}};document.onkeydown=e=>{{if(e.key==='ArrowLeft'){{i--;show()}}if(e.key==='ArrowRight'){{i++;show()}}}};fill();</script>"""
    (review / "index.html").write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    if config["experiment_id"] != EID or config["training_updates"] != 0:
        raise ValueError("SEM429 identity or zero-training contract changed")
    output = ROOT / config["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    split = read(ROOT / config["split_path"])
    validation_names = sorted(split["primary_fold"]["semantic_validation_labeled"])
    public = config["public_unlabeled"]
    public_names = [f"{index:03d}.png" for index in range(public["first_index"], public["last_index"] + 1)]
    if len(validation_names) != config["validation"]["views"] or len(public_names) != public["views"]:
        raise ValueError("Registered role count mismatch")
    result = {
        "schema_version": 1,
        "status": "running",
        "experiment_id": EID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": config["seed"],
        "dataset_manifest": config["dataset_manifest"],
        "split_id": config["split_id"],
        "config_sha256": sha256(CONFIG),
        "split_sha256": sha256(ROOT / config["split_path"]),
        "code_sha256": {name: sha256(ROOT / name) for name in CODE},
        "trained_steps": 0,
        "labels_decoded_during_inference": False,
        "validation_names": validation_names,
        "public_unlabeled_names": public_names,
        "models": [],
    }
    started = time.monotonic()
    try:
        runtime, torch = setup_runtime(args.artifact_root.resolve())
        from src.quality_mask2former import native_predict

        with (ROOT / "outputs/quality/gpu.lock").open("a") as lock:
            queued = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["queue_seconds"] = time.monotonic() - queued
            torch.set_num_threads(4)
            torch.cuda.reset_peak_memory_stats()
            model, record = load_model(runtime, torch, config["validation"], config)
            result["models"].append(record)
            validation_rows = infer_partition(
                model, validation_names, config["validation"], output / "validation", native_predict
            )
            del model
            torch.cuda.empty_cache()
            model, record = load_model(runtime, torch, public, config)
            result["models"].append(record)
            public_rows = infer_partition(
                model, public_names, public, output / "public100", native_predict
            )
            del model
            torch.cuda.empty_cache()
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        seal = {
            "schema_version": 1,
            "experiment_id": EID,
            "config_sha256": result["config_sha256"],
            "code_sha256": result["code_sha256"],
            "labels_decoded": False,
            "validation": validation_rows,
            "public100": public_rows,
        }
        write_new_json(output / "inference-seal.json", seal)
        result["inference_seal_sha256_before_label_access"] = sha256(output / "inference-seal.json")
        result["validation"] = evaluate_validation(config, validation_names, output)
        if sha256(output / "inference-seal.json") != result["inference_seal_sha256_before_label_access"]:
            raise ValueError("Inference seal changed during label evaluation")
        result["public100"] = {
            "accepted_pixels": int(sum(row["counts"]["total"] for row in public_rows)),
            "accepted_fraction": float(
                sum(row["counts"]["total"] for row in public_rows) / (len(public_rows) * 989 * 1320)
            ),
            "accepted_deck_pixels": int(sum(row["counts"]["1"] for row in public_rows)),
            "accepted_tower_pixels": int(sum(row["counts"]["3"] for row in public_rows)),
            "views_with_changes": int(sum(row["counts"]["total"] > 0 for row in public_rows)),
            "ground_truth_accessed": False,
            "rows": public_rows,
        }
        build_review(config, output, validation_names, public_names)
        result["review"] = str((output / "review/index.html").relative_to(ROOT))
        result["status"] = (
            "completed_training_authorized"
            if result["validation"]["training_authorized"]
            else "completed_gate_failed_no_training"
        )
    except BaseException as error:
        result.update(status="failed_preserved", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
