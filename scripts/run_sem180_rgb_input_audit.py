#!/usr/bin/env python3
"""Conditional final-only deck input intervention; no new semantic training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_sem161_deck_binary_final import class_map, metrics_for, protection_exact, one_hot_logits
from src.semantic2d import sha256
from src.semantic_deck_binary import load_binary_model, sliding_deck_probability, read_rgb, recover_deck, deck_gate


def verified_predictions(root, expected):
    path = root / "prediction_manifest.json"
    if sha256(path) != expected:
        raise ValueError("Frozen parent prediction manifest changed")
    result = {}
    for row in json.loads(path.read_text())["predictions"]:
        target = root / row["prediction"]
        if sha256(target) != row["prediction_sha256"]:
            raise ValueError("Frozen parent prediction changed")
        result[row["name"]] = target
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rgb-audit", type=Path, required=True)
    p.add_argument("--new-external-rgb", type=Path)
    p.add_argument("--external-record", type=Path)
    args = p.parse_args()
    root = args.output_root
    cfg_path = REPO / "experiments/runs/SEM-180-RGB-INPUT-AUDIT/config.yaml"
    cfg = json.loads(cfg_path.read_text())
    audit = json.loads(args.rgb_audit.read_text())
    output = root / "r2/sem_180/input_audit/full-20260906-r1"
    output.mkdir(parents=True, exist_ok=False)
    if not (audit["numeric_gate_passes"] and audit["visual_gate_passes"] and audit.get("SEM180_allowed", False)):
        (output / "result.json").write_text(json.dumps({"experiment_id": cfg["experiment_id"],
            "status": "not_run_parent_RGB_gate_failed", "new_semantic_predictions": 0,
            "semantic_labels_opened": 0, "rgb_audit_sha256": sha256(args.rgb_audit),
            "fallback": "unchanged RGB-030-FULL + SEM-161"}, indent=2))
        print("SEM-180 not run: parent RGB gates failed")
        return
    if not args.new_external_rgb or not args.external_record:
        raise ValueError("A verified official-camera RGB-092 external render is required")
    external_record = json.loads(args.external_record.read_text())
    if not (external_record["selected_pose_name"] == "140.png"
            and external_record["input_pose_convention"] == "CAMERA_FROM_WORLD"
            and external_record["output_grid"] == "official_SIMPLE_RADIAL"
            and external_record["uses_reference_rgb_or_labels"] is False
            and external_record["iteration"] == 30000
            and Path(external_record["model_path"]).resolve() == (root / "r1/runs/RGB-092-TRANSIENT").resolve()
            and Path(external_record["output_path"]).resolve() == args.new_external_rgb.resolve()
            and external_record["scaffold_external_appearance_fallback"]["embedding_index"] == 0):
        raise ValueError("External RGB evidence is not the registered same-camera intervention")
    expected = json.loads((cfg_path.parent / "frozen_fallback.json").read_text())
    for key in ["result", "prediction_manifest", "checkpoint"]:
        if sha256(root / expected[key]["path"]) != expected[key]["sha256"]:
            raise ValueError("Frozen SEM-161 artifact changed: " + key)
    baseline_root = root / "r2/sem_161/developer_audit/final-20260904-r1"
    pre_root = root / "r2/sem_141/developer_audit/final-20260904-r1"
    old_result = json.loads((baseline_root / "result.json").read_text())
    pre_result = json.loads((pre_root / "result.json").read_text())
    baseline = verified_predictions(baseline_root, old_result["prediction_manifest_sha256"])
    pre = verified_predictions(pre_root, pre_result["prediction_manifest_sha256"])
    names = json.loads((REPO / cfg["data"]["semantic_split"]).read_text())["semantic_validation_labeled"]
    if set(names) != set(baseline) or set(names) != set(pre):
        raise ValueError("Semantic view set mismatch")
    checkpoint_path = root / "r2/sem_161/final_training/final-20260904-r1/checkpoints/deck_binary_final.pth"
    if sha256(checkpoint_path) != old_result["checkpoint_sha256"]:
        raise ValueError("Frozen SEM-161 checkpoint changed")
    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("SEM-180 must use the frozen semantic2d transformers==4.57.1 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = load_binary_model(root / "r2/sem_005/third_party/nvidia_mit_b2_3bb39e8", 20260805)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
    model = model.to(device).eval()
    images = root / "r1/runs/RGB-092-TRANSIENT/r1_validation_renders"
    predictions = {}
    records = []
    old_rgb_root = root / "r2/sem_102/validation_render/final-20260902-r1/images"
    (output / "predictions").mkdir()
    for name in names:
        path = images / name
        rgb = read_rgb(path)
        probability = sliding_deck_probability(model, rgb, crop=960, stride=480, device=device, amp_dtype=amp)
        base = class_map(baseline[name], context="SEM-161")
        previous = class_map(pre[name], context="SEM-141")
        old_probability = sliding_deck_probability(model, read_rgb(old_rgb_root / name), crop=960, stride=480, device=device, amp_dtype=amp)
        counterfactual, _, _ = recover_deck(base, previous, old_probability, threshold=0.99)
        if not np.array_equal(counterfactual, base):
            raise RuntimeError("Old-input replay changed frozen SEM-161; RGB attribution is not valid")
        candidate, eligible, additions = recover_deck(base, previous, probability, threshold=0.99)
        if not protection_exact(base, previous, candidate, eligible):
            raise RuntimeError("Semantic protection violated")
        target = output / "predictions" / name
        Image.fromarray(candidate).save(target)
        predictions[name] = target
        records.append({"name": name, "rgb_sha256": sha256(path), "prediction_sha256": sha256(target),
                        "added_deck_pixels": int(additions.sum())})
    # All predictions are fixed and saved before any developer labels are opened.
    (output / "prediction_manifest.json").write_text(json.dumps({"predictions": records,
          "developer_labels_opened_before_manifest": False, "threshold": 0.99}, indent=2))
    labels = root / "r2/sem_001/labels"
    before, _ = metrics_for(names, labels, baseline)
    after, _ = metrics_for(names, labels, predictions)
    gate = deck_gate(candidate=after, baseline=before, protected_predictions_exact=True,
              minimum_iou_gain=0.002, minimum_recall_gain=0.005, minimum_precision=0.94,
              minimum_miou_gain=0, maximum_background_fp_ratio=1.1)
    oblique = after["oblique_158_161"]["mean_deck_iou"] >= before["oblique_158_161"]["mean_deck_iou"]
    # External input is already in the official SIMPLE_RADIAL pixel grid.
    external = output / "external_camera"
    external.mkdir()
    probability = sliding_deck_probability(model, read_rgb(args.new_external_rgb), crop=960, stride=480, device=device, amp_dtype=amp)
    base = class_map(baseline_root / "external_camera/semantic.png", context="old external")
    previous = class_map(pre_root / "external_camera/semantic.png", context="pre external")
    candidate, eligible, additions = recover_deck(base, previous, probability, threshold=.99)
    if not protection_exact(base, previous, candidate, eligible):
        raise RuntimeError("External semantic protection violated")
    Image.fromarray(candidate).save(external / "semantic.png")
    np.save(external / "semantic_logits.npy", one_hot_logits(candidate))
    reloaded_png = np.array(Image.open(external / "semantic.png"))
    reloaded_logits = np.load(external / "semantic_logits.npy", allow_pickle=False)
    if not (reloaded_png.dtype == np.uint8 and reloaded_png.shape == (989, 1320)
            and np.array_equal(reloaded_png, candidate)
            and reloaded_logits.dtype == np.float32 and reloaded_logits.shape == (5, 989, 1320)
            and np.isfinite(reloaded_logits).all()
            and np.array_equal(reloaded_logits.argmax(0), candidate)):
        raise RuntimeError("Saved official-camera output failed reload/format contract")
    result = {"experiment_id": cfg["experiment_id"], "status": "completed_developer_audit",
              "baseline_metrics": before, "metrics": after, "gate": gate, "oblique_passes": oblique,
              "protected_exact": True, "external_camera_passes": True, "old_input_replay_exact": True,
              "external_saved_outputs_reloaded": True,
              "external_rgb_sha256": sha256(args.new_external_rgb),
              "external_record_sha256": sha256(args.external_record),
              "external_semantic_sha256": sha256(external / "semantic.png"),
              "external_logits_sha256": sha256(external / "semantic_logits.npy"),
              "checkpoint_sha256": sha256(checkpoint_path), "rgb_audit_sha256": sha256(args.rgb_audit),
              "config_sha256": sha256(cfg_path), "claim_boundary": "Developer-only input intervention, no retraining, old 3D semantic path retained",
              "passes": gate["passed"] and oblique}
    if not all(sha256(images / r["name"]) == r["rgb_sha256"] for r in records):
        raise RuntimeError("Semantic audit changed frozen RGB outputs")
    result["rgb_unchanged_during_semantic_audit"] = True
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(gate))


if __name__ == "__main__":
    main()
