#!/usr/bin/env python3
"""Independent arithmetic/provenance verification of already-scored SEM325 outputs."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.semantic2d import read_json, sha256, write_json


def checked(path, digest):
    if sha256(path) != digest: raise ValueError("Hash differs: " + str(path))


def run(args):
    cfg = read_json(args.config)
    quality = read_json(args.score_dir / "result.json")
    checked(args.config, quality["config_sha256"])
    checked(args.prediction_dir / "result.json", quality["prediction_result_sha256"])
    checked(args.audit_dir / "result.json", quality["audit_result_sha256"])
    pred = read_json(args.prediction_dir / "result.json")
    checked(args.prediction_dir / "access_seal.json", pred["access_seal_sha256"])
    seal = read_json(args.prediction_dir / "access_seal.json")
    for name, digest in seal["source_sha256"].items():
        checked(ROOT / name, digest); checked(args.prediction_dir / "source_snapshot" / name, digest)
    for r in pred["predictions"]: checked(args.prediction_dir / r["path"], r["sha256"])
    sets = [(quality["targets"], quality["summary"])]
    sets += [([r for r in quality["targets"] if r["fold"] == f["fold"]], f) for f in quality["folds"]]
    for rows, saved in sets:
        values = {}
        for key in ("baseline", "candidate"):
            cm = np.stack([r[key] for r in rows]).sum(0)
            diagonal, actual, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
            iou = diagonal / (actual + predicted - diagonal)
            values[key] = {"miou": float(iou.mean()), "iou": float(iou[4]),
                           "precision": float(diagonal[4] / predicted[4]), "recall": float(diagonal[4] / actual[4])}
            if not np.array_equal(cm, saved[key]["confusion_matrix_rows_gt_columns_prediction"]):
                raise ValueError("Confusion aggregate differs")
            for k, v in values[key].items():
                old = saved[key]["mIoU_with_background"] if k == "miou" else saved[key]["per_class"]["foundation"][k]
                if not np.isclose(v, old, rtol=0, atol=1e-12): raise ValueError("Arithmetic differs")
        for k in values["baseline"]:
            saved_key = "delta_miou_pp" if k == "miou" else "delta_foundation_" + k + "_pp"
            if not np.isclose(100 * (values["candidate"][k] - values["baseline"][k]), saved[saved_key], rtol=0, atol=1e-10):
                raise ValueError("Delta arithmetic differs")
        if sum(r["changed"] for r in rows) != sum(saved[k] for k in ("corrected", "harmed", "still_wrong")):
            raise ValueError("Transition accounting differs")
    s, fs, g = quality["summary"], quality["folds"], cfg["quality_gates"]
    decisions = [s["delta_miou_pp"] >= g["minimum_miou_delta_pp"],
                 s["delta_foundation_iou_pp"] >= g["minimum_foundation_iou_delta_pp"],
                 s["delta_foundation_recall_pp"] >= -g["maximum_foundation_recall_loss_pp"],
                 s["candidate"]["per_class"]["foundation"]["recall"] >= g["minimum_foundation_recall"],
                 len([f for f in fs if f["delta_foundation_iou_pp"] > 0]) >= g["minimum_improving_foundation_folds"],
                 min(f["delta_foundation_iou_pp"] for f in fs) >= g["minimum_fold_foundation_iou_delta_pp"],
                 min(f["delta_miou_pp"] for f in fs) >= g["minimum_fold_miou_delta_pp"]]
    if all(decisions) != quality["passed"]: raise ValueError("Quality decision differs")
    return {"experiment_id": cfg["experiment_id"], "partition": quality["partition"],
            "score_sha256": sha256(args.score_dir / "result.json"), "config_sha256": sha256(args.config),
            "auditor_sha256": sha256(Path(__file__)), "status": "passed_independent_quality_arithmetic_and_seals",
            "source_files_verified": len(seal["source_sha256"]), "native_pngs_verified": len(pred["predictions"]),
            "quality_passed": quality["passed"], "new_label_decodes": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prediction-dir", "audit-dir", "score-dir", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "experiments/runs/SEM-325-DENSE-FOUNDATION-REVIEW-OOF/config.yaml")
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result = run(args); write_json(args.output, result); print(result)
