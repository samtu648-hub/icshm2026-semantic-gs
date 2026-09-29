#!/usr/bin/env python3
"""Read-only SEM325 calibration boundary diagnostic; one preregistered counterfactual."""
from __future__ import annotations
import argparse
from copy import copy
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from src.semantic2d import read_json, write_json, sha256, label_array, confusion_metrics
from src.semantic_dense_review import native_support
from src.semantic_source_boundary import foundation_distances, sample_boundary_distance, boundary_histogram, protect_boundary
from run_sem325_native_review import context, snapshot, verify_prediction_seal
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem320_dense_preflight import checked

EXP = "SEM-327-SOURCE-BOUNDARY-DIAG"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"


def aggregate(rows):
    result = {arm: confusion_metrics(sum((np.asarray(r[arm]) for r in rows), np.zeros((5, 5), np.int64)))
              for arm in ("baseline", "dense", "boundary_counterfactual")}
    for reference in ("baseline", "dense"):
        a, b = result[reference], result["boundary_counterfactual"]
        result["delta_vs_" + reference + "_pp"] = {
            "miou": 100 * (b["mIoU_with_background"] - a["mIoU_with_background"]),
            **{k: 100 * (b["per_class"]["foundation"][k] - a["per_class"]["foundation"][k])
               for k in ("iou", "precision", "recall")}}
    result["restored_by_gt_class"] = np.sum([r["restored_by_gt_class"] for r in rows], axis=0).tolist()
    result["distance_bins_by_gt_class"] = np.sum([r["distance_bins_by_gt_class"] for r in rows], axis=0).tolist()
    return result


def run(args):
    torch.set_num_threads(4)
    cfg = read_json(args.config)
    if cfg["experiment_id"] != EXP:
        raise ValueError("Wrong diagnostic experiment")
    old = copy(args); old.config = checked(ROOT / cfg["parent_config"], cfg["parent_config_sha256"])
    old.partition = "calibration"; old.prediction_dir = (ROOT / cfg["parent_prediction"]).parent
    for key in ("parent_prediction", "parent_audit", "parent_quality"):
        checked(ROOT / cfg[key], cfg[key + "_sha256"])
    c = context(old); pred = verify_prediction_seal(old, c)
    quality = read_json(ROOT / cfg["parent_quality"]); audit = read_json(ROOT / cfg["parent_audit"])
    if (quality["prediction_result_sha256"] != cfg["parent_prediction_sha256"] or
            audit["prediction_result_sha256"] != cfg["parent_prediction_sha256"] or
            audit["status"] != "passed_independent_native_audit" or
            quality["audit_result_sha256"] != cfg["parent_audit_sha256"]):
        raise ValueError("Original native/quality audit linkage differs")
    target_paths = {r["target"]: checked(args.artifact_root / c["parent"]["labels"] / r["target"],
                    c["labels"][r["target"]]["label_png_sha256"]) for r in pred["predictions"]}
    access = {"experiment_id": EXP, "config_sha256": sha256(args.config), "source_sha256": snapshot(args),
              "target_label_sha256": {n: sha256(p) for n, p in target_paths.items()},
              "source_label_sha256": {n: sha256(p) for n, p in c["source_paths"].items()},
              "baseline_sha256": {n: sha256(p) for n, p in c["baseline_paths"].items()},
              "rgb_sha256": c["rgb"], "targets": c["targets"], "parent_result_sha256": cfg["parent_prediction_sha256"]}
    write_json(args.output_dir / "access_seal.json", access)
    if shutil.disk_usage(args.output_dir).free / 2**30 < cfg["resource_budget"]["minimum_free_disk_gib"]:
        raise RuntimeError("Disk floor")
    source_cache = {}; rows = []; old_rows = {r["target"]: r for r in quality["targets"]}
    limits = cfg["distance_bin_upper_bounds_source_pixels"]
    for rec in pred["predictions"]:
        name, source = rec["target"], rec["selected_source"]
        baseline = label_array(c["baseline_paths"][name]); dense = label_array(old.prediction_dir / rec["path"])
        changed = baseline != dense; yy, xx = np.where(changed)
        if np.any(changed & ((baseline != 4) | (dense != 0))) or int(changed.sum()) != rec["native"]["changed_pixels"]:
            raise ValueError("Original transition mismatch")
        distances = np.empty(0, np.float64); independent_distances = distances.copy()
        if yy.size:
            if source not in source_cache:
                labels = label_array(c["source_paths"][source]); source_cache[source] = (labels, foundation_distances(labels))
            labels, distance_field = source_cache[source]
            index, _ = c["pairs"][(rec["fold"], name, source)]
            maps = []
            for direction in ("ab", "ba"):
                record = c["records"][f"pair{index}_{direction}"]
                path = (ROOT / c["parent"]["runtime"]).parent / record["path"]
                with np.load(checked(path, record["sha256"]), allow_pickle=False) as arrays:
                    maps.append((arrays["warp"].astype(np.float64), arrays["overlap"].astype(np.float64)))
            points = np.stack((xx + .5, yy + .5), axis=-1)[None]
            a, b = c["poses"][name], c["poses"][source]
            for independent in (False, True):
                sp, supported = native_support(points, maps, a, b, c["cameras"][a.camera_id],
                                               c["cameras"][b.camera_id], c["filters"], independent)
                sampled, values, valid = sample_boundary_distance(sp, labels, distance_field, independent)
                if not np.all(supported & valid & (sampled == 0)):
                    raise ValueError("An old deletion no longer satisfies the original rule")
                if independent: independent_distances = values.ravel()
                else: distances = values.ravel()
            if not np.allclose(distances, independent_distances, rtol=0, atol=1e-10):
                raise ValueError("EDT versus independent native/KD-tree distances differ")
        # Only these105 historical calibration targets are decoded after all access hashes were sealed.
        gt = label_array(target_paths[name]); cm = {}
        for arm, image in (("baseline", baseline), ("dense", dense)):
            matrix = np.bincount((gt.ravel().astype(np.int64) * 5 + image.ravel()), minlength=25).reshape(5, 5)
            old_key = "candidate" if arm == "dense" else arm
            if not np.array_equal(matrix, old_rows[name][old_key]): raise ValueError("Original confusion differs")
            cm[arm] = matrix
        histogram = boundary_histogram(distances, gt[changed], limits)
        check_hist = []
        low = -1
        for high in [*limits, float("inf")]:
            member = ((independent_distances > low) & (independent_distances <= high))
            check_hist.append([int(np.sum(member & (gt[changed] == k))) for k in range(5)])
            low = high
        if not np.array_equal(histogram, check_hist): raise ValueError("Independent bins differ")
        counterfactual, restored = protect_boundary(baseline, dense, distances, cfg["fixed_counterfactual_radius_source_pixels"])
        counts = np.bincount(gt[restored], minlength=5)
        expected = cm["dense"].copy(); expected[:, 0] -= counts; expected[:, 4] += counts
        actual = np.zeros((5, 5), np.int64); np.add.at(actual, (gt.ravel(), counterfactual.ravel()), 1)
        if not np.array_equal(expected, actual): raise ValueError("Counterfactual arithmetic differs")
        if any(not np.array_equal(counterfactual == k, baseline == k) for k in (1, 2, 3)):
            raise ValueError("Protected class changed")
        row = {"target": name, "fold": rec["fold"], "source": source,
               "changed": int(changed.sum()), "baseline": cm["baseline"].tolist(), "dense": cm["dense"].tolist(),
               "boundary_counterfactual": actual.tolist(), "restored_by_gt_class": counts.tolist(),
               "distance_bins_by_gt_class": histogram.tolist(), "independent_distance_and_counts_exact": True}
        rows.append(row)
        print(f"diagnose {len(rows)}/105 {name}: old changes={yy.size} protected={int(counts.sum())}", flush=True)
    summary = aggregate(rows); delta = summary["delta_vs_dense_pp"]
    for field, values in (("baseline", quality["summary"]["baseline"]), ("dense", quality["summary"]["candidate"])):
        if summary[field] != values: raise ValueError("Original aggregate metrics differ")
    hist = np.asarray(summary["distance_bins_by_gt_class"])
    if (int(hist[:, 0].sum()) != quality["summary"]["corrected"] or
            int(hist[:, 4].sum()) != quality["summary"]["harmed"] or
            int(hist[:, 1:4].sum()) != quality["summary"]["still_wrong"]):
        raise ValueError("Original transition totals differ")
    for n, p in target_paths.items(): checked(p, access["target_label_sha256"][n])
    for n, p in c["source_paths"].items(): checked(p, access["source_label_sha256"][n])
    for n, p in c["baseline_paths"].items(): checked(p, access["baseline_sha256"][n])
    for n, digest in c["rgb"].items(): checked(args.artifact_root / n, digest)
    freeze = verify_freeze(ROOT / c["original"]["freeze_manifest"], args.artifact_root)
    if freeze != c["frozen"]: raise ValueError("Production freeze differs")
    eligible = delta["miou"] >= 0 and delta["iou"] >= 0 and delta["recall"] > 0
    return {"experiment_id": EXP, "status": "completed_boundary_diagnostic", "config_sha256": sha256(args.config),
            "git_commit": cfg["git_commit"], "seed": cfg["seed"], "dataset_manifest_sha256": c["original"]["dataset_manifest_sha256"],
            "oof_split_sha256": c["original"]["oof_split_sha256"], "access_seal_sha256": sha256(args.output_dir / "access_seal.json"),
            "parent_prediction_sha256": cfg["parent_prediction_sha256"], "parent_quality_sha256": cfg["parent_quality_sha256"],
            "summary": summary, "folds": [{"fold": f, **aggregate([r for r in rows if r["fold"] == f])} for f in sorted(c["folds"])],
            "targets": rows, "target_label_roles_decoded": 105, "source_label_names_decoded": sorted(source_cache),
            "independent_audit_passed": True, "boundary_candidate_worth_confirming": eligible,
            "rgb_bytes_exact": True, "protected_classes_exact": True, "freeze": freeze,
            "new_matching": 0, "new_training": 0, "semantic_png_exports": 0, "check_target_decodes": 0, "developer_target_decodes": 0,
            "claim_boundary": "Reused105 training-side association and one fixed2px in-memory counterfactual, not exported predictions or confirmed check/final improvement. Source pixel-center distance is not physical surface distance."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    for key in ("artifact-root", "data-root", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    for k, v in vars(args).items():
        if isinstance(v, Path): setattr(args, k, v.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    try:
        result = run(args); result["seconds"] = time.perf_counter() - start
        if result["seconds"] > read_json(args.config)["resource_budget"]["maximum_seconds_per_stage"]:
            raise RuntimeError("Stage budget exceeded")
        write_json(args.output_dir / "result.json", result)
        print({k: result[k] for k in ("status", "seconds", "boundary_candidate_worth_confirming")})
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"experiment_id": EXP, "error_type": type(exc).__name__, "error": str(exc)})
        raise
