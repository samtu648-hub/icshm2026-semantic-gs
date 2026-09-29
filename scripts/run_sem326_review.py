#!/usr/bin/env python3
"""SEM32674 native inference/reconstruction and once-only final developer quality."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from PIL import Image
from src.semantic2d import read_json, write_json, sha256, label_array, update_confusion, confusion_metrics
from src.semantic_dense_review import native_prediction
from src.semantic_dense_final import validate_final_split, segment_groups, final_gates, quality_summaries
from run_sem326_matching import load_plan
from src.semantic_source_access import read_pose_only
from src.geometry_prior.camera_io import read_cameras_txt
from src.semantic_protected_fusion import partition_training_names
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem320_dense_preflight import checked
EXP = "SEM-326-DENSE-FOUNDATION-REVIEW-FINAL"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"


def context(args):
    cfg, original, geo_cfg, frozen, plan = load_plan(args)
    if args.partition != "developer": raise ValueError("Final developer partition required")
    for key in ("label_manifest", "trajectory_segments"):
        checked(ROOT / cfg[key], cfg[key + "_sha256"])
    split = read_json(ROOT / original["semantic_split"])
    train, dev = validate_final_split(split)
    if {p["target"] for p in plan["plans"]} != set(dev) or len(plan["plans"]) != 74:
        raise ValueError("Final74 matching plan scope changed")
    parent = {"labels": cfg["parents"]["labels"]}
    for key, stage in (("runtime", "runtime"), ("geometry", "geometry"), ("geometry_audit", "audit")):
        path = args.matching_root / stage / "result.json"
        parent[key], parent[key + "_sha256"] = str(path.relative_to(ROOT)), sha256(path)
        if read_json(path)["config_sha256"] != sha256(args.config): raise ValueError("Matching config differs")
    geometry, audit, runtime = (read_json(ROOT / parent[k]) for k in ("geometry", "geometry_audit", "runtime"))
    ph = sha256(args.prepare_dir / "result.json")
    if (geometry["prepare_sha256"] != ph or runtime["prepare_sha256"] != ph or
            geometry["runtime_sha256"] != parent["runtime_sha256"] or
            audit["primary_result_sha256"] != parent["geometry_sha256"] or not audit["independent_geometry"]):
        raise ValueError("Unqualified final matching linkage")
    targets = geometry["summary"]["targets"]
    if len(targets) != 74 or {t["target"] for t in targets} != set(dev): raise ValueError("Wrong final target scope")
    for t in targets:
        if t["selected_source"] and t["selected_source"] not in train: raise ValueError("Nontraining source label")
    baseline_root = args.artifact_root / cfg["parents"]["baseline"]
    bm = checked(baseline_root / "prediction_manifest.json", cfg["parents"]["baseline_manifest_sha256"])
    checked(baseline_root / "result.json", cfg["parents"]["baseline_result_sha256"])
    bases = {r["name"]: r for r in read_json(bm)["predictions"]}
    baseline_paths = {n: checked(baseline_root / bases[n]["prediction"], bases[n]["prediction_sha256"]) for n in dev}
    labels = {r["image"]: r for r in read_json(ROOT / cfg["label_manifest"])["records"]}
    bank = plan["reference_bank"]
    if len(bank) != 219 or {r["name"] for r in bank} != set(train): raise ValueError("Final reference bank differs")
    source_paths = {}
    for r in bank:
        if r["label_sha256"] != labels[r["name"]]["label_png_sha256"]: raise ValueError("Bank label provenance differs")
        source_paths[r["name"]] = checked(args.artifact_root / r["label_path"], r["label_sha256"])
        checked(args.data_root / r["source_path"], r["source_sha256"])
    camera_dir = args.data_root / "camera_parameters"
    poses = read_pose_only(checked(camera_dir / "images.txt", plan["camera_images_sha256"]))
    cameras = read_cameras_txt(checked(camera_dir / "cameras.txt", plan["camera_intrinsics_sha256"]))
    records = {r["name"]: r for r in runtime["probes"]}
    pairs = {(p["fold"], p["target"], p["source"]): (i, p) for i, p in enumerate(plan["pairs"])}
    rgb = {p["target_path"]: p["target_sha256"] for p in plan["pairs"]}
    for name, digest in rgb.items(): checked(args.artifact_root / name, digest)
    if frozen != geometry["freeze"]: raise ValueError("Production freeze changed")
    filters = geo_cfg["filters"]
    groups = segment_groups(read_json(ROOT / cfg["trajectory_segments"]), dev)
    return locals()


def snapshot(args):
    paths = {str(args.config.relative_to(ROOT)): sha256(args.config)}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix == ".py" and not path.is_relative_to(ROOT / "outputs"):
                paths[str(path.relative_to(ROOT))] = sha256(path)
    for name in paths:
        dest = args.output_dir / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    return paths


def verify_prediction_seal(args, c):
    result = read_json(args.prediction_dir / "result.json")
    seal = read_json(checked(args.prediction_dir / "access_seal.json", result["access_seal_sha256"]))
    if result["config_sha256"] != sha256(args.config) or result["partition"] != args.partition:
        raise ValueError("Prediction configuration changed")
    for name, digest in seal["source_sha256"].items():
        checked(ROOT / name, digest); checked(args.prediction_dir / "source_snapshot" / name, digest)
    expected = {t["target"] for t in c["targets"]}
    if {r["target"] for r in result["predictions"]} != expected or len(result["predictions"]) != len(expected):
        raise ValueError("Prediction manifest scope differs")
    for r in result["predictions"]: checked(args.prediction_dir / r["path"], r["sha256"])
    return result


def predict(args, c, independent=False):
    prior = verify_prediction_seal(args, c) if independent else None
    if independent:
        import torch
        torch.set_num_threads(4)
    seal = {"experiment_id": EXP, "partition": args.partition, "config_sha256": sha256(args.config),
            "source_sha256": snapshot(args), "source_label_roles": c["targets"],
            "source_label_sha256": {n: sha256(p) for n, p in c["source_paths"].items()},
            "baseline_sha256": {n: sha256(p) for n, p in c["baseline_paths"].items()},
            "rendered_rgb_sha256": c["rgb"], "target_label_roles_decoded": [],
            "matching_geometry_sha256": c["parent"]["geometry_sha256"]}
    write_json(args.output_dir / "access_seal.json", seal)
    rows = []
    for target in c["targets"]:
        name, source = target["target"], target["selected_source"]
        baseline = label_array(c["baseline_paths"][name])
        stats = {"reliable_pixels": 0, "outside_source_labels": 0, "changed_pixels": 0}
        output = baseline.copy()
        if source:
            index, pair = c["pairs"][(target["fold"], name, source)]
            maps = []
            for direction in ("ab", "ba"):
                record = c["records"][f"pair{index}_{direction}"]
                path = (ROOT / c["parent"]["runtime"]).parent / record["path"]
                with np.load(checked(path, record["sha256"]), allow_pickle=False) as arrays:
                    maps.append((arrays["warp"].astype(np.float64), arrays["overlap"].astype(np.float64)))
            a, b = c["poses"][name], c["poses"][source]
            output, stats = native_prediction(baseline, label_array(c["source_paths"][source]), maps,
                a, b, c["cameras"][a.camera_id], c["cameras"][b.camera_id], c["filters"], independent)
        changed = output != baseline
        if np.any(changed & ((baseline != 4) | (output != 0))): raise ValueError("Invalid transition")
        for cls in (1, 2, 3):
            if not np.array_equal(output == cls, baseline == cls): raise ValueError("Protected mask changed")
        if independent:
            old = next(r for r in prior["predictions"] if r["target"] == name)
            if not np.array_equal(output, label_array(args.prediction_dir / old["path"])) or stats != old["native"]:
                raise ValueError("Independent native reconstruction differs: " + name)
            rows.append({"target": name, "exact": True, "native": stats})
        else:
            path = args.output_dir / "predictions" / name
            path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(output).save(path)
            rows.append({**target, "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "native": stats})
        print(f"{'audit' if independent else 'predict'} {len(rows)}/{len(c['targets'])} {name} changes={stats['changed_pixels']}", flush=True)
    for n, p in c["source_paths"].items(): checked(p, seal["source_label_sha256"][n])
    for n, p in c["baseline_paths"].items(): checked(p, seal["baseline_sha256"][n])
    for n, digest in c["rgb"].items(): checked(args.artifact_root / n, digest)
    if verify_freeze(ROOT / c["original"]["freeze_manifest"], args.artifact_root) != c["frozen"]:
        raise ValueError("Freeze changed")
    return {"experiment_id": EXP, "partition": args.partition, "config_sha256": sha256(args.config),
            "status": "passed_independent_native_audit" if independent else "completed_native_predictions",
            "access_seal_sha256": sha256(args.output_dir / "access_seal.json"), "predictions": rows,
            "prediction_result_sha256": sha256(args.prediction_dir / "result.json") if independent else None,
            "protected_masks_exact": True, "rendered_rgb_bytes_exact": True, "freeze": c["frozen"],
            "new_rgb_rendering": False, "target_label_roles_decoded": 0}


def metric_summary(rows):
    result = {k: confusion_metrics(sum((np.array(r[k]) for r in rows), np.zeros((5, 5), np.int64))) for k in ("baseline", "candidate")}
    base, candidate = result["baseline"], result["candidate"]
    result["delta_miou_pp"] = 100 * (candidate["mIoU_with_background"] - base["mIoU_with_background"])
    for metric in ("iou", "precision", "recall"):
        x, y = candidate["per_class"]["foundation"][metric], base["per_class"]["foundation"][metric]
        result[f"delta_foundation_{metric}_pp"] = 100 * (x - y) if x is not None and y is not None else None
    for key in ("changed", "corrected", "harmed", "still_wrong"):
        result[key] = sum(r[key] for r in rows)
    return result


def score(args, c):
    prediction = verify_prediction_seal(args, c)
    audit = read_json(args.audit_dir / "result.json")
    if audit["status"] != "passed_independent_native_audit" or audit["prediction_result_sha256"] != sha256(args.prediction_dir / "result.json"):
        raise ValueError("Missing native reconstruction audit")
    external = read_json(args.external_dir / "result.json")
    if external["status"] != "passed_external_dense_review_audit" or external["config_sha256"] != sha256(args.config) or len(external["probes"]) != 4:
        raise ValueError("Four actual external probes required before target GT scoring")
    target_paths = {n: checked(args.artifact_root / c["parent"]["labels"] / n, c["labels"][n]["label_png_sha256"]) for n in c["dev"]}
    write_json(args.output_dir / "score_access_seal.json", {
        "prediction_result_sha256": sha256(args.prediction_dir / "result.json"),
        "audit_result_sha256": sha256(args.audit_dir / "result.json"),
        "external_result_sha256": sha256(args.external_dir / "result.json"),
        "target_label_sha256": {n: sha256(p) for n,p in target_paths.items()},
        "config_sha256": sha256(args.config), "source_sha256": snapshot(args)})
    rows = []
    for rec in prediction["predictions"]:
        name = rec["target"]
        gt = label_array(target_paths[name]); baseline = label_array(c["baseline_paths"][name])
        candidate = label_array(args.prediction_dir / rec["path"])
        changed = baseline != candidate
        if np.any(changed & ((baseline != 4) | (candidate != 0))): raise ValueError("Invalid final transition")
        matrices = {}
        for key, image in (("baseline", baseline), ("candidate", candidate)):
            cm = np.zeros((5,5), np.int64); update_confusion(cm, gt, image)
            independent = np.histogram2d(gt.ravel(), image.ravel(), bins=(np.arange(6)-.5, np.arange(6)-.5))[0].astype(np.int64)
            if not np.array_equal(cm, independent): raise ValueError("Independent confusion differs")
            matrices[key] = cm.tolist()
        rows.append({"target": name, **matrices, "changed": int(changed.sum()),
                     "corrected": int((changed & (candidate == gt)).sum()), "harmed": int((changed & (baseline == gt)).sum()),
                     "still_wrong": int((changed & (baseline != gt) & (candidate != gt)).sum())})
    summary = metric_summary(rows)
    metrics, groups, subsets, per_view = quality_summaries(rows, c["groups"])
    previous = read_json(c["baseline_root"] / "result.json")["metrics"]
    if metrics["baseline"]["confusion_matrix_rows_gt_columns_prediction"] != previous["confusion_matrix_rows_gt_columns_prediction"]:
        raise ValueError("Frozen SEM161 baseline metric reconstruction differs")
    gates = final_gates(metrics, groups, subsets, c["cfg"]["developer_gates"])
    if verify_freeze(ROOT / c["cfg"]["freeze_manifest"], args.artifact_root) != c["frozen"]: raise ValueError("Freeze changed")
    return {"experiment_id": EXP, "partition": args.partition, "config_sha256": sha256(args.config),
            "git_commit": c["cfg"]["git_commit"], "seed": c["cfg"]["seed"], "dataset_manifest_sha256": c["original"]["dataset_manifest_sha256"],
            "semantic_split_sha256": c["cfg"]["semantic_split_sha256"],
            "prediction_result_sha256": sha256(args.prediction_dir / "result.json"),
            "audit_result_sha256": sha256(args.audit_dir / "result.json"),
            "external_result_sha256": sha256(args.external_dir / "result.json"),
            "score_access_seal_sha256": sha256(args.output_dir / "score_access_seal.json"),
            "status": "completed_developer_gate_passed" if all(gates.values()) else "completed_developer_gate_failed",
            "passed": all(gates.values()), "summary": summary, "metrics": metrics, "groups": c["groups"], "group_metrics": groups,
            "subsets": subsets, "per_view": per_view, "gates": gates, "targets": rows, "target_label_roles_decoded": 74,
            "independent_confusions_exact": True, "production_overwritten": False,
            "claim_boundary": "One fixed final recipe on historically reused74 PINHOLE developer views, not blind or official hidden evaluation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "audit", "score"))
    for name in ("artifact-root", "data-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prediction-dir", "audit-dir", "matching-root", "external-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--partition", choices=("developer",), default="developer")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    args.prepare_dir = args.matching_root / "prepare"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        c = context(args)
        if shutil.disk_usage(args.output_dir).free / 2**30 < c["cfg"]["resource_budget"]["minimum_free_disk_gib"]:
            raise RuntimeError("Disk floor")
        result = score(args, c) if args.stage == "score" else predict(args, c, args.stage == "audit")
        result["seconds"] = time.perf_counter() - start
        if result["seconds"] > c["cfg"]["resource_budget"]["maximum_seconds_per_stage"]: raise RuntimeError("Stage budget exceeded")
        write_json(args.output_dir / "result.json", result)
        print({k: result[k] for k in ("status", "seconds")}, flush=True)
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"experiment_id": EXP, "stage": args.stage,
                   "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__": main()
