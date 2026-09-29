#!/usr/bin/env python3
"""Sealed actual OOF predictions for the fixed second-source foundation veto."""
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
from PIL import Image
import torch
from src.semantic2d import read_json, write_json, sha256, label_array, confusion_metrics
from src.semantic_second_source_veto import second_source, review_second_source
import run_sem325_native_review as calibration_parent
import run_sem325_check_review as check_parent
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem320_dense_preflight import checked

EXP = "SEM-328-SECOND-SOURCE-VETO-OOF"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"


def setup(args):
    cfg = read_json(args.config)
    if cfg["experiment_id"] != EXP: raise ValueError("Wrong experiment")
    # Enforce check progression before constructing any check target/source context.
    if args.partition == "check":
        if args.calibration_result is None: raise ValueError("Need fixed calibration result")
        qualification = read_json(args.calibration_result)
        if (qualification["experiment_id"] != EXP or qualification["partition"] != "calibration" or
                qualification["config_sha256"] != sha256(args.config) or not qualification["paired_improvement"]):
            raise ValueError("Calibration not qualified for fixed check")
    link = cfg["partitions"][args.partition]
    for key in ("prediction", "audit", "quality"): checked(ROOT / link[key], link[key + "_sha256"])
    checked(ROOT / cfg["boundary_diagnostic"], cfg["boundary_diagnostic_sha256"])
    old = copy(args); old.config = checked(ROOT / cfg["parent_config"], cfg["parent_config_sha256"])
    old.prediction_dir = (ROOT / link["prediction"]).parent
    old.matching_root = ROOT / link["matching_root"] if "matching_root" in link else None
    parent = calibration_parent if args.partition == "calibration" else check_parent
    c = parent.context(old); dense = parent.verify_prediction_seal(old, c)
    old_seal = read_json(old.prediction_dir / "access_seal.json")
    old_audit, old_quality = (read_json(ROOT / link[k]) for k in ("audit", "quality"))
    if (old_seal["matching_geometry_sha256"] != c["parent"]["geometry_sha256"] or
            old_audit["prediction_result_sha256"] != link["prediction_sha256"] or
            old_audit["status"] != "passed_independent_native_audit" or
            old_quality["prediction_result_sha256"] != link["prediction_sha256"] or
            old_quality["audit_result_sha256"] != link["audit_sha256"]):
        raise ValueError("Parent linkage differs")
    targets = []; source_paths = {}
    geometry_rows = {(r["fold"], r["target"]): r for r in c["targets"]}
    for r in dense["predictions"]:
        if r["accessible_sources"] != geometry_rows[(r["fold"], r["target"])]["accessible_sources"]:
            raise ValueError("Parent source list differs")
        source = second_source(r); targets.append({**r, "second_source": source})
        if source:
            fold = c["folds"][r["fold"]]
            if source not in fold["train"] or source in fold["oof"] or source in fold["guard"]:
                raise ValueError("Second source outside same-fold training")
            source_paths[source] = checked(args.artifact_root / c["parent"]["labels"] / source,
                                          c["labels"][source]["label_png_sha256"])
    return locals()


def verify_prediction(args, d):
    result = read_json(args.prediction_dir / "result.json")
    if result["config_sha256"] != sha256(args.config) or result["partition"] != args.partition:
        raise ValueError("New prediction configuration differs")
    seal = read_json(checked(args.prediction_dir / "access_seal.json", result["access_seal_sha256"]))
    if seal["parent_prediction_sha256"] != d["link"]["prediction_sha256"]:
        raise ValueError("Dense parent differs")
    for n, h in seal["source_sha256"].items():
        checked(ROOT / n, h); checked(args.prediction_dir / "source_snapshot" / n, h)
    expected = {(r["target"], r["second_source"]) for r in d["targets"]}
    if {(r["target"], r["second_source"]) for r in result["predictions"]} != expected or len(result["predictions"]) != len(expected):
        raise ValueError("Prediction scope differs")
    for r in result["predictions"]: checked(args.prediction_dir / r["path"], r["sha256"])
    return result


def invariant(args, d, seal):
    c = d["c"]
    for n, p in d["source_paths"].items(): checked(p, seal["second_source_label_sha256"][n])
    for n, p in c["baseline_paths"].items(): checked(p, seal["baseline_sha256"][n])
    for n, h in c["rgb"].items(): checked(args.artifact_root / n, h)
    if verify_freeze(ROOT / c["original"]["freeze_manifest"], args.artifact_root) != c["frozen"]:
        raise ValueError("Frozen production artifacts changed")


def predict(args, d, independent=False):
    c = d["c"]; prior = verify_prediction(args, d) if independent else None
    seal = {"experiment_id": EXP, "partition": args.partition, "config_sha256": sha256(args.config),
            "source_sha256": calibration_parent.snapshot(args), "parent_prediction_sha256": d["link"]["prediction_sha256"],
            "source_roles": [{k: r[k] for k in ("fold", "target", "selected_source", "second_source")} for r in d["targets"]],
            "second_source_label_sha256": {n: sha256(p) for n, p in d["source_paths"].items()},
            "baseline_sha256": {n: sha256(p) for n, p in c["baseline_paths"].items()}, "rgb_sha256": c["rgb"],
            "target_label_roles_decoded": 0}
    write_json(args.output_dir / "access_seal.json", seal)
    cache = {}; rows = []; prior_rows = {r["target"]: r for r in prior["predictions"]} if prior else {}
    for r in d["targets"]:
        name, source = r["target"], r["second_source"]
        base = label_array(c["baseline_paths"][name]); dense = label_array(d["old"].prediction_dir / r["path"])
        kwargs = {"independent": independent}
        if source and np.any(base != dense):
            if source not in cache: cache[source] = label_array(d["source_paths"][source])
            index, _ = c["pairs"][(r["fold"], name, source)]; maps = []
            for direction in ("ab", "ba"):
                record = c["records"][f"pair{index}_{direction}"]
                path = (ROOT / c["parent"]["runtime"]).parent / record["path"]
                with np.load(checked(path, record["sha256"]), allow_pickle=False) as arrays:
                    maps.append((arrays["warp"].astype(np.float64), arrays["overlap"].astype(np.float64)))
            a, b = c["poses"][name], c["poses"][source]
            kwargs.update(source_labels=cache[source], maps=maps, target_pose=a, source_pose=b,
                          target_camera=c["cameras"][a.camera_id], source_camera=c["cameras"][b.camera_id], filters=c["filters"])
        candidate, stats = review_second_source(base, dense, **kwargs)
        bypass, _ = review_second_source(base, dense, enabled=False)
        if not np.array_equal(bypass, dense): raise ValueError("Disabled mode differs")
        changed = base != candidate
        if np.any(changed & ((base != 4) | (candidate != 0))): raise ValueError("Invalid new transition")
        if any(not np.array_equal(candidate == k, base == k) for k in (1, 2, 3)):
            raise ValueError("Protected classes differ")
        if independent:
            previous = prior_rows[name]
            if not np.array_equal(candidate, label_array(args.prediction_dir / previous["path"])) or stats != previous["native"]:
                raise ValueError("Independent new full prediction differs: " + name)
            rows.append({"target": name, "second_source": source, "native": stats, "exact": True})
        else:
            path = args.output_dir / "predictions" / name; path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(candidate).save(path)
            rows.append({"target": name, "fold": r["fold"], "selected_source": r["selected_source"], "second_source": source,
                         "path": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "native": stats})
        print(f"{'audit' if independent else 'predict'} {len(rows)}/{len(d['targets'])} {name} veto={stats['foundation_veto_pixels']}", flush=True)
    invariant(args, d, seal)
    return {"experiment_id": EXP, "partition": args.partition, "config_sha256": sha256(args.config),
            "status": "passed_independent_second_source_audit" if independent else "completed_second_source_predictions",
            "access_seal_sha256": sha256(args.output_dir / "access_seal.json"), "predictions": rows,
            "prediction_result_sha256": sha256(args.prediction_dir / "result.json") if independent else None,
            "target_label_roles_decoded": 0, "second_source_names_decoded": sorted(cache), "rgb_bytes_exact": True,
            "protected_classes_exact": True, "disabled_bypass_exact": True, "freeze": c["frozen"], "new_matching": 0, "new_training": 0}


def summarize(rows):
    out = {a: confusion_metrics(sum((np.asarray(r[a]) for r in rows), np.zeros((5, 5), np.int64)))
           for a in ("baseline", "dense", "candidate")}
    for reference in ("baseline", "dense"):
        a, b = out[reference], out["candidate"]
        out["delta_vs_" + reference + "_pp"] = {"miou": 100 * (b["mIoU_with_background"] - a["mIoU_with_background"]),
            **{k: 100 * (b["per_class"]["foundation"][k] - a["per_class"]["foundation"][k]) for k in ("iou", "precision", "recall")}}
    for k in ("restored_by_gt_class", "remaining_deletions_by_gt_class"):
        out[k] = np.sum([r[k] for r in rows], axis=0).tolist()
    return out


def score(args, d):
    c = d["c"]; prediction = verify_prediction(args, d)
    audit = read_json(args.audit_dir / "result.json")
    if (audit["status"] != "passed_independent_second_source_audit" or
            audit["prediction_result_sha256"] != sha256(args.prediction_dir / "result.json") or
            audit["config_sha256"] != sha256(args.config)):
        raise ValueError("Missing independent prediction audit")
    paths = {r["target"]: checked(args.artifact_root / c["parent"]["labels"] / r["target"],
             c["labels"][r["target"]]["label_png_sha256"]) for r in d["targets"]}
    write_json(args.output_dir / "score_access_seal.json", {"experiment_id": EXP, "config_sha256": sha256(args.config),
        "prediction_result_sha256": sha256(args.prediction_dir / "result.json"), "audit_result_sha256": sha256(args.audit_dir / "result.json"),
        "target_label_sha256": {n: sha256(p) for n, p in paths.items()}, "source_sha256": calibration_parent.snapshot(args)})
    old_rows = {r["target"]: r for r in d["old_quality"]["targets"]}
    dense_rows = {r["target"]: r for r in d["targets"]}; rows = []
    for rec in prediction["predictions"]:
        n = rec["target"]; gt = label_array(paths[n]); base = label_array(c["baseline_paths"][n])
        dense = label_array(d["old"].prediction_dir / dense_rows[n]["path"]); candidate = label_array(args.prediction_dir / rec["path"])
        matrices = {}
        for arm, image in (("baseline", base), ("dense", dense), ("candidate", candidate)):
            cm = np.bincount(gt.ravel().astype(np.int64) * 5 + image.ravel(), minlength=25).reshape(5, 5)
            independent = np.zeros((5, 5), np.int64); np.add.at(independent, (gt.ravel(), image.ravel()), 1)
            if not np.array_equal(cm, independent): raise ValueError("Independent confusion differs")
            if arm != "candidate" and not np.array_equal(cm, old_rows[n]["candidate" if arm == "dense" else arm]):
                raise ValueError("Original SEM325 confusion differs")
            matrices[arm] = cm
        restore = dense != candidate
        if np.any(restore & ((candidate != 4) | (base != 4) | (dense != 0))): raise ValueError("Invalid restore map")
        counts = np.bincount(gt[restore], minlength=5); remain = np.bincount(gt[base != candidate], minlength=5)
        expected = matrices["dense"].copy(); expected[:, 0] -= counts; expected[:, 4] += counts
        if not np.array_equal(expected, matrices["candidate"]) or counts.sum() != rec["native"]["foundation_veto_pixels"]:
            raise ValueError("Independent restoration accounting differs")
        rows.append({"target": n, "fold": rec["fold"], **{k: v.tolist() for k, v in matrices.items()},
                     "restored_by_gt_class": counts.tolist(), "remaining_deletions_by_gt_class": remain.tolist()})
    summary = summarize(rows)
    for new, old in (("baseline", "baseline"), ("dense", "candidate")):
        if summary[new] != d["old_quality"]["summary"][old]: raise ValueError("Original full metrics differ")
    for n, p in paths.items(): checked(p, c["labels"][n]["label_png_sha256"])
    invariant(args, d, read_json(args.prediction_dir / "access_seal.json"))
    delta = summary["delta_vs_dense_pp"]
    improved = delta["miou"] > 0 and delta["iou"] > 0 and delta["recall"] >= 0
    return {"experiment_id": EXP, "status": "completed_paired_improvement" if improved else "completed_paired_tradeoff",
            "partition": args.partition, "config_sha256": sha256(args.config), "seed": d["cfg"]["seed"], "git_commit": d["cfg"]["git_commit"],
            "dataset_manifest_sha256": c["original"]["dataset_manifest_sha256"], "oof_split_sha256": c["original"]["oof_split_sha256"],
            "prediction_result_sha256": sha256(args.prediction_dir / "result.json"), "audit_result_sha256": sha256(args.audit_dir / "result.json"),
            "score_access_seal_sha256": sha256(args.output_dir / "score_access_seal.json"), "paired_improvement": improved,
            "summary": summary, "folds": [{"fold": f, **summarize([r for r in rows if r["fold"] == f])} for f in sorted(c["folds"])],
            "targets": rows, "target_label_roles_decoded": len(rows), "independent_confusions_and_accounting_exact": True,
            "claim_boundary": "Actual native OOF predictions on historically reused calibration/check. No74, final deployment or fresh RGB/external-camera rendering in this experiment."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=("predict", "audit", "score"))
    p.add_argument("--partition", choices=("calibration", "check"), required=True); p.add_argument("--config", type=Path, default=CONFIG)
    for k in ("artifact-root", "data-root", "output-dir"): p.add_argument("--" + k, type=Path, required=True)
    for k in ("prediction-dir", "audit-dir", "calibration-result"): p.add_argument("--" + k, type=Path)
    args = p.parse_args()
    for k, v in vars(args).items():
        if isinstance(v, Path): setattr(args, k, v.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False); started = time.perf_counter(); torch.set_num_threads(4)
    try:
        d = setup(args)
        if shutil.disk_usage(args.output_dir).free / 2**30 < d["cfg"]["resource_budget"]["minimum_free_disk_gib"]:
            raise RuntimeError("Disk floor")
        result = score(args, d) if args.stage == "score" else predict(args, d, args.stage == "audit")
        result["seconds"] = time.perf_counter() - started
        if result["seconds"] > d["cfg"]["resource_budget"]["maximum_seconds_per_stage"]: raise RuntimeError("Stage budget exceeded")
        write_json(args.output_dir / "result.json", result)
        print({k: result[k] for k in ("status", "partition", "seconds")}, flush=True)
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"experiment_id": EXP, "stage": args.stage, "error_type": type(exc).__name__, "error": str(exc)})
        raise
