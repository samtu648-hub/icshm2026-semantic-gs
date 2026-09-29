#!/usr/bin/env python3
"""SEM329 final74 prediction, independent replay and three-arm scoring."""
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
from PIL import Image
from src.semantic2d import read_json, write_json, sha256, label_array, confusion_metrics
from src.semantic_second_source_veto import second_source, review_second_source
from src.semantic_foundation_review_final import final_gates
import run_sem326_review as final_parent
import run_sem325_native_review as calibration_parent
from run_sem250_colmap_track_deck_diag import verify_freeze
from run_sem320_dense_preflight import checked
EXP = "SEM-329-SECOND-SOURCE-VETO-FINAL"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"


def setup(args):
    cfg = read_json(args.config)
    if cfg["experiment_id"] != EXP or args.partition != "developer":
        raise ValueError("SEM329 developer only")
    for k in ("parent_config", "oof_config", "qualified_check", "parent_prediction", "parent_audit",
              "parent_quality", "parent_quality_audit", "veto_core", "dataset_manifest", "semantic_split"):
        checked(ROOT / cfg[k], cfg[k + "_sha256"])
    qualification = read_json(ROOT / cfg["qualified_check"])
    if not qualification["paired_improvement"] or qualification["config_sha256"] != cfg["oof_config_sha256"]:
        raise ValueError("Unqualified fixed OOF rule")
    check_seal = read_json(checked((ROOT / cfg["qualified_check"]).parent / "score_access_seal.json",
                                  qualification["score_access_seal_sha256"]))
    for n, h in check_seal["source_sha256"].items():
        checked(ROOT / n, h)
        checked((ROOT / cfg["qualified_check"]).parent / "source_snapshot" / n, h)
    old = copy(args); old.config = ROOT / cfg["parent_config"]
    old.matching_root = ROOT / cfg["matching_root"]; old.prepare_dir = old.matching_root / "prepare"
    old.prediction_dir = (ROOT / cfg["parent_prediction"]).parent
    c = final_parent.context(old); dense = final_parent.verify_prediction_seal(old, c)
    old_audit, old_quality = read_json(ROOT / cfg["parent_audit"]), read_json(ROOT / cfg["parent_quality"])
    if (old_audit["prediction_result_sha256"] != cfg["parent_prediction_sha256"] or
            old_audit["status"] != "passed_independent_native_audit" or
            old_quality["prediction_result_sha256"] != cfg["parent_prediction_sha256"] or
            old_quality["audit_result_sha256"] != cfg["parent_audit_sha256"] or
            read_json(ROOT / cfg["parent_quality_audit"])["quality_result_sha256"] != cfg["parent_quality_sha256"]):
        raise ValueError("Final parent output/audit linkage differs")
    old_seal = read_json(old.prediction_dir / "access_seal.json")
    if old_seal["matching_geometry_sha256"] != c["parent"]["geometry_sha256"]:
        raise ValueError("Cached final correspondence differs")
    geometry = {r["target"]: r for r in c["targets"]}
    targets = []; source_paths = {}
    for r in dense["predictions"]:
        if r["accessible_sources"] != geometry[r["target"]]["accessible_sources"]:
            raise ValueError("Passing source order differs")
        source = second_source(r); targets.append({**r, "second_source": source})
        if source:
            if source not in c["train"]: raise ValueError("Second source not in train219")
            source_paths[source] = c["source_paths"][source]
    link = {"prediction_sha256": cfg["parent_prediction_sha256"]}
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

def require_engineering(args):
    prediction = read_json(args.prediction_dir / "result.json")
    audit = read_json(args.audit_dir / "result.json")
    external = read_json(args.external_dir / "result.json")
    if (audit["status"] != "passed_independent_second_source_audit" or
            audit["prediction_result_sha256"] != sha256(args.prediction_dir / "result.json") or
            audit["config_sha256"] != sha256(args.config)):
        raise ValueError("Missing independent74 native output audit")
    if (external["status"] != "passed_external_second_source_final_audit" or
            external["config_sha256"] != sha256(args.config) or len(external["probes"]) != 4 or
            not external["native_replay_exact"] or not external["frozen_and_dense_references_exact"]):
        raise ValueError("Four actual current-rule external probes required")
    checked(ROOT / "scripts/run_sem329_external.py", external["wrapper_sha256"])
    for probe in external["probes"]:
        checked(args.external_dir / probe["probe"] / "result.json", probe["result_sha256"])
    return prediction, audit, external


def score(args, d):
    prediction = verify_prediction(args, d)
    _, audit, external = require_engineering(args)
    c = d["c"]
    paths = {n: checked(args.artifact_root / c["parent"]["labels"] / n, c["labels"][n]["label_png_sha256"])
             for n in c["dev"]}
    write_json(args.output_dir / "score_access_seal.json", {
        "config_sha256": sha256(args.config), "source_sha256": calibration_parent.snapshot(args),
        "prediction_result_sha256": sha256(args.prediction_dir / "result.json"),
        "audit_result_sha256": sha256(args.audit_dir / "result.json"),
        "external_result_sha256": sha256(args.external_dir / "result.json"),
        "target_label_sha256": {n: sha256(p) for n, p in paths.items()}})
    old_rows = {r["target"]: r for r in d["old_quality"]["targets"]}
    dense_rows = {r["target"]: r for r in d["targets"]}
    rows = []
    for rec in prediction["predictions"]:
        n = rec["target"]; gt = label_array(paths[n])
        base = label_array(c["baseline_paths"][n])
        dense = label_array(d["old"].prediction_dir / dense_rows[n]["path"])
        candidate = label_array(args.prediction_dir / rec["path"]); matrices = {}
        for arm, image in (("baseline", base), ("dense", dense), ("candidate", candidate)):
            cm = np.bincount(gt.ravel().astype(np.int64)*5 + image.ravel(), minlength=25).reshape(5, 5)
            independent = np.zeros((5, 5), np.int64); np.add.at(independent, (gt.ravel(), image.ravel()), 1)
            if not np.array_equal(cm, independent): raise ValueError("Independent confusion differs")
            if arm != "candidate" and not np.array_equal(cm, old_rows[n]["candidate" if arm == "dense" else arm]):
                raise ValueError("Original SEM161/326 per-view confusion differs")
            matrices[arm] = cm
        restored = dense != candidate
        if np.any(restored & ((candidate != 4) | (dense != 0) | (base != 4))):
            raise ValueError("Invalid final veto")
        counts = np.bincount(gt[restored], minlength=5)
        remain = np.bincount(gt[base != candidate], minlength=5)
        expected = matrices["dense"].copy(); expected[:, 0] -= counts; expected[:, 4] += counts
        if not np.array_equal(expected, matrices["candidate"]) or counts.sum() != rec["native"]["foundation_veto_pixels"]:
            raise ValueError("Independent restoration accounting differs")
        rows.append({"target": n, **{k: v.tolist() for k,v in matrices.items()},
                     "restored_by_gt_class": counts.tolist(), "remaining_deletions_by_gt_class": remain.tolist()})
    summary = summarize(rows)
    for new, old in (("baseline", "baseline"), ("dense", "candidate")):
        if summary[new] != d["old_quality"]["summary"][old]: raise ValueError("Original aggregate metrics differ")
    by_name = {r["target"]: r for r in rows}
    groups = {name: summarize([by_name[n] for n in names]) for name, names in c["groups"].items()}
    per_view = {n: {a: confusion_metrics(np.asarray(r[a]))["mIoU_with_background"]
                   for a in ("baseline", "dense", "candidate")} for n,r in by_name.items()}
    worst = sorted(per_view, key=lambda n: (per_view[n]["baseline"], n))[:8]
    challenge = [n for n in per_view if 140 <= int(Path(n).stem) <= 146]
    subsets = {}
    for name, names in (("baseline_worst10", worst), ("challenge_140_146", challenge)):
        if names != d["old_quality"]["subsets"][name]["names"]: raise ValueError("Historical fixed subset differs")
        subsets[name] = {"names": names, **{a+"_mean_miou": float(np.mean([per_view[n][a] for n in names]))
                                         for a in ("baseline", "dense", "candidate")}}
    historical = final_gates(summary, groups, subsets, c["cfg"]["developer_gates"])
    for n,p in paths.items(): checked(p, c["labels"][n]["label_png_sha256"])
    invariant(args, d, read_json(args.prediction_dir / "access_seal.json"))
    benefits = {name: {"miou_positive": summary["delta_vs_"+name+"_pp"]["miou"] > 0,
                      "foundation_iou_positive": summary["delta_vs_"+name+"_pp"]["iou"] > 0,
                      "foundation_recall_delta_pp": summary["delta_vs_"+name+"_pp"]["recall"],
                      "foundation_precision_delta_pp": summary["delta_vs_"+name+"_pp"]["precision"]}
                for name in ("baseline", "dense")}
    return {"experiment_id": EXP, "status": "completed_three_arm_final_comparison", "partition": "developer",
            "config_sha256": sha256(args.config), "git_commit": d["cfg"]["git_commit"], "seed": d["cfg"]["seed"],
            "dataset_manifest_sha256": d["cfg"]["dataset_manifest_sha256"], "semantic_split_sha256": d["cfg"]["semantic_split_sha256"],
            "prediction_result_sha256": sha256(args.prediction_dir/"result.json"),
            "audit_result_sha256": sha256(args.audit_dir/"result.json"), "external_result_sha256": sha256(args.external_dir/"result.json"),
            "score_access_seal_sha256": sha256(args.output_dir/"score_access_seal.json"),
            "summary": summary, "group_metrics": groups, "groups": c["groups"], "subsets": subsets, "per_view": per_view,
            "targets": rows, "benefits": benefits, "historical_sem326_gates_on_candidate": historical,
            "target_label_roles_decoded": 74, "independent_confusions_and_accounting_exact": True,
            "production_overwritten": False,
            "claim_boundary": "Three paired arms on historically reused74, no blind/significance claim. Benefits and recall/precision/local tradeoffs are distinct from historical acceptance gates."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=("predict", "audit", "score"))
    p.add_argument("--config", type=Path, default=CONFIG)
    for k in ("artifact-root", "data-root", "output-dir"): p.add_argument("--"+k, type=Path, required=True)
    for k in ("prediction-dir", "audit-dir", "external-dir"): p.add_argument("--"+k, type=Path)
    args = p.parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path): setattr(args,k,v.resolve())
    args.partition = "developer"; args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); torch.set_num_threads(4)
    try:
        d=setup(args)
        if shutil.disk_usage(args.output_dir).free/2**30 < d["cfg"]["resource_budget"]["minimum_free_disk_gib"]: raise RuntimeError("Disk floor")
        result=score(args,d) if args.stage=="score" else predict(args,d,args.stage=="audit")
        result["seconds"]=time.perf_counter()-started
        if result["seconds"]>d["cfg"]["resource_budget"]["maximum_seconds_per_stage"]: raise RuntimeError("Stage budget")
        write_json(args.output_dir/"result.json",result); print({k:result[k] for k in ("status","seconds")},flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json",{"experiment_id":EXP,"stage":args.stage,"error_type":type(exc).__name__,"error":str(exc)})
        raise
