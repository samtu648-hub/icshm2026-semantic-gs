#!/usr/bin/env python3
"""Read only SEM-311's already scored108 views; no inference or fitting."""
from __future__ import annotations

import argparse
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np
from audit_sem260_error_budget import manifest_paths, png
from check_sem311_quality import check_metrics
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256
from src.semantic_complementarity import describe, direction_screen, edit_vectors
from src.semantic_protected_fusion import partition_training_names
from src.semantic_refinement import class_boundary_distance

ARMS = ("dinov2_vitb14_candidate", "mit_b2_control")


def checked(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError(f"Changed sealed input: {path.name}")
    return path


def run(args):
    cfg = read_json(args.config); parent_path = ROOT/cfg["parent_config"]
    parent = read_json(checked(parent_path, cfg["parent_config_sha256"]))
    run_path = args.run_dir/"result.json"
    source = read_json(checked(run_path, cfg["parent_result_sha256"]))
    if source["status"] != "completed_calibration_gate_failed" or source["mode"] != "full":
        raise ValueError("Expected fixed failed full parent")
    if source["config_sha256"] != cfg["parent_config_sha256"]:
        raise ValueError("Parent config identity differs")
    checked(args.run_dir/"source_snapshot/config.yaml", source["config_sha256"])
    for key in ("semantic_split", "oof_split"):
        checked(ROOT/parent[key], parent[key+"_sha256"])
    semantic = read_json(ROOT/parent["semantic_split"])
    partition = partition_training_names(semantic["semantic_train_labeled"], expected_count=219)
    summary = source["calibration"]
    if read_json(args.run_dir/"calibration_sealed.json") != summary:
        raise ValueError("Calibration seal changed")
    if (source["check_quality_scored_this_run"] or source["developer_quality_scored_this_run"]
            or [r["name"] for r in summary["target_access"]] != partition["calibration"]):
        raise ValueError("Unexpected quality scope")
    seal = read_json(checked(args.run_dir/"prediction_seal.json", source["prediction_seal_sha256"]))
    predictions = read_json(checked(args.run_dir/"prediction_manifest.json", seal["prediction_manifest_sha256"]))
    freeze_path = ROOT/cfg["freeze_manifest"]
    before = verify_freeze(freeze_path, args.artifact_root)
    baseline = manifest_paths(args.artifact_root/parent["parents"]["sem160"], "prediction_manifest.json",
                              "predictions", "prediction", "prediction_sha256")
    rgb = manifest_paths(args.artifact_root/parent["parents"]["rgb"], "rendered_dataset_manifest.json",
                         "rendered_images", "path", "sha256")
    fold_map = {n:str(f["fold"]) for f in read_json(ROOT/parent["oof_split"])["folds"] for n in f["oof"]}
    chunks = {a:{str(i):{"add":[], "remove":[]} for i in range(3)} for a in ARMS}
    view_metrics = {r["name"]:r for r in summary["per_view"]}
    cms = {a:np.zeros((5,5),np.int64) for a in ("baseline", *ARMS)}
    views = []; inputs = []
    for access in summary["target_access"]:
        name = access["name"]; fid = fold_map[name]
        label = checked(args.artifact_root/parent["parents"]["labels"]/name, access["label_sha256"])
        base_path = checked(*baseline[name]); checked(*rgb[name])
        gt = png(label); base = png(base_path)
        prob = {}; pred = {}
        for arm in ARMS:
            rec = predictions[arm][name]
            if str(rec["fold"]) != fid:
                raise ValueError("Parent fold changed")
            pred[arm] = png(checked(rec["path"], rec["sha256"]))
            prob[arm] = np.load(checked(rec["probability_path"], rec["probability_sha256"]), allow_pickle=False)
        gd = class_boundary_distance(gt, 4); bd = class_boundary_distance(base, 4)
        row = {"name":name, "fold":fid, "arms":{}}
        for arm, image in (("baseline", base), *pred.items()):
            cm = np.zeros((5,5),np.int64)
            np.add.at(cm, (gt.reshape(-1), image.reshape(-1)), 1)
            check_metrics(cm, view_metrics[name][arm]); cms[arm] += cm
        for arm, other in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[0])):
            vectors = edit_vectors(gt, base, pred[arm], prob[arm], prob[other], gd, bd)
            row["arms"][arm] = {}
            for direction, values in vectors.items():
                chunks[arm][fid][direction].append(values)
                row["arms"][arm][direction] = {"pixels":len(values["gt"]),
                    "corrected":int(values["useful"].sum()), "harmed":int(values["harmful"].sum()),
                    "gt_counts":np.bincount(values["gt"],minlength=5).tolist()}
        views.append(row)
        inputs.append({"name":name, "fold":fid, "label_sha256":access["label_sha256"],
                       "baseline_sha256":baseline[name][1], "rgb_sha256":rgb[name][1],
                       "arms":{a:predictions[a][name] for a in ARMS}})
        if len(views)%18==0:
            print({"views_checked":len(views), "new_inference":False}, flush=True)
    def combine(rows):
        return {key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
    results = {}
    for arm in ARMS:
        folds = {fid:{d:describe(combine(rows), cfg["confidence_edges"], cfg["boundary_distances_px"])
                      for d,rows in by_direction.items()} for fid,by_direction in chunks[arm].items()}
        aggregate = {d:describe(combine([r for f in chunks[arm].values() for r in f[d]]),
                                 cfg["confidence_edges"], cfg["boundary_distances_px"]) for d in ("add", "remove")}
        expected = summary["changes"][arm]
        if (sum(r["corrected"] for r in aggregate.values()) != expected["corrected"]
                or sum(r["harmed"] for r in aggregate.values()) != expected["harmed"]
                or aggregate["add"]["pixels"] != expected["added_foundation"]
                or aggregate["remove"]["pixels"] != expected["removed_foundation"]):
            raise ValueError("Parent change counts differ")
        results[arm] = {"aggregate":aggregate, "folds":folds,
                       "research_screen":{d:direction_screen({f:r[d] for f,r in folds.items()}) for d in aggregate}}
    for arm, cm in cms.items():
        check_metrics(cm, summary["metrics"][arm])
    after = verify_freeze(freeze_path, args.artifact_root)
    if before != after:
        raise ValueError("Frozen artifacts changed")
    for inp in inputs:
        checked(*baseline[inp["name"]]); checked(*rgb[inp["name"]])
    checked(run_path, cfg["parent_result_sha256"])
    snapshot = args.output_dir/"source_snapshot"; snapshot.mkdir()
    sources = [Path(__file__), ROOT/"src/semantic_complementarity.py", ROOT/"src/semantic_refinement.py",
               ROOT/"scripts/check_sem311_quality.py", ROOT/"src/semantic2d.py",
               ROOT/"src/semantic_protected_fusion.py", ROOT/"scripts/audit_sem260_error_budget.py",
               ROOT/"scripts/run_sem250_colmap_track_deck_diag.py", args.config]
    hashes = {}
    for path in sources:
        shutil.copyfile(path, snapshot/path.name); hashes[str(path.relative_to(ROOT))] = sha256(path)
    return {"experiment_id":cfg["experiment_id"], "status":"completed_read_only_diagnostic",
            "repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "config_sha256":sha256(args.config), "source_sha256":hashes, "seed":cfg["seed"],
            "dataset_manifest_sha256":sha256(ROOT/cfg["dataset_manifest"]),
            "parent_result_sha256":cfg["parent_result_sha256"], "parent_config_sha256":sha256(parent_path),
            "semantic_split_sha256":parent["semantic_split_sha256"], "oof_split_sha256":parent["oof_split_sha256"],
            "quality_scope":"same reused108 calibration, inherited incremental OOF, not blind",
            "views":len(views), "arms":results, "per_view":views, "input_access":inputs,
            "freeze_before":before, "freeze_after":after, "rgb_files_verified_unchanged":len(inputs),
            "parent_metrics_and_changes_exact":True, "check_labels_decoded":0, "developer_labels_decoded":0,
            "new_inference":0, "new_training":0, "candidate_exports":0,
            "interpretation":"Descriptive ranks/cohorts only; GT distances unavailable at inference. No fitted selector or selected confidence/radius. Research screen does not promote a model."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("artifact-root", "run-dir", "output-dir"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"experiments/runs/SEM-312-DINOV2-COMPLEMENTARITY-DIAG/config.yaml")
    args = parser.parse_args(); args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    try:
        result = run(args); result["seconds"] = time.perf_counter()-started
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        write_json(args.output_dir/"result.json", result)
        print({"status":result["status"], "views":result["views"], "seconds":result["seconds"],
               "screen":{a:r["research_screen"] for a,r in result["arms"].items()}}, flush=True)
    except Exception as exc:
        write_json(args.output_dir/"failure.json", {"error_type":type(exc).__name__, "error":str(exc),
                   "seconds":time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
