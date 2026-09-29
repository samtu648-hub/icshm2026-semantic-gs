#!/usr/bin/env python3
"""Frozen-model diagnosis or a single historical foundation-policy control."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_foundation_policy import (STAGES, FrozenFoundationInputs, replace_foundation,
    probability_strata, changes, historical_gate, select_historical)
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_fusion import confusion
from src.semantic_foundation_verifier import foundation_gate, select_foundation_threshold

CODE = ("scripts/run_quality_foundation_study.py", "src/quality_foundation_policy.py", "src/quality_protocol.py",
    "src/quality_static_training.py", "src/quality_experts.py", "src/quality_semantic_audit.py",
    "src/quality_fusion.py", "src/semantic_protected_fusion.py", "src/semantic2d.py",
    "src/semantic_foundation_verifier.py")


def merge_strata(total, rows):
    for name, row in rows.items():
        if name not in total:
            total[name] = {k: list(v) if isinstance(v, list) else v for k, v in row.items()}
        else:
            for key, value in row.items():
                total[name][key] = [a+b for a, b in zip(total[name][key], value)] if isinstance(value, list) else total[name][key]+value


def diagnose(source, output, config):
    source.allowed_roles.update(source.roles)
    result = {}
    for role, names in source.roles.items():
        reader = source.labels(role)
        matrices = {k: np.zeros((5, 5), np.int64) for k in (*STAGES, "raw_expert")}
        histograms, per_view, groups = {}, {}, {}
        costs = {k: 0 for k in ("changed", "corrected", "harmed", "wrong_to_wrong")}
        for i, name in enumerate(names):
            stages, prob, fold = source.read(role, name)
            target = reader.read(name)
            for stage in STAGES:
                matrices[stage] += independent_confusion(target, stages[stage])
            matrices["raw_expert"] += independent_confusion(target, prob.argmax(0).astype(np.uint8))
            h = probability_strata(prob, stages["cable"], target, config["probability_bins"], source.parent["thresholds"]["foundation"])
            c = changes(stages["cable"], stages["foundation"], target)
            merge_strata(histograms, h)
            merge_strata(groups.setdefault(fold, {}), h)
            for k in costs:
                costs[k] += c[k]
            per_view[name] = {"partition": fold, "probability_strata": h, "foundation_rejections": c}
            if i % 25 == 0:
                print(json.dumps({"diagnosing": role, "view": i+1, "total": len(names)}), flush=True)
        if matrices["deck"].tolist() != source.parent["evaluation"][role]["metrics"]["confusion"]:
            raise ValueError("Independent final confusion differs from parent")
        result[role] = {"views": names, "stages": {k: metrics(v) for k, v in matrices.items()},
            "probability_strata": histograms, "per_partition_strata": groups, "foundation_rejections": costs}
        write_new_json(output / role / "per_view.json", per_view)
    return {"status": "completed_fixed_scaffold_diagnosis", "evaluation": result, "selected_threshold": None,
            "threshold_search": False}


def calibrate_policy(source, policy, output):
    names = source.roles["calibration"]
    labels = source.labels("calibration")
    grid = policy["fusion"]["threshold_grid"]
    reference = np.zeros((5, 5), np.int64)
    matrices = [np.zeros((5, 5), np.int64) for _ in grid]
    for i, name in enumerate(names):
        stages, prob, _ = source.read("calibration", name)
        target = labels.read(name)
        before, parent = stages["cable"], stages["deck"]
        reference += confusion(target, replace_foundation(parent, before, prob[4], None))
        for j, threshold in enumerate(grid):
            matrices[j] += confusion(target, replace_foundation(parent, before, prob[4], threshold))
        if i % 25 == 0:
            print(json.dumps({"calibrating": i+1, "total": len(names)}), flush=True)
    baseline = metrics(reference)
    candidates = [{"threshold": t, "metrics": metrics(m)} for t, m in zip(grid, matrices)]
    requirements = policy["oof_gates"]["calibration"]
    for row in candidates:
        row["checks"] = historical_gate(row["metrics"], baseline, requirements)
    chosen = select_historical(candidates)
    threshold = chosen["threshold"] if chosen else source.parent["thresholds"]["foundation"]
    sealed = {"selected_threshold": threshold, "qualifying_candidate": chosen is not None,
        "selection": "historical_constraints_then_foundation_iou_precision_miou_recall_lower_threshold",
        "reference_without_foundation_rejection": baseline, "requirements": requirements,
        "candidates": candidates, "calibration_views": sorted(labels.accessed),
        "check_labels_read_in_this_run": False, "development_labels_read_in_this_run": False,
        "parent_result_sha256": source.config["parent_result_sha256"],
        "other_thresholds_unchanged": {k: v for k, v in source.parent["thresholds"].items() if k != "foundation"}}
    write_new_json(output / "calibration_sealed.json", sealed)
    print(json.dumps({"selected_foundation_threshold": threshold, "qualifying": chosen is not None}), flush=True)
    return sealed


def evaluate_policy(source, threshold, output):
    evaluation = {}
    for role, names in source.roles.items():
        reader = source.labels(role)
        total, base_total = np.zeros((5, 5), np.int64), np.zeros((5, 5), np.int64)
        costs = {k: 0 for k in ("changed", "corrected", "harmed", "wrong_to_wrong")}
        per_view, groups = {}, {}
        for name in names:
            stages, prob, fold = source.read(role, name)
            parent = stages["deck"]
            prediction = replace_foundation(parent, stages["cable"], prob[4], threshold)
            gt = reader.read(name)
            m = independent_confusion(gt, prediction)
            base_m = confusion(gt, parent)
            total += m
            base_total += base_m
            c = changes(parent, prediction, gt)
            for k in costs:
                costs[k] += c[k]
            group = groups.setdefault(fold, {"matrix": np.zeros((5, 5), np.int64), "baseline": np.zeros((5, 5), np.int64)})
            group["matrix"] += m
            group["baseline"] += base_m
            per_view[name] = {"partition": fold, "metrics": metrics(m), "baseline_metrics": metrics(base_m), "changes": c}
            for folder, value in (("predictions", prediction), ("modified", (prediction != parent).astype(np.uint8)*255)):
                dest = output / role / folder / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(value).save(dest)
        if base_total.tolist() != source.parent["evaluation"][role]["metrics"]["confusion"]:
            raise ValueError("Matched baseline changed")
        evaluation[role] = {"views": names, "metrics": metrics(total), "baseline_metrics": metrics(base_total), "changes": costs,
            "per_partition": {k: {"metrics": metrics(v["matrix"]), "baseline_metrics": metrics(v["baseline"])} for k, v in groups.items()}}
        write_new_json(output / role / "per_view.json", per_view)
        print(json.dumps({"evaluated": role, "mIoU": evaluation[role]["metrics"]["mIoU_with_background"], "changes": costs}), flush=True)
    return evaluation


def audit_policy(source, run, record):
    sealed = run / "calibration_sealed.json"
    if sha256(sealed) != record["calibration_sealed_sha256"]:
        raise ValueError("Calibration seal changed")
    seal = json.loads(sealed.read_text())
    threshold = seal["selected_threshold"]
    if threshold != record["selected_threshold"] or seal["parent_result_sha256"] != source.config["parent_result_sha256"]:
        raise ValueError("Sealed choice or parent mismatch")
    if seal["calibration_views"] != sorted(source.roles["calibration"]) or seal["check_labels_read_in_this_run"] or seal["development_labels_read_in_this_run"]:
        raise ValueError("Invalid calibration access seal")
    policy = json.loads((source.root / source.config["historical_policy"]).read_text())
    if [r["threshold"] for r in seal["candidates"]] != policy["fusion"]["threshold_grid"] or seal["requirements"] != policy["oof_gates"]["calibration"]:
        raise ValueError("Historical grid or constraints changed")
    def legacy(m):
        return {**m, "confusion_matrix_rows_gt_columns_prediction": m["confusion"],
                "per_class": {k: {**v, "iou": v["IoU"]} for k, v in m["per_class"].items()}}
    audit_candidates = []
    req = seal["requirements"]
    for row in seal["candidates"]:
        m = legacy(metrics(np.array(row["metrics"]["confusion"], np.int64)))
        if any(m["per_class"]["foundation"][k] is None for k in ("iou", "precision", "recall")):
            if all(row["checks"].values()):
                raise ValueError("Undefined foundation metric passed calibration")
            continue
        gate = foundation_gate(candidate=m, baseline=legacy(seal["reference_without_foundation_rejection"]), immutable_predictions_exact=True,
            minimum_iou_gain=req["minimum_foundation_iou_gain"], minimum_precision_gain=req["minimum_foundation_precision_gain"],
            minimum_recall=req["minimum_foundation_recall"], minimum_miou_gain=req["minimum_mIoU_gain"],
            minimum_background_fp_reduction=req["minimum_background_to_foundation_reduction"])
        if gate["passed"] != all(row["checks"].values()):
            raise ValueError("Historical independent gate disagrees")
        audit_candidates.append({"threshold": row["threshold"], "metrics": m, "gate": gate})
    chosen = select_foundation_threshold(audit_candidates)
    if threshold != (chosen["threshold"] if chosen else source.parent["thresholds"]["foundation"]) or (chosen is not None) != seal["qualifying_candidate"]:
        raise ValueError("Independent historical selection disagrees")
    source.allowed_roles.update(source.roles)
    checked = 0
    for role, names in source.roles.items():
        reader = source.labels(role)
        matrix = np.zeros((5, 5), np.int64)
        cost = {k: 0 for k in ("changed", "corrected", "harmed", "wrong_to_wrong")}
        for name in names:
            stages, prob, _ = source.read(role, name)
            # Independent construction: no replace_foundation call. Only
            # original class4 locations are reclassified; all others stay exact.
            eligible = stages["cable"] == 4
            expected = stages["deck"].copy()
            expected[eligible] = 0
            expected[eligible & (prob[4] >= threshold)] = 4
            for folder, reference in (("predictions", expected), ("modified", (expected != stages["deck"]).astype(np.uint8)*255)):
                path = run / role / folder / name
                if sha256(path) != record["artifacts"][str(path.relative_to(run))]:
                    raise ValueError("Saved candidate hash changed")
                with Image.open(path) as im:
                    actual = np.array(im)
                if not np.array_equal(reference, actual):
                    raise ValueError("Independent prediction reconstruction failed")
            for c in (1, 2, 3):
                if not np.array_equal(expected == c, stages["deck"] == c):
                    raise ValueError("Protected classes differ")
            target = reader.read(name)
            matrix += independent_confusion(target, expected)
            selected = stages["deck"] != expected
            cost["changed"] += int(selected.sum())
            cost["corrected"] += int(np.count_nonzero(target[selected] == expected[selected]))
            cost["harmed"] += int(np.count_nonzero(target[selected] == stages["deck"][selected]))
            cost["wrong_to_wrong"] += int(np.count_nonzero((target[selected] != expected[selected]) & (target[selected] != stages["deck"][selected])))
            checked += 1
        if matrix.tolist() != record["evaluation"][role]["metrics"]["confusion"] or cost != record["evaluation"][role]["changes"]:
            raise ValueError("Independent candidate confusion differs")
    return {"status": "completed_independent_foundation_policy_audit", "views_replayed": checked,
            "run_result_sha256": sha256(run / "result.json"), "probability_network_rerun": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnosis", type=Path)
    p.add_argument("--audit-of", type=Path)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.output.exists():
        raise FileExistsError("Existing study output must never be overwritten")
    if shutil.disk_usage(a.output.parent if a.output.parent.exists() else ROOT).free < config.get("minimum_free_gib", 1)*1024**3:
        raise OSError("Registered disk reserve unavailable")
    a.output.mkdir(parents=True)
    start = time.monotonic()
    record = {"experiment_id": config["experiment_id"], "status": "initializing", "seed": config["seed"],
        "config_sha256": sha256(a.config), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": {n: sha256(ROOT / n) for n in CODE}, "training_steps": 0, "rgb_network_loaded": False,
        "automatic_promotion": False, "historically_reused_development": True, "whole_chain_spatial_validation": False}
    try:
        source = FrozenFoundationInputs(ROOT, a.artifact_root, config)
        record.update(parent_result_sha256=config["parent_result_sha256"], input_hashes=source.file_proofs,
            source_result_hashes=source.source_results, identity=source.identity)
        for name in CODE:
            dest = a.output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        shutil.copy2(a.config, a.output / "config.json")
        write_new_json(a.output / "started.json", record)
        if a.audit_of:
            run_record = json.loads((a.audit_of / "result.json").read_text())
            if run_record["status"] != "completed_fixed_foundation_policy_pending_independent_audit" or run_record["config_sha256"] != record["config_sha256"]:
                raise ValueError("Audit requires the completed exact policy configuration")
            if run_record["code_sha256"] != record["code_sha256"]:
                raise ValueError("Audit code differs from fit")
            for key in ("parent_result_sha256", "source_result_hashes", "input_hashes", "identity"):
                if run_record[key] != record[key]:
                    raise ValueError("Audit sources differ from policy run")
            record.update(audit_policy(source, a.audit_of, run_record))
        elif config.get("diagnostic_only"):
            record.update(diagnose(source, a.output, config))
        else:
            if a.diagnosis is None:
                raise ValueError("Policy study requires completed fixed diagnosis")
            d = json.loads((a.diagnosis / "result.json").read_text())
            if d["status"] != "completed_fixed_scaffold_diagnosis" or d["parent_result_sha256"] != config["parent_result_sha256"]:
                raise ValueError("Diagnosis is incomplete or for different models")
            if d["experiment_id"] != "SEM-333-FOUNDATION-POLICY-DIAG" or d["config_sha256"] != sha256(ROOT / "experiments/runs/SEM-333-FOUNDATION-POLICY-DIAG/config.yaml") or d["source_result_hashes"] != source.source_results:
                raise ValueError("Diagnosis provenance mismatch")
            record["diagnosis_sha256"] = sha256(a.diagnosis / "result.json")
            policy = json.loads((ROOT / config["historical_policy"]).read_text())
            sealed = calibrate_policy(source, policy, a.output)
            digest = sha256(a.output / "calibration_sealed.json")
            source.allowed_roles.update(source.roles)
            evaluation = evaluate_policy(source, sealed["selected_threshold"], a.output)
            if digest != sha256(a.output / "calibration_sealed.json"):
                raise ValueError("Threshold changed after holdout opened")
            record.update(status="completed_fixed_foundation_policy_pending_independent_audit", evaluation=evaluation,
                selected_threshold=sealed["selected_threshold"], qualifying_candidate=sealed["qualifying_candidate"], calibration_sealed_sha256=digest)
        source.finish()
        verify_hashes(ROOT, record["code_sha256"])
        if sha256(a.config) != record["config_sha256"]:
            raise ValueError("Configuration changed during study")
        record.update(frozen_verified=len(source.frozen), access={k: sorted(v) for k, v in source.access.items()},
            artifacts={str(f.relative_to(a.output)): sha256(f) for f in a.output.rglob("*") if f.is_file()})
    except BaseException as exc:
        record.update(status="foundation_study_failed_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic()-start
        write_new_json(a.output / "result.json", record)
        print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
