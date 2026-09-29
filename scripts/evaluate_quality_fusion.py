#!/usr/bin/env python3
"""Calibrate matched old-role fusion, then SEM332, only after the full batch."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels, semantic_access
from src.quality_fusion import pipeline, apply_stage, confusion, choose_threshold
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_deck_evidence import add_deck_evidence, change_counts
from src.semantic_protected_fusion import partition_training_names


def open_parent(path, status):
    r = json.loads((path / "result.json").read_text())
    if r["status"] != status:
        raise ValueError(f"Incomplete prerequisite: {path.name}")
    if "artifacts" in r:
        verify_hashes(path, r["artifacts"])
    return r


def inputs_for_arm(arm, attempt, rgb_access, split, oof):
    root = ROOT / "outputs/quality"
    index, proofs = {}, {}
    parent_hash = sha256(ROOT / arm["rgb_run"] / "result.json")
    for part in ("oof-0", "oof-1", "oof-2", "development"):
        static = ROOT / arm["reuse_static"] if part == "development" and arm["reuse_static"] else root / "semantic" / f"SEM-330-{arm['name']}-{part}-full-{attempt}"
        sa = ROOT / arm["reuse_static_audit"] if part == "development" and arm["reuse_static_audit"] else root / "audits" / static.name
        sr = open_parent(static, "completed_static_fit_pending_independent_evaluation")
        sar = open_parent(sa, "completed_independent_static_semantic_audit")
        access = semantic_access(split, oof, rgb_access, part)
        if (sr["access"] != access or sr["prediction_views"] != access["prediction"] or sr["trained_steps"] != 10000
                or sr["binding"]["parent_result_sha256"] != parent_hash
                or sar["semantic_run_result_sha256"] != sha256(static / "result.json")):
            raise ValueError("Static model OOF provenance differs")
        sources = {"static": static}
        proofs[str(static.relative_to(ROOT))] = sha256(static / "result.json")
        proofs[str(sa.relative_to(ROOT))] = sha256(sa / "result.json")
        for branch in ("fiveclass", "deck", "stay_cable"):
            key = branch+"_"+part.replace("-", "")
            expert = ROOT / arm["reuse_expert"][key] if key in arm["reuse_expert"] else root / "experts" / f"SEM-331-{arm['name']}-{branch}-{part}-full-{attempt}"
            audit = ROOT / arm["reuse_expert_audit"][key] if key in arm.get("reuse_expert_audit", {}) else root / "audits" / f"SEM-331-{arm['name']}-{key}-{attempt}"
            er = open_parent(expert, "completed_expert_fit_pending_independent_audit")
            ar = open_parent(audit, "completed_independent_expert_replay")
            if (er["access"] != access or er["prediction_views"] != access["prediction"] or er["trained_steps"] != 3000
                    or er["binding"]["expert"] != branch or er["binding"]["parent_result_sha256"] != parent_hash
                    or er["binding"]["identity"] != sr["binding"]["identity"] or er["binding"]["seed"] != sr["binding"]["seed"]
                    or ar["run_result_sha256"] != sha256(expert / "result.json")):
                raise ValueError("Expert, RGB, seed or OOF partition mismatch")
            sources[branch] = expert
            proofs[str(expert.relative_to(ROOT))] = sha256(expert / "result.json")
            proofs[str(audit.relative_to(ROOT))] = sha256(audit / "result.json")
        for name in access["prediction"]:
            if name in index:
                raise ValueError("Duplicate predicted view")
            index[name] = {"partition": part, **sources}
    if set(index) != set(split["semantic_train_labeled"]+split["semantic_validation_labeled"]):
        raise ValueError("Missing or extra source views")
    return index, proofs


def read_inputs(entry, name):
    import numpy as np
    from PIL import Image
    with Image.open(entry["static"] / "predictions" / name) as im:
        static = np.array(im)
    probs = {}
    for branch, count in (("fiveclass", 5), ("stay_cable", 2), ("deck", 2)):
        with np.load(entry[branch] / "probabilities" / (Path(name).stem+".npz"), allow_pickle=False) as z:
            prob = z["probabilities"]
        from src.quality_experts import validate_probability
        validate_probability(prob, count, (989, 1320))
        probs[branch] = prob if branch == "fiveclass" else prob[1]
    return static, probs["fiveclass"], probs["stay_cable"], probs["deck"]


def score_views(index, names, labels, thresholds, output, extra_threshold=None, extra_enabled=False):
    import numpy as np
    from PIL import Image
    pooled, base_counts = np.zeros((5, 5), np.int64), np.zeros((5, 5), np.int64)
    per_view, fold_counts, totals = {}, {}, {k: 0 for k in ("changed", "corrected", "harmed", "wrong_to_wrong")}
    for name in names:
        inputs = read_inputs(index[name], name)
        stages = pipeline(*inputs, thresholds)
        target = labels.read(name)
        base = stages["deck"]
        actual = base
        if extra_enabled:
            actual, evidence = add_deck_evidence(base, stages["cable"], stages["foundation"], inputs[1][1], inputs[3], extra_threshold)
            stages["deck_evidence"] = actual
            counts = change_counts(base, actual, target)
            for key in totals:
                totals[key] += counts[key]
            counts["history_excluded_true_deck_misses"] = int((evidence["history_excluded"] & (target == 1)).sum())
        else:
            counts = change_counts(inputs[0], base, target)
        matrix = independent_confusion(target, actual)
        pooled += matrix
        base_counts += confusion(target, base)
        fold = index[name]["partition"]
        fold_counts.setdefault(fold, np.zeros((5, 5), np.int64))
        fold_counts[fold] += matrix
        per_view[name] = {"metrics": metrics(matrix), "changes": counts}
        # Persist every stage and its actual modified pixels, not fabricated evidence.
        previous = inputs[0]
        for stage, prediction in stages.items():
            path = output / stage
            path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction).save(path / name)
            if stage != "static":
                mask_path = output / (stage+"-modified")
                mask_path.mkdir(parents=True, exist_ok=True)
                Image.fromarray((prediction != previous).astype(np.uint8)*255).save(mask_path / name)
            previous = prediction
    write_new_json(output / "per_view.json", per_view)
    return {"metrics": metrics(pooled), "baseline_metrics": metrics(base_counts), "changes": totals if extra_enabled else None,
            "per_partition": {k: metrics(v) for k, v in fold_counts.items()}, "views": names}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "artifact-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--arm", choices=("Scaffold", "PGSR", "SOF-NVS", "SOF-GEO"), required=True)
    p.add_argument("--deck-evidence", action="store_true")
    a = p.parse_args()
    if a.deck_evidence and a.arm != "Scaffold":
        p.error("SEM332 first round is Scaffold only")
    if a.output.exists():
        raise FileExistsError("Refusing to overwrite fusion evidence")
    a.output.mkdir(parents=True)
    started = time.monotonic()
    config_paths = ["experiments/runs/SEM-331-QUALITY-EXPERTS/fusion_config.yaml", "experiments/runs/SEM-332-DECK-EVIDENCE-OOF/config.yaml"]
    record = {"experiment_id": "SEM-332-DECK-EVIDENCE-OOF" if a.deck_evidence else "SEM-331-QUALITY-EXPERTS",
        "status": "initializing", "arm": a.arm, "automatic_promotion": False,
        "code_sha256": {n: sha256(ROOT / n) for n in ("scripts/evaluate_quality_fusion.py", "src/quality_fusion.py",
            "src/quality_experts.py", "src/quality_deck_evidence.py", "src/quality_static_training.py", "src/quality_semantic_audit.py",
            "src/semantic_protected_fusion.py", "src/semantic2d.py", "src/quality_protocol.py")},
        "config_sha256": {n: sha256(ROOT / n) for n in config_paths},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "historically_reused_development": True, "whole_chain_spatial_validation": False}
    try:
        import numpy as np
        batch = open_parent(a.batch, "completed_four_arm_inputs_fusion_calibration_pending")
        bp = ROOT / "experiments/runs/SEM-331-QUALITY-EXPERTS/batch_config.yaml"
        if sha256(bp) != batch["config_sha256"][str(bp.relative_to(ROOT))]:
            raise ValueError("Batch configuration changed")
        cfg = json.loads((ROOT / config_paths[0]).read_text())
        ecfg = json.loads((ROOT / config_paths[1]).read_text())
        arm = next(v for v in json.loads(bp.read_text())["arms"] if v["name"] == a.arm)
        split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
        oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
        protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
        verify_hashes(ROOT, protocol["input_hashes"])
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        frozen_hashes = {r["path"]: r["sha256"] for r in frozen["artifacts"]}
        verify_hashes(a.artifact_root, frozen_hashes)
        rgb_access = json.loads((ROOT / arm["rgb_run"] / "access.json").read_text())
        index, proofs = inputs_for_arm(arm, a.batch.name, rgb_access, split, oof)
        halves = partition_training_names(sorted(split["semantic_train_labeled"]), expected_count=219)
        if [len(halves[k]) for k in ("calibration", "check")] != [108, 111]:
            raise ValueError("Calibration/check roles changed")
        sc = json.loads((ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/config.yaml").read_text())
        manifest = json.loads((ROOT / sc["label_manifest"]).read_text())
        if sha256(ROOT / sc["label_manifest"]) != oof["label_manifest_sha256"]:
            raise ValueError("Wrong label manifest")
        for name in (*record["code_sha256"], *config_paths):
            dest = a.output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        labels = TrainingLabels(a.artifact_root / sc["labels"], halves["calibration"], manifest, [1320, 989])
        record.update(input_hashes=protocol["input_hashes"], input_result_hashes=proofs, batch_sha256=sha256(a.batch / "result.json"), seed=cfg["seed"])
        write_new_json(a.output / "started.json", record)
        thresholds, curves = {}, {}
        if a.deck_evidence:
            base_path = a.batch / "fusion-Scaffold/calibration_sealed.json"
            base_result = open_parent(base_path.parent, "completed_matched_fusion_pending_human_review")
            if base_result["calibration_sealed_sha256"] != sha256(base_path):
                raise ValueError("Matched baseline calibration seal differs")
            base = json.loads(base_path.read_text())
            if base["input_result_hashes"] != proofs:
                raise ValueError("SEM332 requires the exact matched baseline calibration")
            thresholds = base["thresholds"]
            stages_to_fit = [{"name": "deck_evidence", "target_class": 1, "thresholds": ecfg["threshold_grid"]}]
            record["base_calibration_sha256"] = sha256(base_path)
        else:
            stages_to_fit = cfg["stages"]
        selected_extra = None
        for stage in stages_to_fit:
            matrices = [np.zeros((5, 5), np.int64) for _ in stage["thresholds"]]
            for name in halves["calibration"]:
                inputs = read_inputs(index[name], name)
                current = pipeline(*inputs, thresholds)
                target = labels.read(name)
                previous = {"fiveclass_lane": "static", "cable": "fiveclass_lane", "foundation": "cable", "deck": "foundation", "deck_evidence": "deck"}[stage["name"]]
                for j, threshold in enumerate(stage["thresholds"]):
                    if a.deck_evidence:
                        prediction, _ = add_deck_evidence(current["deck"], current["cable"], current["foundation"], inputs[1][1], inputs[3], threshold)
                    else:
                        prediction = apply_stage(stage["name"], current[previous], *inputs[1:], threshold, current["cable"])
                    matrices[j] += confusion(target, prediction)
            candidates = [{"threshold": t, "metrics": metrics(m)} for t, m in zip(stage["thresholds"], matrices)]
            selected = choose_threshold(candidates, stage["target_class"], deck_primary=a.deck_evidence)
            curves[stage["name"]] = candidates
            if a.deck_evidence:
                selected_extra = selected["threshold"]
            else:
                thresholds[stage["name"]] = selected["threshold"]
            print(json.dumps({"calibrated_stage": stage["name"], "threshold": selected["threshold"]}), flush=True)
        # This immutable record exists before any check/developer label is read.
        sealed = {"thresholds": thresholds, "extra_threshold": selected_extra, "curves": curves,
            "calibration_views": sorted(labels.accessed), "check_labels_read": False, "developer_labels_read": False,
            "input_result_hashes": proofs, "config_sha256": record["config_sha256"]}
        write_new_json(a.output / "calibration_sealed.json", sealed)
        sealed_hash = sha256(a.output / "calibration_sealed.json")
        evaluation = {}
        for role, names in (("calibration", halves["calibration"]), ("check", halves["check"]), ("development", sorted(split["semantic_validation_labeled"]))):
            reader = TrainingLabels(a.artifact_root / sc["labels"], names, manifest, [1320, 989])
            evaluation[role] = score_views(index, names, reader, thresholds, a.output / role, selected_extra, a.deck_evidence)
            write_new_json(a.output / (role+"_metrics.json"), evaluation[role])
            if sha256(a.output / "calibration_sealed.json") != sealed_hash:
                raise ValueError("Calibration changed after holdout labels opened")
        verify_hashes(ROOT, record["code_sha256"])
        verify_hashes(ROOT, record["config_sha256"])
        verify_hashes(ROOT, protocol["input_hashes"])
        verify_hashes(a.artifact_root, frozen_hashes)
        for path, digest in proofs.items():
            if sha256(ROOT / path / "result.json") != digest:
                raise ValueError("Source evidence changed during calibration")
        record.update(status="completed_matched_fusion_pending_human_review", calibration_sealed_sha256=sealed_hash,
            thresholds=thresholds, extra_threshold=selected_extra, evaluation=evaluation, frozen_verified=len(frozen_hashes),
            target_label_hashes={n: next(r["label_png_sha256"] for r in manifest["records"] if r["image"] == n)
                                 for n in split["semantic_train_labeled"]+split["semantic_validation_labeled"]},
            artifacts={str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file()})
    except BaseException as exc:
        record.update(status="fusion_evaluation_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic()-started
        write_new_json(a.output / "result.json", record)
        print(json.dumps({"status": record["status"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
