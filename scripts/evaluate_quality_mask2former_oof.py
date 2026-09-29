#!/usr/bin/env python3
"""SEM341: three-arm108 calibration sealed before111; fresh independent CPU replay."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_mask2former_training import CONFIG, CODE as TRAIN_CODE, Context
from src.quality_mask2former_comparison import (ARMS, VARIANTS, CONTRASTS, independent_map,
    validate_arms, assert_baseline_counts, class_transitions)
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_experts import validate_probability
from src.quality_fusion import pipeline, apply_stage, confusion, choose_threshold
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_foundation_policy import changes
from src.quality_paired_calibration import LabelRelease
from src.semantic_protected_fusion import partition_training_names
from scripts.evaluate_quality_expert_pair import source_index, CODE as PAIR_CODE

CODE = tuple(sorted(set(TRAIN_CODE + PAIR_CODE + (
    "scripts/evaluate_quality_mask2former_oof.py", "scripts/audit_quality_mask2former_oof.py",
    "src/quality_mask2former_comparison.py", "src/quality_paired_calibration.py"))))


def accepted_index(batch, artifact_root, cfg):
    paired_cfg = json.loads((ROOT / cfg["pair_config"]).read_text())
    _, _, index, proofs = source_index(ROOT, paired_cfg, ROOT / cfg["pair_batch"])
    proofs = {n + "/result.json": digest for n, digest in proofs.items()}
    contexts = []
    for part in cfg["partitions"]:
        ctx = Context(ROOT, artifact_root, part)
        contexts.append(ctx)
        run, audit = batch / ("full-"+part), batch / ("audit-"+part)
        r, a = (json.loads((p / "result.json").read_text()) for p in (run, audit))
        if (r["status"] != "completed_mask2former_fit_pending_replay" or r["trained_steps"] != 6000
                or r["mode"] != "full" or r["binding"] != ctx.binding or r["access"] != ctx.access
                or r["prediction_views"] != ctx.access["prediction"] or r["target_labels_decoded"]
                or a["status"] != "completed_mask2former_fold_acceptance" or not a["two_fresh_exact"]
                or a["binding"] != ctx.binding or a["run_result_sha256"] != sha256(run / "result.json")
                or a["prediction_views"] != ctx.access["prediction"]
                or a["script_sha256"] != sha256(ROOT / "scripts/audit_quality_mask2former_oof.py")):
            raise ValueError("Incomplete, unaccepted or differently bound candidate fold")
        verify_hashes(run, r["artifacts"])
        verify_hashes(ROOT, a["replay_result_hashes"])
        proofs.update(a["replay_result_hashes"])
        for folder in (run, audit):
            proofs[str((folder / "result.json").relative_to(ROOT))] = sha256(folder / "result.json")
        for name in ctx.access["prediction"]:
            fold, sources = index[name]
            if (fold != part or sources["fiveclass"][1]["access"] != ctx.access
                    or sources["static"][1]["binding"]["identity"] != ctx.binding["rgb_identity"]):
                raise ValueError("Candidate/control OOF, RGB or geometry identity differs")
            # Entries share a fold's source dictionary; repeated identical assignment is intentional.
            sources["mask2former"] = (run, r)
    if set(index) != set(contexts[0].split["semantic_train_labeled"]) or len(index) != 219:
        raise ValueError("Expected exact219 sharedRGB screening predictions")
    return index, proofs, contexts


def load_inputs(sources, name):
    path, result = sources["static"]
    rel = "predictions/"+name
    verify_hashes(path, {rel: result["artifacts"][rel]})
    with Image.open(path / rel) as image:
        static = np.array(image)
    probs = {}
    for output_name, source_name, prefix, classes in (
        ("b2_3k", "fiveclass", "step-3000/", 5), ("b2_6k", "sixk", "step-6000/", 5),
        ("mask2former_6k", "mask2former", "", 5), ("deck", "deck", "", 2),
        ("stay_cable", "stay_cable", "", 2)):
        path, result = sources[source_name]
        rel = prefix+"probabilities/"+Path(name).stem+".npz"
        verify_hashes(path, {rel: result["artifacts"][rel]})
        with np.load(path / rel, allow_pickle=False) as z:
            value = z["probabilities"]
        validate_probability(value, classes, (989, 1320))
        probs[output_name] = value if classes == 5 else value[1]
    return static, probs


def calibrate(index, names, labels, grid, replay=False):
    thresholds, curves = {arm: {} for arm in ARMS}, {arm: {} for arm in ARMS}
    counter = independent_confusion if replay else confusion
    for stage in grid["stages"]:
        counts = {arm: [np.zeros((5, 5), np.int64) for _ in stage["thresholds"]] for arm in ARMS}
        for i, name in enumerate(names):
            static, probs = load_inputs(index[name][1], name)
            target = labels.read(name)
            for arm in ARMS:
                current = None if replay else pipeline(static, probs[arm], probs["stay_cable"], probs["deck"], thresholds[arm])
                previous = {"fiveclass_lane": "static", "cable": "fiveclass_lane", "foundation": "cable", "deck": "foundation"}[stage["name"]]
                for j, threshold in enumerate(stage["thresholds"]):
                    choice = dict(thresholds[arm], **{stage["name"]: threshold})
                    # Later stages are OFF: same sequential objective as SEM331.
                    inputs = (static, probs[arm], probs["stay_cable"], probs["deck"], choice)
                    prediction = independent_map(*inputs) if replay else apply_stage(stage["name"], current[previous],
                        probs[arm], probs["stay_cable"], probs["deck"], threshold, current["cable"])
                    counts[arm][j] += counter(target, prediction)
            if i % 24 == 0:
                print(json.dumps({"calibration_stage": stage["name"], "views": i+1, "total": len(names), "replay": replay}), flush=True)
        for arm in ARMS:
            rows = [{"threshold": t, "metrics": metrics(m)} for t, m in zip(stage["thresholds"], counts[arm])]
            curves[arm][stage["name"]] = rows
            thresholds[arm][stage["name"]] = choose_threshold(rows, stage["target_class"])["threshold"]
        print(json.dumps({"calibrated_stage": stage["name"], "choices": thresholds}), flush=True)
    return thresholds, curves


def score(index, names, labels, cfg, thresholds, output, role, replay_source=None):
    pooled = {k: np.zeros((5, 5), np.int64) for k in VARIANTS}
    costs = {k: {n: 0 for n in ("changed", "corrected", "harmed", "wrong_to_wrong")} for k in CONTRASTS}
    cubes = {k: np.zeros((5, 5, 5), np.int64) for k in CONTRASTS}
    groups, per_view = {}, {}
    for i, name in enumerate(names):
        part, sources = index[name]
        static, probs = load_inputs(sources, name)
        maps = {}
        for arm in ARMS:
            maps[arm+"/raw"] = probs[arm].argmax(0).astype(np.uint8)
            for mode, t in (("fixed", cfg["comparison"]["fixed_thresholds"]), ("calibrated", thresholds[arm])):
                inputs = (static, probs[arm], probs["stay_cable"], probs["deck"], t)
                actual = independent_map(*inputs) if replay_source else pipeline(*inputs)["deck"]
                maps[arm+"/"+mode] = actual
        if replay_source:
            for key, value in maps.items():
                with Image.open(replay_source / role / key / name) as image:
                    if not np.array_equal(value, np.asarray(image)):
                        raise ValueError("Independent native prediction differs: "+key+"/"+name)
        else:
            for key, value in maps.items():
                path = output / role / key / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(value).save(path)
        target = labels.read(name)
        matrices = {k: independent_confusion(target, v) for k, v in maps.items()}
        group = groups.setdefault(part, {k: np.zeros((5, 5), np.int64) for k in VARIANTS})
        view_costs = {}
        for key, (old, new) in CONTRASTS.items():
            c = changes(maps[old], maps[new], target)
            cube = class_transitions(maps[old], maps[new], target)
            if int(cube.sum()) != c["changed"]:
                raise ValueError("Transition accounting differs")
            cubes[key] += cube
            view_costs[key] = c
            for n in c:
                costs[key][n] += c[n]
            # Lossless per-pixel old/new IDs; unlike a boolean mask, records the reason's class transition.
            encoded = np.where(maps[old] != maps[new], maps[old]*5+maps[new]+1, 0).astype(np.uint8)
            path = (replay_source or output) / role / "transitions" / key / name
            if replay_source:
                with Image.open(path) as image:
                    if not np.array_equal(encoded, np.asarray(image)):
                        raise ValueError("Changed-pixel evidence differs")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(encoded).save(path)
        for key in VARIANTS:
            pooled[key] += matrices[key]; group[key] += matrices[key]
        per_view[name] = {"partition": part, "metrics": {k: metrics(m) for k, m in matrices.items()}, "changes": view_costs}
        if i % 24 == 0:
            print(json.dumps({"scoring": role, "views": i+1, "total": len(names), "replay": bool(replay_source)}), flush=True)
    if replay_source and per_view != json.loads((replay_source / role / "per_view.json").read_text()):
        raise ValueError("Per-view scores/changes differ")
    (output / role).mkdir(parents=True, exist_ok=True)
    write_new_json(output / role / "per_view.json", per_view)
    return {"views": names, "metrics": {k: metrics(m) for k, m in pooled.items()}, "changes": costs,
        "transition_counts_old_new_truth": {k: v.tolist() for k, v in cubes.items()},
        "per_partition": {part: {k: metrics(m) for k, m in group.items()} for part, group in groups.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("batch", "artifact-root", "output"):
        p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--replay", type=Path, help="Read-only original comparison to independently reconstruct")
    a = p.parse_args()
    a.batch, a.output = a.batch.resolve(), a.output.resolve()
    if a.replay:
        a.replay = a.replay.resolve()
    if not a.output.is_relative_to(ROOT / "outputs/quality/mask2former"):
        p.error("Use a new isolated output")
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg = json.loads((ROOT / CONFIG).read_text())
    record = {"experiment_id": cfg["experiment_id"], "status": "initializing_mask2former_comparison",
        "seed": cfg["seed"], "git_commit": cfg["git_commit"], "config_sha256": sha256(ROOT / CONFIG),
        "code_sha256": {n: sha256(ROOT/n) for n in CODE}, "new_training": False,
        "development_labels_decoded": False, "automatic_promotion": False,
        "whole_chain_spatial_validation": False, "historically_reused_data": True,
        "replay": bool(a.replay), "transition_png_encoding": "0 unchanged; otherwise old_class*5+new_class+1"}
    try:
        validate_arms(cfg["comparison"]["arms"])
        index, proofs, contexts = accepted_index(a.batch, a.artifact_root, cfg)
        ctx = contexts[0]
        grid = json.loads((ROOT / cfg["grid_config"]).read_text())
        paired = json.loads((ROOT / cfg["pair_result"]).read_text())
        roles = partition_training_names(sorted(index), expected_count=219)
        record.update(input_result_hashes=proofs, roles=roles, candidate_bindings={c.binding["partition"]: c.binding for c in contexts})
        original = None
        if a.replay:
            original = json.loads((a.replay / "result.json").read_text())
            if (original["status"] != "completed_mask2former_comparison_pending_independent_replay"
                    or original["config_sha256"] != record["config_sha256"] or original["code_sha256"] != record["code_sha256"]
                    or original["input_result_hashes"] != proofs or original["roles"] != roles):
                raise ValueError("Replay source is incomplete or differently bound")
            verify_hashes(a.replay, original["artifacts"])
            record["replay_source_result_sha256"] = sha256(a.replay / "result.json")
        for name in (*CODE, CONFIG):
            dest = a.output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, dest)
        write_new_json(a.output / "started.json", record)
        release = LabelRelease(a.output)
        release.require("calibration")
        labels = TrainingLabels(a.artifact_root / "r2/sem_001/labels", roles["calibration"], ctx.label_manifest, cfg["resolution"])
        thresholds, curves = calibrate(index, roles["calibration"], labels, grid, replay=bool(a.replay))
        sealed = {"thresholds": thresholds, "curves": curves, "calibration_views": sorted(labels.accessed),
            "check_labels_read": False, "development_labels_read": False,
            "input_result_hashes": proofs, "config_sha256": record["config_sha256"]}
        if a.replay and sealed != json.loads((a.replay / "calibration_sealed.json").read_text()):
            raise ValueError("Independent full calibration curves/selections differ")
        write_new_json(a.output / "calibration_sealed.json", sealed)
        release.seal()
        evaluation = {}
        for role in ("calibration", "check"):
            release.require(role)
            reader = TrainingLabels(a.artifact_root / "r2/sem_001/labels", roles[role], ctx.label_manifest, cfg["resolution"])
            evaluation[role] = score(index, roles[role], reader, cfg, thresholds, a.output, role, a.replay)
            if sorted(reader.accessed) != sorted(roles[role]):
                raise ValueError("Incomplete target scoring coverage")
            assert_baseline_counts(evaluation[role]["metrics"], paired["evaluation"][role]["metrics"])
            if a.replay and evaluation[role] != original["evaluation"][role]:
                raise ValueError("Independent pooled/group counts differ")
        release.require("check")
        verify_hashes(ROOT, record["code_sha256"])
        verify_hashes(ROOT, {CONFIG: record["config_sha256"], **proofs})
        for c in contexts:
            c.finish()
        if a.replay:
            verify_hashes(a.replay, {"result.json": record["replay_source_result_sha256"], **original["artifacts"]})
        record.update(status="completed_mask2former_comparison_independent_replay" if a.replay else "completed_mask2former_comparison_pending_independent_replay",
            evaluation=evaluation, thresholds=thresholds, calibration_sealed_sha256=release.seal_hash,
            independently_reconstructed_views=219 if a.replay else 0,
            frozen_artifacts_verified=len(ctx.frozen), rgb_exports_unchanged=len(ctx.rgb_hashes),
            artifacts={str(f.relative_to(a.output)): sha256(f) for f in a.output.rglob("*") if f.is_file() and "code" not in f.relative_to(a.output).parts})
    except BaseException as exc:
        record.update(status="failed_mask2former_comparison_not_method_rejection", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic()-started
        write_new_json(a.output / "result.json", record)
        print(json.dumps({"status": record["status"], "seconds": record["seconds"], "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
