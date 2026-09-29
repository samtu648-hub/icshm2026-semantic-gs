#!/usr/bin/env python3
"""Deterministic matched attribution fitting and sealed spatial OOF audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_sem230_offset_semantic_oof as common
from src.scaffold_semantic_sidecar import frozen_generated_geometry, _render_color_channels
from src.scaffold_offset_semantic import active_offset_indices
from src.scaffold_visibility_supervision import load_visibility_extension, rasterize_frozen_geometry_buffers
from src.offset_responsibility import contributor_evidence, dominant_evidence, posterior, render_table
from src.camera_adapter import redistort_semantic_probabilities


def diagnostics(geometry, view, extension):
    b = rasterize_frozen_geometry_buffers(view, geometry)
    return extension.diagnostic_stats(b.ranges, b.point_list, b.means2d, b.conic_opacity, int(view.image_height), int(view.image_width))


def panels(names, labels, parents, candidates, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = common.label_array(labels / name)
        old, new = [common.png_map(paths[name], context="SEM240 panel") for paths in (parents, candidates)]
        change = np.zeros((*old.shape, 3), np.uint8)
        changed = old != new
        change[changed & (new == target)] = (48, 220, 96)
        change[changed & (new != target)] = (255, 48, 48)
        columns = [common.semantic_rgb(value) for value in (target, old, new)] + [change]
        Image.fromarray(np.concatenate(columns, 1)).save(destination / name)
        ys, xs = np.where((target == 1) | (old == 1) | (new == 1))
        if len(xs):
            y0, y1 = max(0, int(ys.min())-24), min(old.shape[0], int(ys.max())+25)
            x0, x1 = max(0, int(xs.min())-24), min(old.shape[1], int(xs.max())+25)
            zoom = np.concatenate([col[y0:y1, x0:x1] for col in columns], 1)
            Image.fromarray(zoom).save(destination / (Path(name).stem+"_deck_zoom.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "data-root", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    out, artifacts = args.output_dir, args.artifact_root
    if out.exists():
        raise FileExistsError("New run directory required")
    out.mkdir(parents=True)
    cfgpath = ROOT / "experiments/runs/SEM-240-OFFSET-RESPONSIBILITY-OOF/config.yaml"
    config = common.read_json(cfgpath)
    parentcfg = common.read_json(ROOT / "experiments/runs/SEM-230-OFFSET-SEMANTIC-OOF/config.yaml")
    splitpath = ROOT / parentcfg["data"]["semantic_split"]
    oofpath = ROOT / parentcfg["data"]["oof_split"]
    for path, expected in ((splitpath, parentcfg["data"]["semantic_split_sha256"]), (oofpath, parentcfg["data"]["oof_split_sha256"])):
        common.require(common.sha256(path) == expected, "Split changed")
    names = common.select_rendered_training_views(common.read_json(splitpath), expected_count=219)
    split = common.read_json(oofpath)
    partition = common.partition_training_names(names, expected_count=219)
    labelroot = artifacts / "r2/sem_001/labels"
    roots = (artifacts / "r2/sem_110/sem001_training_predictions/full-20260902-r1", artifacts / "r2/sem_150/foundation_gate_oof/full-20260904-r1", artifacts / "r2/sem_160/deck_binary_oof_summary/full-20260904-r1")
    maps = []
    for root, manifest, digestkey, ident in zip(roots, ("training_sem001_render_manifest.json", "prediction_manifest.json", "prediction_manifest.json"), ("sem001_training_manifest_sha256", "sem150_prediction_manifest_sha256", "sem160_prediction_manifest_sha256"), ("SEM-110-PROTECTED-FUSION-OOF", "SEM-150-FOUNDATION-GATE-OOF", "SEM-160-DECK-BINARY-OOF")):
        maps.append(common.parent_maps(root, manifest_name=manifest, expected_sha256=parentcfg["inputs"][digestkey], expected_experiment=ident, names=names))
    rgbroot = artifacts / "r1/runs/RGB-030-FULL"
    rgbhash = common.frozen_rgb_hashes(rgbroot, 30000)
    common.require(rgbhash == parentcfg["frozen_rgb"]["artifact_sha256"], "Frozen RGB changed")
    model = common.load_frozen_scaffold(artifacts / "r1/third_party/scaffold-gs", rgbroot / "point_cloud/iteration_30000/point_cloud.ply", common.ScaffoldSettings(train_camera_count=303, n_offsets=10))
    pipe = common.pipeline()
    extension = load_visibility_extension(out / "extension")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # Synthetic probes only: no labels needed for the actual adjoint gate.
    _, camera, view = common.sem001.official_view(args.data_root, "001.png", 0)
    rgb_before = common.render_frozen_rgb(view, model, pipe)
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    probe = torch.arange(view.image_width, device="cuda").expand(view.image_height, -1).remainder(2)
    evidence = contributor_evidence(geometry, view, probe)
    fg, maximum, _, dominant = diagnostics(geometry, view, extension)
    native_fg = _render_color_channels(geometry, torch.ones((len(geometry[0]), 3), device="cuda"), view, torch.zeros(3, device="cuda"))[0]
    fgdiff = float((fg-native_fg).abs().max())
    testcolors = torch.rand((len(geometry[0]), 3), device="cuda")
    probes = torch.stack((probe == 0, probe == 1, torch.ones_like(probe, dtype=torch.bool))).float()
    forward = (_render_color_channels(geometry, testcolors, view, torch.zeros(3, device="cuda"))*probes).sum()
    adjoint = (evidence*testcolors).sum()
    relative_error = float((forward-adjoint).abs()/forward.abs().clamp_min(1))
    winner = dominant_evidence(dominant, maximum, probe, len(geometry[0]))
    engine = {"foreground_max_difference": fgdiff, "adjoint_relative_error": relative_error, "dominant_sum_error": float(abs(winner[:,2].sum()-maximum.sum())/maximum.sum().clamp_min(1)), "rgb_before_sha256": common.png_sha256(common.tensor_rgb(rgb_before), out / "rgb_before.png")}
    common.write_json(out / "engineering_preflight.json", engine)
    common.require(fgdiff <= config["engineering"]["foreground_max_difference"] and relative_error <= config["engineering"]["adjoint_relative_error_max"] and engine["dominant_sum_error"] < 1e-4, "Exact contribution engineering gate failed")
    predictions = {arm: {} for arm in ("control", "candidate")}
    fold_records = []
    for fold_id in range(3):
        fold = common.fold_for_index(split, fold_id)
        train, oof = sorted(fold["train"]), sorted(fold["oof"])
        common.require(set(train).isdisjoint(oof) and set(train).isdisjoint(fold["guard"]) and set(train).issubset(names), "Fold leakage")
        counts = {arm: torch.zeros((len(model.get_anchor), 10, 3), device="cuda") for arm in predictions}
        support = {arm: torch.zeros((len(model.get_anchor), 10), dtype=torch.int32, device="cuda") for arm in predictions}
        records = []
        foldstart = time.perf_counter()
        for index, name in enumerate(train):
            target = torch.from_numpy(common.label_array(labelroot / name).astype(np.int64)).cuda()
            _, _, trainview = common.sem001.official_view(args.data_root, name, 0)
            geom, vis = frozen_generated_geometry(trainview, model, pipe)
            anchors, offsets = active_offset_indices(vis, geom[4], n_offsets=10)
            direct = contributor_evidence(geom, trainview, target)
            foreground, maximum, _, dominant = diagnostics(geom, trainview, extension)
            unique = dominant_evidence(dominant, maximum, target, len(geom[0]))
            for arm, values in (("control", direct), ("candidate", unique)):
                counts[arm][anchors, offsets] += values
                support[arm][anchors, offsets] += (values[:, :2].sum(1) > 0).int()
            records.append({"name": name, "label_sha256": common.sha256(labelroot/name), "all_weight": float(direct[:,2].sum()), "winner_weight": float(unique[:,2].sum()), "active_offsets": len(anchors)})
            if index % 10 == 0:
                print(json.dumps({"fold": fold_id, "fitted": index+1, "total": len(train)}), flush=True)
            if time.perf_counter()-start > config["budget"]["max_gpu_hours"]*3600:
                raise RuntimeError("Registered wall budget exceeded")
        fitting_seconds = time.perf_counter()-foldstart
        metadata = {"experiment_id": config["experiment_id"], "fold": fold_id, "train": train, "oof": oof, "guard": fold["guard"], "config_sha256": common.sha256(cfgpath), "oof_split_sha256": common.sha256(oofpath), "rgb_hashes": rgbhash, "seed": 0}
        tables = {}
        for arm in counts:
            checkpoint = out / f"fold{fold_id}_{arm}.pth"
            table = posterior(counts[arm])
            torch.save({"metadata": metadata, "arm": arm, "probability": table.cpu(), "evidence": counts[arm].cpu(), "support_views": support[arm].cpu()}, checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            common.require(payload["metadata"] == metadata and torch.equal(payload["probability"], table.cpu()), "Checkpoint mismatch")
            tables[arm] = payload["probability"].cuda()
        for name in oof:
            _, _, testview = common.sem001.official_view(args.data_root, name, 0)
            geom, vis = frozen_generated_geometry(testview, model, pipe)
            old = common.png_map(maps[2][name], context="SEM240 baseline")
            mutable, _ = common.semantic230_eligibility(base=old, sem001_map=common.png_map(maps[0][name], context="base"), sem150_map=common.png_map(maps[1][name], context="foundation"))
            for arm, table in tables.items():
                with torch.no_grad():
                    pred = render_table(geom, vis, table, testview).argmax(0).cpu().numpy().astype(np.uint8)
                fused = common.fuse_binary_lane(old, pred, mutable)
                path = out / arm / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(fused).save(path)
                predictions[arm][name] = path
        # Exercise the learned table, not the old sidecar, through official redistortion.
        _, extcam, extview = common.sem001.official_view(args.data_root, "140.png", 0)
        geom, vis = frozen_generated_geometry(extview, model, pipe)
        with torch.no_grad():
            ext = render_table(geom, vis, tables["candidate"], extview).cpu().numpy()
        official = redistort_semantic_probabilities(np.concatenate((ext, np.zeros((3,*ext.shape[1:]), np.float32))), extcam)
        common.require(official.shape == (5, extcam.height, extcam.width) and np.isfinite(official).all(), "External learned branch invalid")
        Image.fromarray(official.argmax(0).astype(np.uint8)).save(out / f"fold{fold_id}_external_binary.png")
        common.write_json(out / f"fold{fold_id}_contributions.json", records)
        fold_records.append({**metadata, "fitting_seconds": fitting_seconds, "total_seconds": time.perf_counter()-foldstart, "checkpoints": {arm: common.sha256(out/f"fold{fold_id}_{arm}.pth") for arm in tables}, "reload_exact": True, "external_learned_branch_passed": True})
        common.write_json(out / "fold_results.json", fold_records)
    for arm in predictions:
        common.require(set(predictions[arm]) == set(names), "Incomplete OOF")
    rgb_after = common.render_frozen_rgb(view, model, pipe)
    rgbexact = bool(torch.equal(rgb_before, rgb_after)) and common.frozen_rgb_hashes(rgbroot, 30000) == rgbhash and not common.frozen_rgb_trainable_parameters(model)
    engine["rgb_tensor_exact_after_fitting"] = rgbexact
    common.write_json(out / "engineering_final.json", engine)
    manifest = {"experiment_id": config["experiment_id"], "status": "sealed_before_calibration", "config_sha256": common.sha256(cfgpath), "code_hashes": {name: common.sha256(ROOT/name) for name in ("scripts/run_sem240_offset_responsibility.py", "src/offset_responsibility.py", "src/scaffold_visibility_supervision.py")}, "arms": {arm: [{"name": name, "path": str(paths[name].relative_to(out)), "sha256": common.sha256(paths[name])} for name in names] for arm, paths in predictions.items()}}
    common.write_json(out / "prediction_manifest.json", manifest)
    results = {}
    for arm in predictions:
        evaluation = common.evaluate_names(names=partition["calibration"], labels_root=labelroot, baseline_paths=maps[2], candidate_paths=predictions[arm], sem001_paths=maps[0], sem150_paths=maps[1])
        results[arm] = {"evaluation": evaluation, "gate": common.gate(evaluation, config["gates"], rgb_unchanged=rgbexact)}
    common.write_json(out / "calibration_result.json", {"prediction_manifest_sha256": common.sha256(out / "prediction_manifest.json"), "arms": results})
    for arm in predictions:
        panels([n for n in config["panels"] if n in partition["calibration"]], labelroot, maps[2], predictions[arm], out / f"{arm}_panels")
    passed = results["candidate"]["gate"]["passed"]
    result = {"experiment_id": config["experiment_id"], "status": "calibration_passed_pending_check" if passed else "completed_calibration_gate_failed", "calibration": {arm: record["gate"] for arm,record in results.items()}, "fold_results": fold_records, "seconds": time.perf_counter()-start, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20, "rgb_exact": rgbexact, "check_metrics_opened": False, "developer_metrics_opened": False, "final_fitted": False, "calibration_sha256": common.sha256(out/"calibration_result.json"), "prediction_manifest_sha256": common.sha256(out/"prediction_manifest.json")}
    common.write_json(out / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "fold_results"}), flush=True)


if __name__ == "__main__":
    main()
