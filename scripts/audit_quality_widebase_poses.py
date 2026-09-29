#!/usr/bin/env python3
"""RGB118: baseline-only extension of accepted RGB117 features and matching protocol."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-run", "parent-audit", "data-root", "artifact-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wide-run", type=Path, help="If provided, independently replay an already completed RGB118")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "outputs/quality"):
        raise ValueError("Require gitignored output")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    config_path = ROOT / "experiments/runs/RGB-118-CAMERA-WIDEBASE-DIAG/config.yaml"
    config = json.loads(config_path.read_text())
    code_paths = ("scripts/audit_quality_widebase_poses.py", "src/quality_pose_diagnostics.py", "src/quality_protocol.py", "scripts/prepare_rgb_colmap_scene.py")
    record = {"experiment_id": config["experiment_id"], "seed": config["seed"], "status": "running_widebase_diagnosis",
              "config_sha256": sha256(config_path), "code_sha256": {p: sha256(ROOT / p) for p in code_paths},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "parent_result_sha256": sha256(args.parent_run / "result.json"), "parent_audit_sha256": sha256(args.parent_audit / "result.json"),
              "camera_updates": False, "rgb_decoded": False, "labels_decoded": False}
    write_new_json(args.output / "started.json", record)
    try:
        import numpy as np
        import cv2
        from scripts.prepare_rgb_colmap_scene import parse_single_simple_radial_camera
        from src.quality_pose_diagnostics import (load_training_records, select_pairs, fundamental_from_poses, mutual_matches,
                                                 stable_number, evaluate_pair, summary)
        parent = json.loads((args.parent_run / "result.json").read_text())
        audit = json.loads((args.parent_audit / "result.json").read_text())
        if parent["status"] != "completed_camera_consistency_diagnosis_not_pose_truth" or audit["status"] != "completed_independent_camera_evidence_audit" or audit["parent_result_sha256"] != record["parent_result_sha256"]:
            raise ValueError("RGB117 must pass independent audit")
        verify_hashes(args.parent_run, parent["artifacts"])
        verify_hashes(ROOT, parent["code_sha256"])
        verify_hashes(ROOT, parent["input_hashes"])
        verify_hashes(args.data_root / "camera_parameters", parent["camera_input_hashes"])
        original = json.loads((args.parent_run / "config.yaml").read_text())
        if sha256(args.parent_run / "config.yaml") != parent["config_sha256"]:
            raise ValueError("Parent config mismatch")
        if config["seed"] != original["seed"]:
            raise ValueError("Baseline comparison must preserve seed")
        if shutil.disk_usage(args.output).free / 2**30 < original["execution"]["minimum_free_gib"]:
            raise ValueError("Disk reserve not met")
        protocol = json.loads((ROOT / original["protocol"]).read_text())
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
        verify_hashes(args.artifact_root, hashes)
        mc = original["independent_matching"]
        cv2.setNumThreads(original["execution"]["opencv_threads"])
        access = json.loads((args.parent_run / "access.json").read_text())
        # Metadata only: no observation decode is needed to select camera pairs.
        from scripts.prepare_rgb_colmap_scene import pose_from_fields
        records = {}
        with (args.data_root / "camera_parameters/images.txt").open() as stream:
            expecting_pose = True
            for raw in stream:
                line = raw.strip()
                if line.startswith("#"):
                    continue
                if expecting_pose:
                    if not line:
                        continue
                    camera = pose_from_fields(line.split())
                    if camera.name in access["train"]:
                        records[camera.name] = camera
                    expecting_pose = False
                else:
                    expecting_pose = True
        if set(records) != set(access["train"]):
            raise ValueError("Incomplete training pose inventory")
        _, w, h, f, cx, cy, _ = parse_single_simple_radial_camera(args.data_root / "camera_parameters/cameras.txt")
        k = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
        pairs, missing = select_pairs(records, config["neighbor_ranks"], mc["maximum_optical_axis_angle_deg"], mc["minimum_baseline"])
        parent_pairs = set(map(tuple, json.loads((args.parent_run / "pairs.json").read_text())["pairs"]))
        write_new_json(args.output / "access.json", access)
        write_new_json(args.output / "pairs.json", {"pairs": pairs, "missing": missing, "overlap_with_parent": len(set(pairs) & parent_pairs)})
        replay = None
        if args.wide_run:
            replay = json.loads((args.wide_run / "result.json").read_text())
            for key in ("config_sha256", "code_sha256", "parent_result_sha256", "parent_audit_sha256"):
                if record[key] != replay[key]:
                    raise ValueError(f"Replay binding mismatch: {key}")
            if replay["status"] != "completed_widebase_diagnosis_pending_replay":
                raise ValueError("Uncompleted widebase parent")
            verify_hashes(args.wide_run, replay["artifacts"])
            record["wide_result_sha256"] = sha256(args.wide_run / "result.json")
            pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, config["replay_pairs"], dtype=int)]
        (args.output / "matches").mkdir()
        rows, all_official, all_fitted = [], [], []
        for index, pair in enumerate(pairs):
            a, b = pair
            with np.load(args.parent_run / "features" / (Path(a).stem + ".npz")) as first, np.load(args.parent_run / "features" / (Path(b).stem + ".npz")) as second:
                matches = mutual_matches(first["descriptors"], second["descriptors"], mc, stable_number(config["seed"], *pair))
                official = fundamental_from_poses(records[a].rotation, records[a].translation, records[b].rotation, records[b].translation, k)
                row, arrays = evaluate_pair(first["xy"][matches[:, 0]], second["xy"][matches[:, 1]], official, pair, mc, config["seed"], (w, h))
            centre_a = -records[a].rotation.T @ records[a].translation
            centre_b = -records[b].rotation.T @ records[b].translation
            row["baseline_official_coordinate_units"] = float(np.linalg.norm(centre_a - centre_b))
            stem = "-".join(Path(n).stem for n in pair)
            if replay is not None:
                if row != json.loads((args.wide_run / "matches" / (stem + ".json")).read_text()):
                    raise ValueError(f"Fresh widebase summary replay differs: {pair}")
                with np.load(args.wide_run / "matches" / (stem + ".npz")) as cached:
                    if set(arrays) != set(cached.files):
                        raise ValueError("Widebase replay fields differ")
                    for key, value in arrays.items():
                        np.testing.assert_array_equal(value, cached[key])
            write_new_json(args.output / "matches" / (stem + ".json"), row)
            np.savez_compressed(args.output / "matches" / (stem + ".npz"), **arrays)
            rows.append(row)
            if "official_check_error" in arrays:
                all_official.extend(arrays["official_check_error"])
                all_fitted.extend(arrays["image_f_check_error"])
            if index % 50 == 0 or index == len(pairs) - 1:
                print(json.dumps({"stage": "widebase_replay" if replay else "widebase_pairs", "completed": index + 1, "total": len(pairs)}), flush=True)
        successful = [r for r in rows if r["status"] == "completed_correspondence_crossfit_not_pose_truth"]
        result_summary = {"pairs": len(rows), "completed_crossfits": len(successful), "insufficient_or_failed_pairs": len(rows) - len(successful),
            "possible_planar_or_low_parallax_pairs": sum(r["possible_planar_or_low_parallax"] for r in successful),
            "candidate_pairs": [r["pair"] for r in rows if r["pose_inconsistency_candidate"]],
            "pooled_all_check_matches_official_sampson_px": summary(all_official), "pooled_all_check_matches_image_f_sampson_px": summary(all_fitted)}
        write_new_json(args.output / "independent_matches.json", {"summary": result_summary, "per_pair": rows})
        verify_hashes(ROOT, record["code_sha256"])
        verify_hashes(ROOT, parent["code_sha256"])
        verify_hashes(ROOT, parent["input_hashes"])
        verify_hashes(args.data_root / "camera_parameters", parent["camera_input_hashes"])
        verify_hashes(args.artifact_root, hashes)
        if sha256(config_path) != record["config_sha256"]:
            raise ValueError("Widebase config changed")
        shutil.copy2(Path(__file__), args.output / "audit_source.py")
        shutil.copy2(config_path, args.output / "config.yaml")
        record.update(status="completed_fresh_widebase_replay" if replay else "completed_widebase_diagnosis_pending_replay",
                      summary=result_summary, frozen_verified=len(hashes), seconds=time.monotonic() - began,
                      pair_replays=[list(p) for p in pairs] if replay else [], runtime={"opencv":cv2.__version__,"numpy":np.__version__})
        record["artifacts"] = {str(p.relative_to(args.output)): sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
        write_new_json(args.output / "result.json", record)
        print(json.dumps({"status":record["status"], "seconds":record["seconds"], "summary":result_summary}), flush=True)
    except BaseException as exc:
        record.update(status="failed_widebase_camera_diagnosis",error=f"{type(exc).__name__}: {exc}",seconds=time.monotonic()-began)
        write_new_json(args.output / "failure.json", record)
        raise


if __name__ == "__main__":
    main()
