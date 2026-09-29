#!/usr/bin/env python3
"""Complete RGB118 verification for the scored branch, not just abstentions."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "parent-run", "parent-audit", "artifact-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / "outputs/quality"):
        raise ValueError("Use gitignored output")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    config_path = ROOT / "experiments/runs/RGB-118-CAMERA-WIDEBASE-DIAG/audit_config.yaml"
    config = json.loads(config_path.read_text())
    record = {"experiment_id": config["experiment_id"], "status": "running_scored_pair_audit", "seed": config["seed"],
              "run_result_sha256": sha256(args.run / "result.json"), "audit_config_sha256": sha256(config_path),
              "code_sha256": {"scripts/audit_quality_widebase_scored.py": sha256(Path(__file__))},
              "camera_updates": False, "rgb_decoded": False, "labels_decoded": False}
    write_new_json(args.output / "started.json", record)
    try:
        import numpy as np
        import cv2
        from src.quality_pose_diagnostics import mutual_matches, evaluate_pair, stable_number, summary
        parent = json.loads((args.parent_run / "result.json").read_text())
        run = json.loads((args.run / "result.json").read_text())
        audit = json.loads((args.parent_audit / "result.json").read_text())
        if run["status"] != "completed_widebase_diagnosis_pending_replay" or run["experiment_id"] != config["experiment_id"]:
            raise ValueError("Wrong widebase run")
        if run["parent_result_sha256"] != sha256(args.parent_run / "result.json") or audit["parent_result_sha256"] != run["parent_result_sha256"] or audit["status"] != "completed_independent_camera_evidence_audit":
            raise ValueError("Unaccepted parent feature/audit binding")
        if run["parent_audit_sha256"] != sha256(args.parent_audit / "result.json"):
            raise ValueError("Parent audit changed")
        verify_hashes(args.run, run["artifacts"])
        verify_hashes(args.parent_run, parent["artifacts"])
        verify_hashes(ROOT, run["code_sha256"])
        verify_hashes(ROOT, parent["code_sha256"])
        verify_hashes(ROOT, parent["input_hashes"])
        original = json.loads((args.parent_run / "config.yaml").read_text())
        protocol = json.loads((ROOT / original["protocol"]).read_text())
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
        verify_hashes(args.artifact_root, frozen_hashes)
        rows = json.loads((args.run / "independent_matches.json").read_text())["per_pair"]
        scored = [r for r in rows if r["status"] == "completed_correspondence_crossfit_not_pose_truth"]
        selected = [scored[i] for i in np.linspace(0, len(scored) - 1, config["scored_pair_replays"], dtype=int)]
        cv2.setNumThreads(original["execution"]["opencv_threads"])
        mc = original["independent_matching"]
        for row in selected:
            pair = row["pair"]
            stem = "-".join(Path(n).stem for n in pair)
            with np.load(args.parent_run / "features" / (Path(pair[0]).stem + ".npz")) as first, np.load(args.parent_run / "features" / (Path(pair[1]).stem + ".npz")) as second, np.load(args.run / "matches" / (stem + ".npz")) as saved:
                matches = mutual_matches(first["descriptors"], second["descriptors"], mc, stable_number(original["seed"], *pair))
                actual, arrays = evaluate_pair(first["xy"][matches[:, 0]], second["xy"][matches[:, 1]], saved["official_f"], pair, mc, original["seed"], original["resolution"])
                expected = {k:v for k,v in row.items() if k != "baseline_official_coordinate_units"}
                if actual != expected:
                    raise ValueError(f"Scored replay differs: {pair}")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, saved[key])
        # Deliberately different arithmetic from the tested batched production formula.
        def scalar_coordinates(f, a, b):
            x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
            l0, l1, l2 = (f[i, 0]*x + f[i, 1]*y + f[i, 2] for i in range(3))
            m0, m1 = (f[0, i]*u + f[1, i]*v + f[2, i] for i in range(2))
            return abs(u*l0 + v*l1 + l2) / np.sqrt(l0*l0 + l1*l1 + m0*m0 + m1*m1)
        count = 0
        for row in scored:
            stem = "-".join(Path(n).stem for n in row["pair"])
            with np.load(args.run / "matches" / (stem + ".npz")) as data:
                a, b = data["a"][data["check"]], data["b"][data["check"]]
                count += len(a)
                for key, matrix in (("official", "official_f"), ("image_f", "image_f")):
                    actual = scalar_coordinates(data[matrix], a, b)
                    np.testing.assert_allclose(actual, data[key + "_check_error"], rtol=1e-9, atol=1e-8)
                    expected = row[key + "_sampson_px"]
                    computed = summary(actual)
                    for statistic in ("mean", "median", "p90", "p99", "maximum"):
                        np.testing.assert_allclose(computed[statistic], expected[statistic], rtol=1e-9, atol=1e-8)
                    for statistic in ("count", "above_1px", "above_3px", "above_5px"):
                        if computed[statistic] != expected[statistic]:
                            raise ValueError("Independent threshold counts differ")
                    if float((actual <= mc["support_threshold_px"]).mean()) != row[key + "_support"]:
                        raise ValueError("Independent support differs")
        verify_hashes(ROOT, record["code_sha256"])
        verify_hashes(ROOT, run["code_sha256"])
        verify_hashes(args.artifact_root, frozen_hashes)
        if sha256(config_path) != record["audit_config_sha256"]:
            raise ValueError("Audit config changed")
        shutil.copy2(Path(__file__), args.output / "audit_source.py")
        shutil.copy2(config_path, args.output / "audit_config.yaml")
        record.update(status="completed_scored_widebase_independent_audit", scored_pairs=len(scored), independent_check_matches=count,
                      fresh_scored_pair_replays=[r["pair"] for r in selected], frozen_verified=len(frozen_hashes),
                      git_commit=run["git_commit"], seconds=time.monotonic()-began)
        record["artifacts"] = {str(p.relative_to(args.output)):sha256(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
        write_new_json(args.output / "result.json", record)
        print(json.dumps({k:v for k,v in record.items() if k in ("status","scored_pairs","independent_check_matches","fresh_scored_pair_replays","seconds")}), flush=True)
    except BaseException as exc:
        record.update(status="failed_scored_widebase_audit",error=f"{type(exc).__name__}: {exc}",seconds=time.monotonic()-began)
        write_new_json(args.output / "failure.json", record)
        raise


if __name__ == "__main__":
    main()
