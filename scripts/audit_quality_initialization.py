#!/usr/bin/env python3
"""QUAL003: intersect independently checked RGB117 tracks with the actual PLY."""
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
from src.quality_coverage import retained_conditioning


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "data-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config_path = ROOT / "experiments/runs/QUAL-003-COVERAGE-INIT-DIAG/config.yaml"
    cfg = json.loads(config_path.read_text())
    record = {"experiment_id": cfg["experiment_id"], "stage": "actual_initialization_intersection", "status": "running",
        "seed": cfg["seed"], "config_sha256": sha256(config_path), "quality_metrics": None,
        "developer_rgb_decoded": False, "developer_labels_decoded": False, "excluded_observations_decoded": False,
        "camera_updates": False, "initialization_updates": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    code = ("scripts/audit_quality_initialization.py", "src/quality_coverage.py", "src/quality_protocol.py",
            "src/quality_pose_diagnostics.py", "scripts/prepare_rgb_colmap_scene.py", "src/quality_static_training.py")
    record["code_sha256"] = {n: sha256(ROOT / n) for n in code}
    try:
        import numpy as np
        from plyfile import PlyData
        from src.quality_pose_diagnostics import load_training_records
        from scripts.prepare_rgb_colmap_scene import parse_single_simple_radial_camera, undistort_observations
        from src.quality_static_training import TrainingLabels
        protocol = json.loads((ROOT / cfg["protocol"]).read_text())
        verify_hashes(ROOT, protocol["input_hashes"])
        scene = a.artifact_root / protocol["rgb"]["scene"]
        sparse = scene / "sparse/0"
        paths = ["preparation_manifest.json", "sparse/0/points3D.txt", "sparse/0/points3D.ply"]
        record["scene_input_hashes"] = {n: sha256(scene / n) for n in paths}
        if sha256(sparse / "points3D.ply") != cfg["initialization"]["point_ply_sha256"]:
            raise ValueError("Not the actual shared training PLY")
        rows = [line.split() for line in (sparse / "points3D.txt").read_text().splitlines() if line and not line.startswith("#")]
        ids = np.asarray([int(r[0]) for r in rows], np.int64)
        xyz = np.asarray([[float(v) for v in r[1:4]] for r in rows])
        vertex = PlyData.read(sparse / "points3D.ply")["vertex"].data
        actual_xyz = np.stack([vertex[n] for n in ("x", "y", "z")], axis=1)
        if not np.array_equal(xyz.astype(actual_xyz.dtype), actual_xyz):
            raise ValueError("Text point IDs/coordinates do not reproduce the actual PLY order")
        parent_path = ROOT / cfg["initialization"]["conditioning_parent"]
        parent = json.loads((parent_path / "result.json").read_text())
        if parent["status"] != "completed_independent_camera_evidence_audit":
            raise ValueError("RGB117 independent evidence audit missing")
        verify_hashes(parent_path, parent["artifacts"])
        run = ROOT / "outputs/quality/diagnostics/RGB-117-camera-consistency-r1"
        original = json.loads((run / "result.json").read_text())
        if sha256(run / "result.json") != parent["parent_result_sha256"]:
            raise ValueError("Conditioning audit belongs to different diagnostic parent")
        access = json.loads((run / "access.json").read_text())
        train = access["train"]
        with np.load(parent_path / "track_conditioning.npz", allow_pickle=False) as z:
            conditioning = {n: z[n].copy() for n in z.files}
        summary, retained = retained_conditioning(ids, conditioning, cfg["initialization"]["point_count"])
        # Independent Python-set check guards join order and NumPy membership.
        id_set = set(ids.tolist())
        if retained.tolist() != [int(i) in id_set for i in conditioning["point_id"]]:
            raise ValueError("Independent point-ID join disagrees")
        camera_root = a.data_root / "camera_parameters"
        verify_hashes(camera_root, original["camera_input_hashes"])
        records = load_training_records(camera_root / "images.txt", train)
        _, width, height, focal, cx, cy, radial = parse_single_simple_radial_camera(camera_root / "cameras.txt")
        k = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]])
        undistort_observations(list(records.values()), k, radial)
        split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
        static_cfg = json.loads((ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/config.yaml").read_text())
        manifest = json.loads((ROOT / static_cfg["label_manifest"]).read_text())
        labels = TrainingLabels(a.artifact_root / static_cfg["labels"], split["semantic_train_labeled"], manifest, [width, height])
        labels.audit_counts()
        large = set(conditioning["point_id"][(conditioning["error_px"] > 5) & retained].tolist())
        weak_large = set(conditioning["point_id"][(conditioning["error_px"] > 5) & (conditioning["source_max_ray_angle_deg"] < 1) & retained].tolist())
        all_observed = set()
        per_view = {}
        # Counts are sparse observed pixels, never dense bridge coverage or visibility truth.
        for n in train:
            observed = [o for o in records[n].observations if o.point_id in id_set]
            all_observed.update(o.point_id for o in observed)
            if n not in labels.allowed:
                continue
            gt = labels.read(n)
            counts = {"all_retained": np.zeros(5, np.int64), "large_retained": np.zeros(5, np.int64), "weak_large_retained": np.zeros(5, np.int64)}
            pixel_sets = {key: set() for key in counts}
            for obs in observed:
                x, y = np.rint(obs.undistorted_xy).astype(int)
                if not (0 <= x < width and 0 <= y < height):
                    continue
                pixel_sets["all_retained"].add((x, y))
                if obs.point_id in large:
                    pixel_sets["large_retained"].add((x, y))
                if obs.point_id in weak_large:
                    pixel_sets["weak_large_retained"].add((x, y))
            for key, pixels in pixel_sets.items():
                for x, y in pixels:
                    counts[key][gt[y, x]] += 1
            per_view[n] = {key: value.tolist() for key, value in counts.items()}
        if all_observed != id_set:
            raise ValueError("Actual initialization includes points without training observations")
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        verify_hashes(a.artifact_root, {i["path"]: i["sha256"] for i in frozen["artifacts"]})
        np.savez_compressed(a.output / "retained_conditioning.npz", retained=retained, **conditioning)
        write_new_json(a.output / "per_training_view_sparse_coverage.json", per_view)
        for name in (*code, str(config_path.relative_to(ROOT))):
            dest = a.output / "code" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        verify_hashes(ROOT, record["code_sha256"])
        verify_hashes(scene, record["scene_input_hashes"])
        record.update(status="completed_actual_initialization_diagnosis", summary=summary,
            input_hashes=protocol["input_hashes"], conditioning_result_sha256=sha256(parent_path / "result.json"),
            camera_input_hashes=original["camera_input_hashes"], observation_views=train,
            training_label_views=sorted(labels.accessed), training_label_hashes={n: labels.records[n]["label_png_sha256"] for n in labels.accessed},
            sparse_class_counts={key: np.sum([v[key] for v in per_view.values()], axis=0).tolist() for key in ("all_retained", "large_retained", "weak_large_retained")},
            interpretation="pixel-deduplicated observed sparse supports only; no dense or causal geometric accuracy", frozen_verified=len(frozen["artifacts"]),
            artifacts={str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file()})
    except BaseException as exc:
        record.update(status="initialization_diagnosis_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        write_new_json(a.output / "result.json", record)
        print(json.dumps({"status": record["status"], "summary": record.get("summary"), "output": str(a.output)}), flush=True)


if __name__ == "__main__":
    main()
