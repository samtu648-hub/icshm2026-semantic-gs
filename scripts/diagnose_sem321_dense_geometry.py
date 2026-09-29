#!/usr/bin/env python3
"""Read the sealed SEM320 pairs only; no matcher, labels or RGB decoding."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np
from run_sem250_colmap_track_deck_diag import verify_freeze
from src.semantic2d import read_json, write_json, sha256
from src.geometry_prior.camera_io import read_cameras_txt, read_images_txt
from src.semantic_dense_geometry import (normalized_grid, cycle_support, fundamental_from_w2c,
                                         symmetric_epipolar_pixels, screen_pair)

EXP = "SEM-321-ROMAV2-DENSE-GEOMETRY-DIAG"
CONFIG = ROOT / "experiments/runs" / EXP / "config.yaml"


def checked(path, digest):
    path = Path(path)
    if sha256(path) != digest: raise ValueError("Immutable input changed: " + path.name)
    return path


def quantiles(values):
    values = np.asarray(values); values = values[np.isfinite(values)]
    return {"count": int(values.size), "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, .9)) if values.size else None}


def camera_matrix(camera):
    if camera.model != "SIMPLE_RADIAL": raise ValueError("Expected registered source camera model")
    f, cx, cy, radial = camera.params
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64), np.array([radial, 0, 0, 0, 0], dtype=np.float64)


def undistort(points, intrinsics, distortion):
    return cv2.undistortPoints(points.reshape(-1, 1, 2), intrinsics, distortion, P=intrinsics).reshape(points.shape)


def diagnose(args):
    cfg = read_json(args.config)
    for key in ("parent_config", "prepare", "runtime", "replay"):
        checked(ROOT / cfg[key], cfg[key + "_sha256"])
    parent = read_json(ROOT / cfg["parent_config"])
    for key in ("dataset_manifest", "semantic_split", "oof_split"):
        checked(ROOT / parent[key], parent[key + "_sha256"])
    plan, run, replay = (read_json(ROOT / cfg[k]) for k in ("prepare", "runtime", "replay"))
    if not replay["fresh_process_exact"] or replay["prepare_sha256"] != cfg["prepare_sha256"] or run["prepare_sha256"] != cfg["prepare_sha256"]:
        raise ValueError("Engineering parent not verified")
    frozen = verify_freeze(ROOT / cfg["freeze_manifest"], args.artifact_root)
    camera_dir = args.data_root / "camera_parameters"
    cameras = read_cameras_txt(checked(camera_dir / "cameras.txt", plan["camera_intrinsics_sha256"]))
    images = {r.name: r for r in read_images_txt(checked(camera_dir / "images.txt", plan["camera_images_sha256"])).values()}
    records = {p["name"]: p for p in run["probes"]}
    snapshot = args.output_dir / "source_snapshot"; snapshot.mkdir()
    files = [args.config, Path(__file__), ROOT / "src/semantic_dense_geometry.py", ROOT / "src/geometry_prior/camera_io.py",
             ROOT / "src/semantic2d.py", ROOT / "scripts/run_sem250_colmap_track_deck_diag.py"]
    sources = {}
    for p in files:
        name = str(p.relative_to(ROOT)); sources[name] = sha256(p)
        target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
    results = []
    for i, pair in enumerate(plan["pairs"]):
        maps = []
        for direction in ("ab", "ba"):
            r = records[f"pair{i}_{direction}"]
            with np.load(checked((ROOT / cfg["runtime"]).parent / r["path"], r["sha256"]), allow_pickle=False) as d:
                maps.append((d["warp"].astype(np.float64), d["overlap"].astype(np.float64)))
        image_a, image_b = images[pair["target"]], images[pair["source"]]
        camera_a, camera_b = cameras[image_a.camera_id], cameras[image_b.camera_id]
        ka, da = camera_matrix(camera_a); kb, db = camera_matrix(camera_b)
        fundamental = fundamental_from_w2c(image_a.rotation_world_to_camera, image_a.tvec,
                                          image_b.rotation_world_to_camera, image_b.tvec, ka, kb)
        (ab, oa), (ba, ob) = maps
        filters = cfg["filters"]
        support = cycle_support(ab, oa, ba, ob, (camera_a.width, camera_a.height),
                                filters["overlap_each_direction_minimum"], filters["cycle_maximum_native_pixels"])
        grid = normalized_grid(*ab.shape[:2])
        pa = (grid + 1) * np.array([camera_a.width, camera_a.height]) / 2
        pb_raw = (ab + 1) * np.array([camera_b.width, camera_b.height]) / 2
        pb = undistort(pb_raw, kb, db)
        identity_b = undistort((grid + 1) * np.array([camera_b.width, camera_b.height]) / 2, kb, db)
        epi = symmetric_epipolar_pixels(pa, pb, fundamental)
        identity_epi = symmetric_epipolar_pixels(pa, identity_b, fundamental)
        pre = support["cycle_kept"]
        reliable = pre & (epi <= filters["epipolar_maximum_native_pixels"])
        reliable_fraction = float(reliable.mean())
        retention = float(reliable.sum() / pre.sum()) if pre.any() else None
        q, qi = quantiles(epi[pre]), quantiles(identity_epi[pre])
        checks = screen_pair(reliable_fraction, retention, q["median"], qi["median"], cfg["geometry_screen"])
        row = {"fold": pair["fold"], "target": pair["target"], "source": pair["source"],
               "native_size_wh": [camera_a.width, camera_a.height], "grid_points": int(oa.size),
               "in_bounds": int(support["valid"].sum()), "both_overlap_half": int(support["confident"].sum()),
               "cycle_kept": int(pre.sum()), "reliable_points": int(reliable.sum()),
               "reliable_grid_fraction": reliable_fraction, "epipolar_inliers_after_cycle": retention,
               "cycle_native_pixels_on_confident": quantiles(support["cycle"][support["confident"]]),
               "epipolar_native_pixels_after_cycle": q, "identity_epipolar_on_same_support": qi,
               "source_radial_k1": float(db[0]), "target_is_pinhole": True,
               "coverage_4x4": [[float(tile.mean()) for tile in np.array_split(band, 4, axis=1)] for band in np.array_split(reliable, 4, axis=0)],
               "checks": checks}
        results.append(row)
    gates = {f"fold_{r['fold']}_{k}": r["checks"][k] for r in results for k in ("coverage", "epipolar_retention")}
    gates["two_pairs_beat_identity"] = sum(r["checks"]["beats_identity_factor2"] for r in results) >= cfg["geometry_screen"]["minimum_pairs_beating_identity_median_by_factor2"]
    if verify_freeze(ROOT / cfg["freeze_manifest"], args.artifact_root) != frozen: raise ValueError("Frozen artifacts changed")
    return {"experiment_id": EXP, "status": "completed_geometry_screen_passed" if all(gates.values()) else "completed_geometry_screen_failed",
            "passed": all(gates.values()), "checks": gates, "pairs": results, "source_sha256": sources,
            "config_sha256": sha256(args.config), "parent_config_sha256": cfg["parent_config_sha256"],
            "runtime_result_sha256": cfg["runtime_sha256"], "freeze_after": frozen,
            "new_inference": 0, "rgb_decoded": 0, "labels_decoded": 0, "semantic_exports": 0,
            "claim_boundary": "Small3-pair geometry-only diagnostic. Inlier agreement is not ground-truth correspondence or semantic precision. No source labels transferred."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "data-root", "output-dir"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG); args = p.parse_args()
    args.output_dir = args.output_dir.resolve(); args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        budget = read_json(args.config)["resource_budget"]
        if shutil.disk_usage(args.output_dir).free / 2**30 < budget["minimum_free_disk_gib"]: raise RuntimeError("Disk floor")
        result = diagnose(args); result["seconds"] = time.perf_counter() - started
        if result["seconds"] > budget["maximum_seconds"]: raise RuntimeError("Diagnostic time budget")
        write_json(args.output_dir / "result.json", result)
        print({k: result[k] for k in ("status", "checks", "seconds")})
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"experiment_id": EXP, "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__": main()
