#!/usr/bin/env python3
"""Independent CPU Torch sampler / normalized-camera epipolar recomputation.

Uses no SEM321 numeric helper, no OpenCV undistortion, labels or RGB decoding.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
import torch.nn.functional as functional
from src.semantic2d import read_json, write_json, sha256
from src.geometry_prior.camera_io import read_cameras_txt, read_images_txt
from run_sem250_colmap_track_deck_diag import verify_freeze


def checked(path, digest):
    path = Path(path)
    if sha256(path) != digest: raise ValueError("Input hash changed: " + path.name)
    return path


def recover_normalized(points, camera, distorted):
    f, cx, cy, radial = camera.params
    xd = (points - np.array([cx, cy])) / f
    x = xd.copy()
    if distorted:
        for _ in range(15):
            x = xd / (1 + radial * (x * x).sum(-1, keepdims=True))
    return np.concatenate((x, np.ones((*x.shape[:-1], 1))), axis=-1)


def statistics(values):
    return {"count": int(values.size), "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, .9)) if values.size else None}


def audit(args):
    result = read_json(args.result_dir / "result.json")
    config_path = ROOT / "experiments/runs/SEM-321-ROMAV2-DENSE-GEOMETRY-DIAG/config.yaml"
    cfg = read_json(checked(config_path, result["config_sha256"]))
    for name, digest in result["source_sha256"].items():
        checked(ROOT / name, digest); checked(args.result_dir / "source_snapshot" / name, digest)
    for key in ("prepare", "runtime", "replay", "parent_config"):
        checked(ROOT / cfg[key], cfg[key + "_sha256"])
    plan = read_json(ROOT / cfg["prepare"])
    run = read_json(ROOT / cfg["runtime"])
    replay = read_json(ROOT / cfg["replay"])
    assert replay["fresh_process_exact"] and result["labels_decoded"] == result["rgb_decoded"] == result["new_inference"] == result["semantic_exports"] == 0
    camera_dir = args.data_root / "camera_parameters"
    cameras = read_cameras_txt(checked(camera_dir / "cameras.txt", plan["camera_intrinsics_sha256"]))
    images = {c.name: c for c in read_images_txt(checked(camera_dir / "images.txt", plan["camera_images_sha256"])).values()}
    records = {r["name"]: r for r in run["probes"]}
    rows = []
    for i, (pair, expected) in enumerate(zip(plan["pairs"], result["pairs"], strict=True)):
        maps = []
        for direction in ("ab", "ba"):
            rec = records[f"pair{i}_{direction}"]
            with np.load(checked((ROOT / cfg["runtime"]).parent / rec["path"], rec["sha256"]), allow_pickle=False) as d:
                maps.append((d["warp"].astype(np.float64), d["overlap"].astype(np.float64)))
        (ab, oa), (ba, ob) = maps
        h, w = oa.shape
        sample_grid = torch.from_numpy(ab[None])
        sample = lambda a: functional.grid_sample(torch.from_numpy(a).permute(2, 0, 1)[None], sample_grid,
                            mode="bilinear", padding_mode="border", align_corners=False)[0].permute(1, 2, 0).numpy()
        back, backward_overlap = sample(ba), sample(ob[..., None])[..., 0]
        valid = (np.abs(ab[..., 0]) <= 1 - 1 / w) & (np.abs(ab[..., 1]) <= 1 - 1 / h)
        ay, ax = np.mgrid[:h, :w]
        grid = np.stack(((ax + .5) * 2 / w - 1, (ay + .5) * 2 / h - 1), -1)
        a, b = images[pair["target"]], images[pair["source"]]
        ca, cb = cameras[a.camera_id], cameras[b.camera_id]
        cycle = np.sqrt(np.sum(((back - grid) * [ca.width / 2, ca.height / 2]) ** 2, axis=-1))
        filters = cfg["filters"]
        confident = valid & (oa >= filters["overlap_each_direction_minimum"]) & (backward_overlap >= filters["overlap_each_direction_minimum"])
        kept = confident & (cycle <= filters["cycle_maximum_native_pixels"])
        ta, tb = np.eye(4), np.eye(4)
        for matrix, image in ((ta, a), (tb, b)):
            matrix[:3, :3], matrix[:3, 3] = image.rotation_world_to_camera, image.tvec
        relative = tb @ np.linalg.inv(ta)
        t = relative[:3, 3]
        cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        essential = cross @ relative[:3, :3]
        aa = recover_normalized((grid + 1) * [ca.width / 2, ca.height / 2], ca, False)

        def epipolar(source_norm):
            bb = recover_normalized((source_norm + 1) * [cb.width / 2, cb.height / 2], cb, True)
            lb, la = aa @ essential.T, bb @ essential
            residual = np.abs((bb * lb).sum(-1))
            return np.maximum(ca.params[0] * residual / np.linalg.norm(la[..., :2], axis=-1),
                              cb.params[0] * residual / np.linalg.norm(lb[..., :2], axis=-1))

        epi, identity_epi = epipolar(ab), epipolar(grid)
        reliable = kept & (epi <= filters["epipolar_maximum_native_pixels"])
        count_fields = {"in_bounds": int(valid.sum()), "both_overlap_half": int(confident.sum()),
                        "cycle_kept": int(kept.sum()), "reliable_points": int(reliable.sum())}
        if any(expected[k] != v for k, v in count_fields.items()): raise ValueError("Independent support counts differ")
        for key, values in (("cycle_native_pixels_on_confident", cycle[confident]),
                            ("epipolar_native_pixels_after_cycle", epi[kept]),
                            ("identity_epipolar_on_same_support", identity_epi[kept])):
            actual = statistics(values)
            if actual["count"] != expected[key]["count"]: raise ValueError("Quantile scope differs")
            for q in ("median", "p90"):
                if actual[q] is None:
                    if expected[key][q] is not None: raise ValueError("Empty value differs")
                elif not np.isclose(actual[q], expected[key][q], rtol=0, atol=2e-5):
                    raise ValueError("Independent geometry quantile differs")
        tiles = [[float(tile.mean()) for tile in np.array_split(band, 4, axis=1)] for band in np.array_split(reliable, 4)]
        if tiles != expected["coverage_4x4"]: raise ValueError("Independent spatial coverage differs")
        coverage = float(reliable.mean())
        retention = float(reliable.sum() / kept.sum()) if kept.any() else None
        med, control_med = statistics(epi[kept])["median"], statistics(identity_epi[kept])["median"]
        gates = cfg["geometry_screen"]
        checks = {"coverage": coverage >= gates["minimum_reliable_grid_fraction_each_pair"],
                  "epipolar_retention": retention is not None and retention >= gates["minimum_epipolar_inlier_fraction_after_cycle_each_pair"],
                  "beats_identity_factor2": med is not None and control_med is not None and control_med > 0 and med <= control_med / 2}
        if checks != expected["checks"] or coverage != expected["reliable_grid_fraction"] or retention != expected["epipolar_inliers_after_cycle"]:
            raise ValueError("Independent research screen differs")
        rows.append({"fold": pair["fold"], **count_fields, "checks": checks, "counts_and_tiles_exact": True})
    checks = {f"fold_{row['fold']}_{key}": row["checks"][key] for row in rows for key in ("coverage", "epipolar_retention")}
    checks["two_pairs_beat_identity"] = sum(row["checks"]["beats_identity_factor2"] for row in rows) >= cfg["geometry_screen"]["minimum_pairs_beating_identity_median_by_factor2"]
    if checks != result["checks"] or all(checks.values()) != result["passed"]: raise ValueError("Overall decision differs")
    if verify_freeze(ROOT / cfg["freeze_manifest"], args.artifact_root) != result["freeze_after"]: raise ValueError("Frozen artifacts changed")
    return {"experiment_id": cfg["experiment_id"], "status": "passed_independent_geometry_recomputation",
            "geometry_passed": result["passed"], "result_sha256": sha256(args.result_dir / "result.json"),
            "config_sha256": sha256(config_path), "code_sha256": sha256(Path(__file__)), "pairs": rows,
            "quantile_tolerance_native_pixels": 2e-5, "sampler": "CPU torch float64 grid_sample",
            "undistortion": "15 fixed-point radial inverse steps, not OpenCV", "epipolar": "normalized-camera essential matrix from4x4 w2c relative transform",
            "labels_decoded": 0, "rgb_decoded": 0, "new_model_inference": 0, "frozen_artifacts_verified": 18}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "data-root", "result-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); started = time.perf_counter()
    try:
        result = audit(args); result["seconds"] = time.perf_counter() - started
        write_json(args.output_dir / "result.json", result)
        print({k: result[k] for k in ("status", "geometry_passed", "seconds")})
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__": main()
