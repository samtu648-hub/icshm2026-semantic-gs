"""Pose-only source retrieval: no observation, image or semantic inputs."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from src.geometry_prior.camera_io import ColmapImage


def read_pose_only(path: str | Path):
    """Read COLMAP poses, consume but never parse each following observation line.

    Empty observation lines are valid. Returned records contain no point IDs or
    keypoints, so retrieval also accepts a genuinely novel pose with no tracks.
    """
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields = raw.strip().split()
            if len(fields) < 10:
                raise ValueError("Malformed pose header")
            q = np.array(fields[1:5], dtype=float)
            t = np.array(fields[5:8], dtype=float)
            if not np.isfinite(q).all() or not np.isfinite(t).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-4):
                raise ValueError("Invalid camera pose")
            record = ColmapImage(int(fields[0]), " ".join(fields[9:]), int(fields[8]), q, t, ())
            if record.name in result:
                raise ValueError("Duplicate pose name")
            result[record.name] = record
            if next(handle, None) is None:
                raise ValueError("Missing observation line")
    return result


def rank_sources(target, source_names, poses, maximum_angle=60.):
    """Rank available source poses; no target filename ordering or tracks used."""
    a = poses[target]
    rows = []
    for name in sorted(source_names):
        if name == target:
            raise ValueError("Target cannot be a source")
        b = poses[name]
        angle = math.degrees(math.acos(float(np.clip(a.optical_axis @ b.optical_axis, -1, 1))))
        distance = float(np.linalg.norm(a.center - b.center))
        rows.append({"source": name, "center_distance": distance,
                     "angle_degrees": angle, "eligible": angle <= maximum_angle})
    return sorted(rows, key=lambda r: (r["center_distance"], r["angle_degrees"], r["source"]))


def select_plan(folds, calibration, poses, top_k=3, maximum_angle=60.):
    if top_k != 3 or maximum_angle != 60.:
        raise ValueError("This experiment fixes top3 and60deg; no sweep")
    plans = []
    for fold in folds:
        train, oof, guard = (set(fold[k]) for k in ("train", "oof", "guard"))
        if train & oof or train & guard or oof & guard:
            raise ValueError("Overlapping spatial split")
        targets = sorted(oof & set(calibration))
        if not targets:
            raise ValueError("Empty calibration fold")
        target = targets[0]
        ranking = rank_sources(target, train, poses, maximum_angle)
        selected = [r for r in ranking if r["eligible"]][:top_k]
        plans.append({"fold": fold["fold"], "target": target, "ranking": ranking,
                      "sources": [r["source"] for r in selected], "candidate_budget_filled": len(selected) == top_k})
    return plans


def summarize_access(plans, pair_rows):
    targets = []
    for plan in plans:
        rows = [r for r in pair_rows if r["fold"] == plan["fold"]]
        if [r["source"] for r in rows] != plan["sources"] or any(r["target"] != plan["target"] for r in rows):
            raise ValueError("Missing/reordered candidate result")
        accessible = [r for r in rows if r["checks"]["coverage"] and r["checks"]["epipolar_retention"]]
        selected = accessible[0] if accessible else None
        targets.append({"fold": plan["fold"], "target": plan["target"],
                        "candidate_budget_filled": plan["candidate_budget_filled"],
                        "rank1_accessible": bool(rows and rows[0] in accessible),
                        "accessible_sources": [r["source"] for r in accessible],
                        "selected_source": selected["source"] if selected else None,
                        "selected_rank": selected["rank"] if selected else None,
                        "selected_beats_identity": bool(selected and selected["checks"]["beats_identity_factor2"])})
    passed = (len(targets) == 3 and all(t["candidate_budget_filled"] and t["selected_source"] for t in targets)
              and sum(t["selected_beats_identity"] for t in targets) >= 2)
    return {"passed": bool(passed), "targets": targets,
            "rank1_accessible_targets": sum(t["rank1_accessible"] for t in targets),
            "top3_accessible_targets": sum(bool(t["selected_source"]) for t in targets)}
