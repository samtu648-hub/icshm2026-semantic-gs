"""SEM325 check111 selection; identical source ranking and pair gates."""
from __future__ import annotations
import numpy as np
from src.semantic_source_access import rank_sources


def select_plan(folds, calibration, poses, top_k=3, maximum_angle=60.):
    if top_k != 3 or maximum_angle != 60.: raise ValueError("Fixed retrieval policy")
    plans = []
    for fold in folds:
        train, oof, guard = (set(fold[k]) for k in ("train", "oof", "guard"))
        if train & oof or train & guard or oof & guard: raise ValueError("Overlapping split")
        for target in sorted(oof & set(calibration)):
            ranking = rank_sources(target, train, poses, maximum_angle)
            sources = [r["source"] for r in ranking if r["eligible"]][:top_k]
            plans.append({"fold":fold["fold"],"target":target,"ranking":ranking,
                          "sources":sources,"candidate_budget_filled":len(sources)==top_k})
    return plans


def summarize_access(plans, pair_rows):
    expected = [(p["fold"],p["target"],s) for p in plans for s in p["sources"]]
    if [(r["fold"],r["target"],r["source"]) for r in pair_rows] != expected:
        raise ValueError("Missing/reordered candidate results")
    targets = []
    for plan in plans:
        rows = [r for r in pair_rows if (r["fold"],r["target"]) == (plan["fold"],plan["target"])]
        good = [r for r in rows if r["checks"]["coverage"] and r["checks"]["epipolar_retention"]]
        selected = good[0] if good else None
        targets.append({"fold":plan["fold"],"target":plan["target"],"candidate_count":len(rows),
            "rank1_accessible":bool(rows and rows[0] in good),"accessible_sources":[r["source"] for r in good],
            "selected_source":selected["source"] if selected else None,"selected_rank":selected["rank"] if selected else None,
            "selected_reliable_grid_fraction":selected["reliable_grid_fraction"] if selected else 0.,
            "selected_epipolar_retention":selected["epipolar_inliers_after_cycle"] if selected else None,
            "selected_beats_identity":bool(selected and selected["checks"]["beats_identity_factor2"])})
    folds = []
    for fold in sorted({p["fold"] for p in plans}):
        rows = [r for r in targets if r["fold"] == fold]
        folds.append({"fold":fold,"targets":len(rows),"accessible":sum(bool(r["selected_source"]) for r in rows),
            "rank1_accessible":sum(r["rank1_accessible"] for r in rows),
            "median_coverage_including_abstentions":float(np.median([r["selected_reliable_grid_fraction"] for r in rows]))})
    count = sum(bool(t["selected_source"]) for t in targets)
    return {"passed": True,
        "meaning":"geometry_complete_with_abstentions_not_semantic_quality",
        "target_count":len(targets),"pair_count":len(pair_rows),"rank1_accessible_targets":sum(t["rank1_accessible"] for t in targets),
        "top3_accessible_targets":count,"accessible_pairs":sum(r["checks"]["coverage"] and r["checks"]["epipolar_retention"] for r in pair_rows),
        "folds":folds,"targets":targets}
