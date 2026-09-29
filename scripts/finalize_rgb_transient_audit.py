#!/usr/bin/env python3
"""Join explicit native-image review with frozen numerical and integrity gates."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.semantic2d import sha256


def visual_gate(review, expected_names):
    rows = review["views"]
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)) or set(names) != set(expected_names):
        raise ValueError("Visual review must cover each fixed view exactly once")
    allowed = {"improved", "unchanged_or_uncertain", "worse"}
    if any(r["candidate_vs_control"] not in allowed or
           r["candidate_vs_historical"] not in allowed or
           not isinstance(r["structure_erosion"], bool) or not r["observations"].strip()
           for r in rows):
        raise ValueError("Explicit paired judgment and observations required")
    improved = sum(r["candidate_vs_control"] == "improved" for r in rows)
    no_harm = all(r["candidate_vs_control"] != "worse" and
                  r["candidate_vs_historical"] != "worse" and
                  not r["structure_erosion"] for r in rows)
    return {"improved_views_vs_control": improved, "no_obvious_harm": no_harm,
            "passes": improved >= 2 and no_harm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    output = root / "r2/rgb_092/final_audit/full-20260906-r1"
    audit_path = output / "result.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "numeric_audit_complete_visual_review_pending":
        raise ValueError("Only an unfinalized audit can be finalized")
    review = json.loads(args.review.read_text())
    cfg = json.loads((REPO / "experiments/runs/RGB-092-TRANSIENT/config.yaml").read_text())
    visual = visual_gate(review, cfg["evaluation"]["visual_views"])
    for row in review["views"]:
        panel = output / (Path(row["name"]).stem + "_native_comparison.png")
        if row["panel_sha256"] != sha256(panel):
            raise ValueError("Reviewed panel changed")
    integrity = {}
    resource_usage = {}
    samples = [json.loads(line) for line in
               (root / "r2/rgb_092/queue-r2/gpu_samples.jsonl").read_text().splitlines()]
    for ident in ["RGB-091-CTRL", "RGB-092-TRANSIENT"]:
        run = root / "r1/runs" / ident
        execution = json.loads((run / "execution.json").read_text())
        reload = json.loads((run / "reload.json").read_text())
        active_samples = [s for s in samples if s["returncode"] == 0 and
                          execution["started_unix"] <= s["time"] <= execution["ended_unix"]]
        resource_usage[ident] = {
            "wrapper_wall_seconds_including_reload_and_evaluation": execution["elapsed_seconds"],
            "peak_sampled_device_memory_MiB": max(int(s["gpu"].split(",")[0]) for s in active_samples),
            "sampling_seconds": 30,
            "sample_count": len(active_samples),
            "final_anchor_count": reload["model_parameter_count"]}
        integrity[ident] = (execution["status"] == "completed" and
                            execution["frozen_parent_unchanged"] and
                            reload["passes"] and reload["checkpoint_iteration"] == 30000 and
                            sha256(run / "chkpnt30000.pth") == execution["checkpoint_sha256"] and
                            all(sha256(root / name) == digest for name, digest in execution["frozen_parent_hashes"].items()))
        integrity[ident + "_initialization_and_cameras"] = all(
            sha256(run / name) == sha256(root / "r1/runs/RGB-030-FULL" / name)
            for name in ["input.ply", "cameras.json"])
        if ident == "RGB-092-TRANSIENT":
            integrity["uncertainty_reload"] = execution["uncertainty_reload"]["passed"]
    mask_path = root / "r2/rgb_092/mask_audit/full-20260906-r1/result.json"
    mask = json.loads(mask_path.read_text())
    integrity["training_protection"] = (mask["training_views"] == 219 and mask["validation_views"] == 0 and
                                         all(row["protected_violations"] == 0 for row in mask["per_view"]))
    integrity["mask_checkpoint"] = mask["checkpoint_sha256"] == execution["uncertainty_checkpoint_sha256"]
    video_path = root / "r2/rgb_092/flythrough/full-20260906-r1/result.json"
    video = json.loads(video_path.read_text())
    integrity["fixed_trajectory_video"] = (video["status"] == "completed" and video["frames"] == 144 and
                                            video["coordinate_frame"] == "official_COLMAP" and
                                            sha256(video_path.parent / video["video"]) == video["video_sha256"])
    allowed = audit["numeric_gate_passes"] and visual["passes"] and all(integrity.values())
    audit.update(status="completed_candidate_RGB_gate_passed" if allowed else "completed_candidate_not_promoted",
                 visual_gate_passes=visual["passes"], visual_review=review, visual_gate=visual,
                 visual_review_sha256=sha256(args.review), integrity_checks=integrity, SEM180_allowed=allowed,
                 resource_usage=resource_usage, mask_audit_sha256=sha256(mask_path),
                 flythrough_result_sha256=sha256(video_path))
    # Preserve the pre-review numerical record, including its hash.
    frozen_numeric = output / "numeric_result_before_visual_review.json"
    if frozen_numeric.exists():
        raise FileExistsError(frozen_numeric)
    frozen_numeric.write_bytes(audit_path.read_bytes())
    audit["numeric_record_sha256"] = sha256(frozen_numeric)
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps({"status": audit["status"], "visual": visual,
                      "integrity": integrity, "SEM180_allowed": allowed}, indent=2))


if __name__ == "__main__":
    main()
