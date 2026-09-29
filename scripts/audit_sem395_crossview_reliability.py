#!/usr/bin/env python3
"""Independently audit sealed SEM-395 counts and outcome arithmetic."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_protocol import sha256, verify_hashes, write_new_json

EID = "SEM-395-RGB150-CROSSVIEW-RELIABILITY-DIAG"
CONFIG = ROOT / "experiments/runs" / EID / "config.yaml"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.int64)
    for target_class in range(5):
        for predicted_class in range(5):
            matrix[target_class, predicted_class] = np.count_nonzero(
                (truth == target_class) & (prediction == predicted_class)
            )
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="r2")
    args = parser.parse_args()
    if (
        not args.attempt.startswith("r")
        or not args.attempt[1:].isdigit()
        or int(args.attempt[1:]) < 1
    ):
        raise ValueError("Attempt must be r followed by a positive integer")

    config = read_json(CONFIG)
    run = ROOT / "outputs/quality/SEM-395" / args.attempt
    output = run / "audit"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EID,
        "attempt": args.attempt,
        "stage": "independent_audit",
        "status": "running",
        "holdout_rgb_read": 0,
        "holdout_labels_read": 0,
        "audit_source_sha256": sha256(Path(__file__)),
    }
    try:
        prepare_result = read_json(run / "prepare/result.json")
        evidence_result = read_json(run / "evidence/result.json")
        score_result = read_json(run / "score/result.json")
        for stage, record in (
            ("prepare", prepare_result),
            ("evidence", evidence_result),
            ("score", score_result),
        ):
            if record["status"] != "completed":
                raise ValueError(f"SEM-395 {stage} is incomplete")
        verify_hashes(run / "prepare", prepare_result["artifacts"])
        verify_hashes(run / "evidence", evidence_result["artifacts"])
        verify_hashes(run / "score", score_result["artifacts"])

        plan = read_json(run / "prepare/target-source-plan.json")
        access = read_json(run / "prepare/access.json")
        summary = read_json(run / "score/summary.json")
        pair_summary = read_json(run / "evidence/pair-summary.json")
        target_summary = read_json(run / "evidence/target-summary.json")
        allowed = set(access["allowed_label_names_after_evidence_seal"])
        forbidden = set(access["forbidden_holdout_names"]) | set(
            access["forbidden_unlabeled_names"]
        )
        if allowed & forbidden or set(score_result["allowed_labels_read"]) != allowed:
            raise ValueError(
                "SEM-395 score label access differs from the sealed contract"
            )

        manifest = read_json(ROOT / config["label_manifest"])
        label_hashes = {
            row["image"]: row["label_png_sha256"] for row in manifest["records"]
        }
        label_root = ROOT / config["label_root"]
        audited_label_hashes = {}
        for name in sorted(allowed):
            digest = sha256(label_root / name)
            if digest != label_hashes[name]:
                raise ValueError(f"Training label changed: {name}")
            audited_label_hashes[name] = digest

        probability_root = ROOT / config["parents"]["sem392"]["probabilities"]
        total_baseline = np.zeros((5, 5), dtype=np.int64)
        total_counterfactual = np.zeros((5, 5), dtype=np.int64)
        counts = {
            "eligible_pixels": 0,
            "changed_pixels": 0,
            "corrected_pixels": 0,
            "damaged_pixels": 0,
            "net_correct_pixels": 0,
            "sources_agree_target_gt_disagrees": 0,
        }
        foreground_pixels = 0
        reliable_foreground = 0
        decoded_targets = []
        for row in plan["targets"]:
            name = row["target"]
            truth = np.asarray(Image.open(label_root / name), dtype=np.uint8)
            decoded_targets.append(name)
            filename = Path(name).with_suffix(".npz").name
            probability_path = probability_root / filename
            if (
                sha256(probability_path)
                != prepare_result["selected_probability_hashes"][filename]
            ):
                raise ValueError(f"SEM392 probability changed: {filename}")
            with np.load(probability_path, allow_pickle=False) as archive:
                probability = np.asarray(archive["probability"], dtype=np.float32)
            baseline = np.argmax(probability, axis=0).astype(np.uint8)
            with np.load(
                run / "evidence/targets" / f"{Path(name).stem}.npz",
                allow_pickle=False,
            ) as archive:
                indices = np.asarray(archive["target_index"], dtype=np.int64)
                consensus = np.asarray(archive["consensus_class"], dtype=np.uint8)
            candidate = baseline.copy()
            candidate.ravel()[indices] = consensus
            total_baseline += confusion(truth, baseline)
            total_counterfactual += confusion(truth, candidate)
            selected_truth = truth.ravel()[indices]
            old = baseline.ravel()[indices]
            corrected = (old != selected_truth) & (consensus == selected_truth)
            damaged = (old == selected_truth) & (consensus != selected_truth)
            counts["eligible_pixels"] += int(indices.size)
            counts["changed_pixels"] += int(np.count_nonzero(old != consensus))
            counts["corrected_pixels"] += int(np.count_nonzero(corrected))
            counts["damaged_pixels"] += int(np.count_nonzero(damaged))
            counts["sources_agree_target_gt_disagrees"] += int(
                np.count_nonzero(consensus != selected_truth)
            )
            foreground_pixels += int(np.count_nonzero(truth > 0))
            reliable_foreground += int(np.count_nonzero(selected_truth > 0))
        counts["net_correct_pixels"] = (
            counts["corrected_pixels"] - counts["damaged_pixels"]
        )

        stage_counts = {
            name: sum(row["stage_counts"][name] for row in pair_summary)
            for name in evidence_result["aggregate_stage_counts"]
        }
        if stage_counts != evidence_result["aggregate_stage_counts"]:
            raise ValueError("Independent first-rejection counts disagree")
        geometry_rejects = sum(
            value
            for name, value in stage_counts.items()
            if name not in {"source_low_confidence", "reliable"}
        )
        semantic_rejects = stage_counts["source_low_confidence"] + sum(
            row["evidence_counts"]["source_semantic_disagreement"]
            for row in target_summary
        )
        target_pixels = len(plan["targets"]) * 989 * 1320
        reliable_fraction = counts["eligible_pixels"] / target_pixels
        foreground_fraction = reliable_foreground / foreground_pixels
        if counts["net_correct_pixels"] > 0:
            outcome_key = (
                "reliable_positive_net_and_coverage"
                if reliable_fraction >= 0.01 and foreground_fraction >= 0.005
                else "positive_but_too_sparse"
            )
        else:
            outcome_key = (
                "geometry_dominant_failure"
                if geometry_rejects > semantic_rejects
                else "semantic_or_annotation_dominant_failure"
            )
        outcome = config["outcome_rules"][outcome_key]

        expected_coverage = summary["coverage"]
        checks = {
            "baseline_confusion_exact": total_baseline.tolist()
            == summary["baseline"]["confusion"],
            "counterfactual_confusion_exact": total_counterfactual.tolist()
            == summary["counterfactual"]["confusion"],
            "replacement_counts_exact": counts == summary["counts"],
            "target_pixel_count_exact": target_pixels
            == expected_coverage["target_pixels"],
            "foreground_pixel_count_exact": foreground_pixels
            == expected_coverage["target_foreground_gt_pixels"],
            "reliable_foreground_count_exact": reliable_foreground
            == expected_coverage["reliable_target_foreground_gt_pixels"],
            "rejection_counts_exact": stage_counts
            == summary["rejection"]["first_rejection_counts"],
            "geometry_rejects_exact": geometry_rejects
            == summary["rejection"]["geometry_rejects"],
            "semantic_rejects_exact": semantic_rejects
            == summary["rejection"][
                "semantic_confidence_plus_two_source_disagreement_rejects"
            ],
            "outcome_exact": outcome == summary["outcome"] == score_result["outcome"],
            "score_label_hashes_exact": audited_label_hashes
            == read_json(run / "score/label-access.json")["labels"],
        }
        if not all(checks.values()):
            raise ValueError(f"Independent SEM-395 audit failed: {checks}")
        result.update(
            status="passed_independent_arithmetic_audit",
            labels_rehashed=len(audited_label_hashes),
            target_labels_decoded=len(decoded_targets),
            decoded_target_names=decoded_targets,
            counts=counts,
            geometry_rejects=geometry_rejects,
            semantic_rejects=semantic_rejects,
            outcome=outcome,
            checks=checks,
            prerequisite_result_sha256={
                stage: sha256(run / stage / "result.json")
                for stage in ("prepare", "render", "evidence", "score")
            },
        )
    except BaseException as exc:
        result.update(
            status="failed_preserved",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
