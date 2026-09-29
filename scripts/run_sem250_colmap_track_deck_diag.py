#!/usr/bin/env python3
"""Test whether independent COLMAP tracks cover SEM-161 deck omissions.

This is deliberately a training-side feasibility diagnostic, not another
semantic candidate.  It builds a source-only Beta posterior for each released
COLMAP track, using labels from a fold's spatially separated train cameras,
then places the posterior's untuned Bayes-majority evidence at observations in
that fold's OOF cameras.  Target labels are consulted only after every target
evidence map is built, for reporting coverage and an oracle upper bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.geometry_prior.camera_io import read_images_txt
from src.semantic_colmap_track_evidence import (
    binary_source_posteriors,
    deck_confusion,
    exact_oof_source_contract,
    observation_index,
    oracle_deck_iou_gain,
    target_majority_map,
)
from src.semantic_protected_fusion import partition_training_names


EXPERIMENT_ID = "SEM-250-COLMAP-TRACK-DECK-DIAG"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def label_array(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path), dtype=np.uint8)
    require(image.ndim == 2 and image.max(initial=0) <= 4, f"invalid official label map: {path.name}")
    return image


def read_point_ids(path: Path) -> set[int]:
    point_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                point_ids.add(int(line.split(maxsplit=1)[0]))
    require(point_ids, "prepared scene has no re-triangulated points")
    return point_ids


def verify_freeze(freeze_manifest: Path, artifact_root: Path) -> dict[str, object]:
    manifest = read_json(freeze_manifest)
    verified = []
    for artifact in manifest["artifacts"]:
        path = artifact_root / artifact["path"]
        require(path.is_file(), f"frozen artifact missing: {artifact['role']}")
        actual = sha256(path)
        require(actual == artifact["sha256"], f"frozen artifact changed: {artifact['role']}")
        verified.append({"role": artifact["role"], "sha256": actual})
    return {"manifest_sha256": sha256(freeze_manifest), "count": len(verified), "artifacts": verified}


def aggregate_confusion(items: list[dict[str, int | float]]) -> dict[str, int | float]:
    tp, fp, fn = (sum(int(item[key]) for item in items) for key in ("tp", "fp", "fn"))
    denominator = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": float(tp / denominator) if denominator else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--sem160-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    require(not output.exists(), "SEM-250 requires a fresh output directory")
    output.mkdir(parents=True)
    config_path = ROOT / "experiments/runs/SEM-250-COLMAP-TRACK-DECK-DIAG/config.yaml"
    config = read_json(config_path)
    require(config["experiment_id"] == EXPERIMENT_ID, "wrong SEM-250 config")
    require(config["status"] == "registered_not_run", "SEM-250 config is not runnable")
    split_path = ROOT / config["data"]["semantic_split"]
    oof_path = ROOT / config["data"]["oof_split"]
    require(sha256(split_path) == config["data"]["semantic_split_sha256"], "semantic split changed")
    require(sha256(oof_path) == config["data"]["oof_split_sha256"], "OOF split changed")
    freeze = verify_freeze(ROOT / config["freeze_manifest"], args.artifact_root)
    require(freeze["count"] == 18, "unexpected frozen artifact count")

    split, oof = read_json(split_path), read_json(oof_path)
    names = sorted(split["semantic_train_labeled"])
    require(len(names) == 219 and len(set(names)) == 219, "SEM-250 requires exactly 219 training views")
    images = {record.name: record for record in read_images_txt(args.scene_dir / "sparse" / "0" / "images.txt").values()}
    require(set(names) == set(images), "prepared scene must contain exactly the authorised 219 training views")
    require(sha256(args.scene_dir / "sparse" / "0" / "images.txt") == config["inputs"]["scene_images_sha256"], "prepared scene images changed")
    require(sha256(args.scene_dir / "sparse" / "0" / "points3D.txt") == config["inputs"]["scene_points_sha256"], "prepared scene points changed")
    point_ids = read_point_ids(args.scene_dir / "sparse" / "0" / "points3D.txt")
    observations = observation_index(images, names, width=1320, height=989, permitted_track_ids=point_ids)
    require(observations, "no released observations point to re-triangulated tracks")

    manifest_path = args.sem160_root / "prediction_manifest.json"
    require(sha256(manifest_path) == config["inputs"]["sem160_prediction_manifest_sha256"], "SEM-160 manifest changed")
    parent = read_json(manifest_path)
    require(parent["experiment_id"] == "SEM-160-DECK-BINARY-OOF", "wrong parent prediction manifest")
    parent_paths = {record["name"]: args.sem160_root / record["prediction"] for record in parent["predictions"]}
    require(set(parent_paths) == set(names), "SEM-160 maps do not cover authorised training views")
    for name, path in parent_paths.items():
        require(sha256(path) == next(record["prediction_sha256"] for record in parent["predictions"] if record["name"] == name), f"SEM-160 prediction changed: {name}")

    started = time.perf_counter()
    access_audit: list[dict[str, object]] = []
    all_baselines: list[dict[str, int | float]] = []
    covered_false_negatives = 0
    proposed_additions = 0
    proposed_true_deck = 0
    per_view: list[dict[str, object]] = []
    all_track_support: list[int] = []
    for fold in sorted(oof["folds"], key=lambda value: int(value["fold"])):
        fold_id = int(fold["fold"])
        train, targets, guard = set(fold["train"]), sorted(fold["oof"]), set(fold["guard"])
        require(train.isdisjoint(targets) and train.isdisjoint(guard) and set(targets).isdisjoint(guard), f"fold {fold_id} leakage")
        require(train | set(targets) | guard == set(names), f"fold {fold_id} is incomplete")
        source_cache: dict[str, np.ndarray] = {}
        source_accesses: set[str] = set()

        def source_label(name: str) -> np.ndarray:
            require(name in train, f"fold {fold_id} attempted an OOF/guard source label")
            source_accesses.add(name)
            if name not in source_cache:
                source_cache[name] = label_array(args.labels_root / name)
            return source_cache[name]

        posterior, support = binary_source_posteriors(
            observations,
            train,
            source_label,
            minimum_source_observations=int(config["evidence"]["minimum_source_observations"]),
        )
        all_track_support.extend(support.values())
        require(source_accesses.issubset(train), f"fold {fold_id} source-label isolation failed")
        for name in targets:
            exact_oof_source_contract(name, train)
            evidence = target_majority_map((989, 1320), observations, posterior, target_name=name)
            # The target is loaded only after its source-only evidence map has
            # been created and cannot flow back into the posterior.
            target = label_array(args.labels_root / name)
            baseline_map = label_array(parent_paths[name])
            baseline = deck_confusion(baseline_map, target)
            eligible = evidence & (baseline_map == 0)
            true_additions = eligible & (target == 1)
            false_additions = eligible & (target != 1)
            covered = int(true_additions.sum())
            all_baselines.append(baseline)
            covered_false_negatives += covered
            proposed_additions += int(eligible.sum())
            proposed_true_deck += covered
            per_view.append(
                {
                    "fold": fold_id,
                    "name": name,
                    "track_majority_pixels": int(evidence.sum()),
                    "eligible_background_pixels": int(eligible.sum()),
                    "covered_deck_false_negatives": covered,
                    "false_positive_additions": int(false_additions.sum()),
                    "baseline_deck": baseline,
                    "oracle_deck_iou_gain": oracle_deck_iou_gain(baseline, covered),
                }
            )
        access_audit.append(
            {
                "fold": fold_id,
                "source_views": len(train),
                "oof_views": len(targets),
                "guard_views": len(guard),
                "source_label_reads": len(source_accesses),
                "target_label_read_only_after_evidence": True,
                "source_target_disjoint": True,
                "posterior_tracks": len(posterior),
            }
        )

    baseline = aggregate_confusion(all_baselines)
    coverage_fraction = float(covered_false_negatives / int(baseline["fn"])) if int(baseline["fn"]) else 0.0
    evidence_precision = float(proposed_true_deck / proposed_additions) if proposed_additions else 1.0
    oracle_gain = oracle_deck_iou_gain(baseline, covered_false_negatives)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_preflight_qualified" if oracle_gain >= config["gates"]["minimum_oracle_deck_iou_gain"] and evidence_precision >= config["gates"]["minimum_evidence_deck_precision"] else "completed_preflight_not_eligible",
        "scope": "training-side diagnostic only; no semantic prediction or checkpoint was written",
        "freeze": freeze,
        "inputs": {
            "sem160_prediction_manifest_sha256": sha256(manifest_path),
            "scene_images_sha256": sha256(args.scene_dir / "sparse" / "0" / "images.txt"),
            "scene_points_sha256": sha256(args.scene_dir / "sparse" / "0" / "points3D.txt"),
            "retriangulated_track_count": len(point_ids),
            "track_ids_with_authorised_observations": len(observations),
        },
        "source_isolation": access_audit,
        "baseline_deck": baseline,
        "evidence": {
            "minimum_source_observations": config["evidence"]["minimum_source_observations"],
            "bayes_majority_decision": "posterior_deck > 0.5; fixed and not tuned",
            "median_source_observations_per_supported_track": int(np.median(all_track_support)) if all_track_support else 0,
            "proposed_background_to_deck_pixels": proposed_additions,
            "true_deck_false_negatives_covered": covered_false_negatives,
            "coverage_of_baseline_deck_false_negatives": coverage_fraction,
            "evidence_deck_precision": evidence_precision,
            "oracle_deck_iou_gain_without_false_positives": oracle_gain,
        },
        "engineering": {
            "no_scaffold_model_or_renderer_invoked": True,
            "frozen_rgb_artifacts_rehashed": True,
            "rgb_output_unchanged_by_construction": True,
            # All 219 OOF targets above are evaluated. Source-only fitting is
            # not calibration/check isolation; derive access from actual rows.
            "check_metrics_opened": bool(set(row["name"] for row in per_view) & set(partition_training_names(names, expected_count=219)["check"])),
            "target_labels_evaluated": [row["name"] for row in per_view],
            "developer_metrics_opened": False,
            "external_camera_candidate": "not created: sparse feasibility did not write a deployable semantic branch",
            "seconds": time.perf_counter() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "per_view": per_view,
    }
    write_json(output / "result.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_view"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
