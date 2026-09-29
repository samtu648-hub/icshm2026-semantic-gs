#!/usr/bin/env python3
"""Independently recount SEM414/415 predictions and summarize both v2 seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_sem414_sem415_spatial_seed_pair import guard, require_seal
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.spatial_seed_replication import (
    CLASSES,
    confusion_metrics,
    direction_gates,
    flatten,
)


def read(path):
    return json.loads(Path(path).read_text())


def compare_metrics(actual, saved):
    for key in ("mIoU_with_background", "mIoU_without_background"):
        a, s = actual[key], saved[key]
        if (a is None) != (s is None) or (a is not None and abs(a - s) > 1e-12):
            raise ValueError(f"Independent {key} differs")
    for name in CLASSES:
        for key in ("tp", "fp", "fn", "IoU", "precision", "recall"):
            a, s = actual["per_class"][name][key], saved["per_class"][name][key]
            if (a is None) != (s is None) or (a is not None and abs(a - s) > 1e-12):
                raise ValueError(f"Independent class metric differs: {name} {key}")


def inner_boundary(mask):
    """Explicit four-neighbor erosion, independently of scipy binary_erosion."""
    import numpy as np

    pad = np.pad(mask, 1, constant_values=False)
    interior = mask & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return mask & ~interior


def recount(cfg, split, saved):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    output = ROOT / cfg["output_path"]
    fold = split["primary_fold"]
    train = set(fold["semantic_train_labeled"])
    holdout = set(fold["semantic_validation_labeled"])
    allowed = train | holdout
    groups = {n: g["group_id"] for g in split["groups"] for n in g["images"]}
    label_manifest = read(ROOT / cfg["label_manifest"])
    labels = {r["image"]: r["label_png_sha256"] for r in label_manifest["records"]}
    verify_hashes(ROOT / cfg["label_root"], {n: labels[n] for n in allowed})
    predictions = output / "inference/predictions"
    expected_rgb = set(fold["train_rgb_400"]) | holdout
    if set(saved["prediction_hashes"]) != expected_rgb or len(expected_rgb) != 379:
        raise ValueError("Prediction camera universe changed")
    verify_hashes(predictions, saved["prediction_hashes"])
    views = {r["name"]: r for r in saved["per_view"]}
    if set(views) != allowed:
        raise ValueError("Scored label universe changed")
    totals = {k: np.zeros((5, 5), dtype=np.int64) for k in saved["metrics"]}
    boundaries = {k: np.zeros(4, dtype=np.int64) for k in totals}
    for index, name in enumerate(sorted(allowed)):
        target = np.asarray(Image.open(ROOT / cfg["label_root"] / name))
        prediction = np.asarray(Image.open(predictions / name))
        if (
            target.shape != (989, 1320)
            or target.shape != prediction.shape
            or target.dtype != np.uint8
            or prediction.dtype != np.uint8
        ):
            raise ValueError("Native class-ID grid changed")
        if target.max() > 4 or prediction.max() > 4:
            raise ValueError("Invalid class ID")
        matrix = np.zeros((5, 5), dtype=np.int64)
        np.add.at(matrix, (target.ravel(), prediction.ravel()), 1)
        if matrix.tolist() != views[name]["confusion"]:
            raise ValueError(f"Saved per-view confusion differs: {name}")
        compare_metrics(confusion_metrics(matrix.tolist()), views[name])
        role = "train" if name in train else "holdout"
        if views[name]["role"] != role or views[name]["group"] != groups.get(name):
            raise ValueError("Per-view role or region differs")
        gt_edge, pred_edge = (
            inner_boundary(target == 2),
            inner_boundary(prediction == 2),
        )
        counts = np.asarray(
            [
                np.count_nonzero(
                    pred_edge & (np.asarray(distance_transform_edt(~gt_edge)) <= 2)
                ),
                np.count_nonzero(pred_edge),
                np.count_nonzero(
                    gt_edge & (np.asarray(distance_transform_edt(~pred_edge)) <= 2)
                ),
                np.count_nonzero(gt_edge),
            ],
            dtype=np.int64,
        )
        for key in [role, groups[name]] if name in holdout else [role]:
            totals[key] += matrix
            boundaries[key] += counts
        if index % 50 == 0:
            print(
                json.dumps(
                    {"arm": cfg["experiment_id"], "independent_views": index + 1}
                ),
                flush=True,
            )
    for role, matrix in totals.items():
        if matrix.tolist() != saved["metrics"][role]["confusion"]:
            raise ValueError(f"Pooled confusion differs: {role}")
        compare_metrics(confusion_metrics(matrix.tolist()), saved["metrics"][role])
        if boundaries[role].tolist() != saved["boundary_metrics"][role]["counts"]:
            raise ValueError(f"Independent boundary counts differ: {role}")
        c = boundaries[role]
        p, r = c[0] / c[1], c[2] / c[3]
        f1 = 2 * p * r / (p + r)
        if abs(f1 - saved["boundary_metrics"][role]["boundary_f1"]) > 1e-12:
            raise ValueError("Independent boundary F1 differs")
    return {
        "views_recounted": len(allowed),
        "prediction_hashes_verified": len(expected_rgb),
        "confusion_exact": True,
        "boundary_counts_exact": True,
        "label_hashes_verified": len(allowed),
    }


def main():
    pair, arms, split, pair_output = guard()
    require_seal(arms, pair_output)
    output = ROOT / pair["output_path"]
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "experiment_id": pair["experiment_id"],
        "status": "starting",
        "git_commit": pair["git_commit"],
        "split_id": pair["split_id"],
        "claim_scope": pair["scope"],
        "config_sha256": sha256(
            ROOT / "experiments/runs/QUAL-013-SPATIAL-V2-SEED-REPLICATION/config.yaml"
        ),
        "protocol_sha256": sha256(ROOT / pair["protocol"]),
        "script_sha256": sha256(Path(__file__)),
        "pair_binding_sha256": sha256(pair_output / "binding.json"),
        "pair_seal_sha256": sha256(pair_output / "pair-seal.json"),
        "independent_audit": {},
    }
    started = time.monotonic()
    try:
        raw = {"20260805": {}, "20260919": {}}
        input_hashes = {}
        for arm, cfg in arms.items():
            path = ROOT / cfg["output_path"] / "inference/result.json"
            new = read(path)
            if new["status"] != "completed":
                raise ValueError("Inference is incomplete")
            checkpoint = ROOT / cfg["output_path"] / "mask2former/model-6000.pt"
            if (
                sha256(checkpoint) != new["checkpoint_sha256_before_holdout_access"]
                or sha256(checkpoint) != new["checkpoint_sha256_after_holdout_access"]
            ):
                raise ValueError("Checkpoint changed during confirmation access")
            result["independent_audit"][arm] = recount(cfg, split, new)
            for phase in ("before", "after"):
                replay_path = pair_output / f"rgb-{phase}-{arm}.json"
                replay = read(replay_path)
                if replay["status"] != "passed_exact_rgb_replay" or any(
                    not v["exact"] for v in replay["views"].values()
                ):
                    raise ValueError("Frozen RGB replay failed")
                input_hashes[str(replay_path.relative_to(ROOT))] = sha256(replay_path)
            rgb_parent = read(ROOT / cfg["rgb_run"] / "result.json")
            verify_hashes(ROOT / cfg["rgb_run"], rgb_parent["artifacts"])
            rgb_evaluation = read(ROOT / cfg["rgb_evaluation"] / "result.json")
            verify_hashes(
                ROOT / cfg["rgb_evaluation"] / "images", rgb_evaluation["image_hashes"]
            )
            old_path = ROOT / pair["arms"][arm]["first_seed_result"]
            raw["20260805"][arm], raw["20260919"][arm] = read(old_path), new
            input_hashes[str(path.relative_to(ROOT))] = sha256(path)
            input_hashes[str(old_path.relative_to(ROOT))] = sha256(old_path)
        roles = ["train", "holdout", "region-1", "region-2", "region-3"]
        paired, rows = {}, []
        for seed, data in raw.items():
            a, s = data["absgs"], data["standard"]
            paired[seed] = {"absgs": {}, "standard": {}, "delta": {}, "class_delta": {}}
            for role in roles:
                av, sv = flatten(a, role), flatten(s, role)
                paired[seed]["absgs"][role], paired[seed]["standard"][role] = av, sv
                paired[seed]["delta"][role] = {k: av[k] - sv[k] for k in av}
                paired[seed]["class_delta"][role] = {
                    name: {
                        k: a["metrics"][role]["per_class"][name][k]
                        - s["metrics"][role]["per_class"][name][k]
                        for k in ("IoU", "precision", "recall", "fp", "fn")
                    }
                    for name in CLASSES
                }
            va = {v["name"]: v for v in a["per_view"] if v["role"] == "holdout"}
            vs = {v["name"]: v for v in s["per_view"] if v["role"] == "holdout"}
            for name in sorted(va):
                rows.append(
                    {
                        "seed": seed,
                        "view": name,
                        "region": va[name]["group"],
                        "foreground_delta_pp": 100
                        * (
                            va[name]["mIoU_without_background"]
                            - vs[name]["mIoU_without_background"]
                        ),
                        "five_class_delta_pp": 100
                        * (
                            va[name]["mIoU_with_background"]
                            - vs[name]["mIoU_with_background"]
                        ),
                        "stay_cable_iou_delta_pp": 100
                        * (
                            va[name]["per_class"]["stay_cable"]["IoU"]
                            - vs[name]["per_class"]["stay_cable"]["IoU"]
                        ),
                    }
                )
        means = {
            arm: {
                role: {
                    k: sum(paired[seed][arm][role][k] for seed in paired) / 2
                    for k in paired["20260805"][arm][role]
                }
                for role in roles
            }
            for arm in ("absgs", "standard", "delta")
        }
        gates = direction_gates(
            {seed: row["delta"]["holdout"] for seed, row in paired.items()}
        )
        guard()
        require_seal(arms, pair_output)
        with (output / "per-view.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result.update(
            status="completed_replication_passed"
            if gates["all_direction_gates_pass"]
            else "completed_replication_mixed_or_failed",
            input_hashes=input_hashes,
            paired=paired,
            two_seed_means=means,
            direction_gates=gates,
            engineering_audit_passed=True,
            primary_metric="foreground_miou",
            decision="Report each seed and regional trade-offs; no automatic new training or replacement",
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "new_seed_delta": paired["20260919"]["delta"]["holdout"],
                    "mean_delta": means["delta"]["holdout"],
                    "gates": gates,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        result.update(
            status="failed_preserved", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_new_json(output / "result.json", result)


if __name__ == "__main__":
    main()
