#!/usr/bin/env python3
"""Measure pixelwise semantic complementarity of AbsGS and Standard on v1/v2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
OUTPUT = ROOT / "outputs/quality/QUAL-020/complementarity.json"
CLASSES = ("background", "deck", "stay_cable", "tower", "foundation")
FOLDS = {
    "v1": {
        "split": ROOT / "experiments/splits/spatial_confirmation_v1.json",
        "absgs": ROOT / "outputs/quality/SEM-396/r1/inference/predictions",
        "standard": ROOT / "outputs/quality/SEM-407/r1/inference/predictions",
    },
    "v2": {
        "split": ROOT / "experiments/splits/spatial_confirmation_v2.json",
        "absgs": ROOT / "outputs/quality/SEM-412/r1/inference/predictions",
        "standard": ROOT / "outputs/quality/SEM-413/r1/inference/predictions",
    },
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def counts(target, absgs, standard):
    a_ok, s_ok = absgs == target, standard == target
    total = target.size
    return {
        "pixels": int(total),
        "both_correct": int((a_ok & s_ok).sum()),
        "absgs_only_correct": int((a_ok & ~s_ok).sum()),
        "standard_only_correct": int((~a_ok & s_ok).sum()),
        "both_wrong_same_prediction": int((~a_ok & ~s_ok & (absgs == standard)).sum()),
        "both_wrong_different_prediction": int((~a_ok & ~s_ok & (absgs != standard)).sum()),
        "absgs_accuracy": float(a_ok.mean()),
        "standard_accuracy": float(s_ok.mean()),
        "oracle_accuracy": float((a_ok | s_ok).mean()),
        "standard_rescue_fraction_of_absgs_errors": float((~a_ok & s_ok).sum() / max(1, (~a_ok).sum())),
        "absgs_rescue_fraction_of_standard_errors": float((a_ok & ~s_ok).sum() / max(1, (~s_ok).sum())),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = {
        "schema_version": 1,
        "status": "completed_read_only",
        "experiment_id": "QUAL-020-DUAL-RECONSTRUCTION-COMPLEMENTARITY",
        "scope": "two historically exposed zero-overlap spatial confirmation folds; descriptive upper bound, not a learned selector score",
        "folds": {},
    }
    for fold_name, paths in FOLDS.items():
        split = read(paths["split"])
        names = sorted(split["primary_fold"]["semantic_validation_labeled"])
        group = {name: row["group_id"] for row in split["groups"] for name in row["images"]}
        arrays = []
        by_group = {}
        for name in names:
            target = np.asarray(Image.open(LABELS / name), dtype=np.uint8)
            absgs = np.asarray(Image.open(paths["absgs"] / name), dtype=np.uint8)
            standard = np.asarray(Image.open(paths["standard"] / name), dtype=np.uint8)
            arrays.append((target, absgs, standard))
            by_group.setdefault(group[name], []).append((target, absgs, standard))
        target = np.concatenate([row[0].ravel() for row in arrays])
        absgs = np.concatenate([row[1].ravel() for row in arrays])
        standard = np.concatenate([row[2].ravel() for row in arrays])
        fold = counts(target, absgs, standard)
        fold["per_target_class"] = {}
        for class_id, class_name in enumerate(CLASSES):
            mask = target == class_id
            fold["per_target_class"][class_name] = counts(target[mask], absgs[mask], standard[mask])
        fold["per_group"] = {}
        for key, rows in sorted(by_group.items()):
            gt = np.concatenate([row[0].ravel() for row in rows])
            aa = np.concatenate([row[1].ravel() for row in rows])
            ss = np.concatenate([row[2].ravel() for row in rows])
            fold["per_group"][key] = counts(gt, aa, ss)
        result["folds"][fold_name] = fold
    structural_rescue = {
        fold: {
            name: row["per_target_class"][name]["standard_rescue_fraction_of_absgs_errors"]
            for name in CLASSES[1:]
        }
        for fold, row in result["folds"].items()
    }
    result["interpretation"] = {
        "standard_rescue_fraction_of_absgs_errors_by_structural_class": structural_rescue,
        "selector_is_not_oracle": True,
        "fusion_requires_training_only_observable_reliability_signal": True,
    }
    OUTPUT.parent.mkdir(parents=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

