#!/usr/bin/env python3
"""Independent CPU audit for sealed SEM429/430 photo-teacher diagnostics."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confusion(target, prediction):
    return np.bincount((5 * target.astype(np.int64) + prediction).ravel(), minlength=25).reshape(5, 5)


def audit_run(base: Path, protected):
    result = read(base / "result.json")
    seal = read(base / "inference-seal.json")
    names = result["validation_names"]
    labels = ROOT / "outputs/quality/all_data/SEM-370-r1/labels"
    baseline_total = np.zeros((5, 5), np.int64)
    candidate_total = np.zeros((5, 5), np.int64)
    changed = corrected = harmed = neutral = 0
    protected_exact = True
    for name in names:
        target = np.asarray(Image.open(labels / name), np.uint8)
        baseline = np.asarray(Image.open(base / "validation/render" / name), np.uint8)
        candidate = np.asarray(Image.open(base / "validation/candidate" / name), np.uint8)
        mask = baseline != candidate
        before, after = baseline == target, candidate == target
        changed += int(mask.sum())
        corrected += int(np.sum(mask & ~before & after))
        harmed += int(np.sum(mask & before & ~after))
        neutral += int(np.sum(mask & ~before & ~after))
        baseline_total += confusion(target, baseline)
        candidate_total += confusion(target, candidate)
        protected_exact &= all(np.array_equal(baseline == class_id, candidate == class_id) for class_id in protected)
    observed = result["validation"]
    rows = seal.get("validation", seal.get("rows"))
    hashes_exact = True
    for row in rows:
        partition = base / "validation"
        candidate_key = "candidate_prediction_sha256" if "candidate_prediction_sha256" in row else "candidate_sha256"
        mask_key = "accepted_mask_sha256" if "accepted_mask_sha256" in row else "mask_sha256"
        hashes_exact &= sha(partition / "candidate" / row["name"]) == row[candidate_key]
        hashes_exact &= sha(partition / "accepted-mask" / row["name"]) == row[mask_key]
    return {
        "experiment_id": result["experiment_id"],
        "baseline_confusion_exact": baseline_total.tolist() == observed["baseline"]["confusion"],
        "candidate_confusion_exact": candidate_total.tolist() == observed["candidate"]["confusion"],
        "changed_exact": changed == observed["changed_pixels"],
        "corrected_exact": corrected == observed["corrected_pixels"],
        "harmed_exact": harmed == observed["harmed_pixels"],
        "neutral_exact": neutral == observed["neutral_wrong_to_wrong"],
        "protected_predictions_exact": bool(protected_exact),
        "sealed_output_hashes_exact": bool(hashes_exact),
    }


def main():
    sem429 = ROOT / "outputs/quality/SEM-429/r2"
    sem430 = ROOT / "outputs/quality/SEM-430/r1"
    split1 = read(ROOT / "experiments/splits/spatial_confirmation_v1.json")
    split2 = read(ROOT / "experiments/splits/spatial_confirmation_v2.json")
    views1 = set(split1["primary_fold"]["semantic_validation_labeled"])
    views2 = set(split2["primary_fold"]["semantic_validation_labeled"])
    checks = {
        "sem429": audit_run(sem429, (2, 4)),
        "sem430": audit_run(sem430, (2, 3, 4)),
        "validation_folds_zero_overlap": not bool(views1 & views2),
        "sem396_checkpoint_exact": sha(ROOT / "outputs/quality/SEM-396/r1/mask2former/model-6000.pt") == "eab4aa600104c0c4f1341c89306d41750c4552d33bb607c6c46390735ce6e8eb",
        "sem378_checkpoint_exact": sha(ROOT / "outputs/quality/SEM-378/r1/mask2former/model-6000.pt") == "7b513da44b8bdf695be55f8808916018b7519e52289d4d45d293a71a6792db95",
        "sem412_checkpoint_exact": sha(ROOT / "outputs/quality/SEM-412/r1/mask2former/model-6000.pt") == read(ROOT / "outputs/quality/SEM-412/r1/mask2former/result.json")["artifacts"]["model-6000.pt"],
    }
    flat = [value for value in checks.values() if isinstance(value, bool)]
    for section in (checks["sem429"], checks["sem430"]):
        flat.extend(value for key, value in section.items() if key != "experiment_id")
    output = {
        "schema_version": 1,
        "status": "passed" if all(flat) else "failed",
        "checks": checks,
        "passed": int(sum(flat)),
        "total": len(flat),
    }
    path = sem430 / "audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if output["status"] != "passed":
        raise SystemExit("SEM429/430 audit failed")
    print(json.dumps({"status": output["status"], "passed": output["passed"], "total": output["total"]}))


if __name__ == "__main__":
    main()
