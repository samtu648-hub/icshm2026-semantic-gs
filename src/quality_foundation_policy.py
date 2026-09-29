"""Fixed-model foundation diagnostics and historical calibration-policy control."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

from src.quality_protocol import sha256, verify_hashes
from src.quality_static_training import TrainingLabels
from src.quality_experts import validate_probability
from src.semantic_protected_fusion import partition_training_names

STAGES = ("static", "fiveclass_lane", "cable", "foundation", "deck")


def replace_foundation(parent, before, probability, threshold):
    if parent.dtype != np.uint8 or before.dtype != np.uint8 or parent.ndim != 2 or parent.shape != before.shape:
        raise ValueError("Expected aligned official class maps")
    if not parent.size or parent.max() > 4 or before.max() > 4:
        raise ValueError("Class outside0..4")
    if probability.shape != parent.shape or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Invalid same-grid foundation probability")
    if threshold is not None and (not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Invalid foundation threshold")
    eligible = before == 4
    if not np.isin(parent[eligible], [0, 4]).all():
        raise ValueError("Parent deck gate violated foundation provenance")
    result = parent.copy()
    result[eligible] = 4 if threshold is None else np.where(probability[eligible] < threshold, 0, 4)
    if any(not np.array_equal(parent == c, result == c) for c in (1, 2, 3)):
        raise ValueError("Protected class changed")
    return result


def changes(base, actual, target):
    changed = base != actual
    return {"changed": int(changed.sum()), "corrected": int((changed & (actual == target)).sum()),
            "harmed": int((changed & (base == target)).sum()),
            "wrong_to_wrong": int((changed & (base != target) & (actual != target)).sum())}


def historical_gate(candidate, baseline, requirements):
    f, b = candidate["per_class"]["foundation"], baseline["per_class"]["foundation"]
    if any(f[k] is None or b[k] is None for k in ("IoU", "precision", "recall")):
        return {"defined": False}
    fp = baseline["confusion"][0][4]
    reduction = 1 - candidate["confusion"][0][4] / fp if fp else 0.
    return {"defined": True,
        "foundation_iou_gain": f["IoU"] >= b["IoU"] + requirements["minimum_foundation_iou_gain"],
        "foundation_precision_gain": f["precision"] >= b["precision"] + requirements["minimum_foundation_precision_gain"],
        "foundation_recall": f["recall"] >= requirements["minimum_foundation_recall"],
        "miou_gain": candidate["mIoU_with_background"] >= baseline["mIoU_with_background"] + requirements["minimum_mIoU_gain"],
        "background_false_positive_reduction": reduction >= requirements["minimum_background_to_foundation_reduction"]}


def select_historical(records):
    passed = [r for r in records if r["checks"] and all(r["checks"].values())]
    if not passed:
        return None
    def rank(row):
        f = row["metrics"]["per_class"]["foundation"]
        return f["IoU"], f["precision"], row["metrics"]["mIoU_with_background"], f["recall"], -row["threshold"]
    return max(passed, key=rank)


def probability_strata(probability, before, target, bins, threshold):
    validate_probability(probability, 5, before.shape)
    if before.shape != target.shape or bins[0] != 0 or bins[-1] != 1 or not np.all(np.diff(bins) > 0):
        raise ValueError("Invalid stratum grid or probability bins")
    p, arg = probability[4], probability.argmax(0)
    eligible = before == 4
    result = {}
    for name, roi in (("true_foundation", target == 4), ("true_background", target == 0), ("other_true_class", (target != 0) & (target != 4))):
        mask = eligible & roi
        values = p[mask]
        result[name] = {"count": int(mask.sum()), "probability_sum": float(values.astype(np.float64).sum()),
            "histogram": np.histogram(values, bins=bins)[0].tolist(),
            "argmax_counts": np.bincount(arg[mask], minlength=5).tolist(),
            "rejected": int((mask & (p < threshold)).sum()),
            "rejected_but_argmax_foundation": int((mask & (p < threshold) & (arg == 4)).sum())}
    return result


class FrozenFoundationInputs:
    """Only the exact accepted new Scaffold sources; no old expert/RGB mixing."""
    def __init__(self, root, artifact_root, config):
        self.root, self.artifact_root, self.config = root, artifact_root, config
        self.base = root / config["parent_run"]
        if sha256(self.base / "result.json") != config["parent_result_sha256"]:
            raise ValueError("Different matched Scaffold parent")
        self.parent = json.loads((self.base / "result.json").read_text())
        if self.parent["status"] != "completed_matched_fusion_pending_human_review" or self.parent["arm"] != "Scaffold" or self.parent["seed"] != config["seed"]:
            raise ValueError("Unaccepted parent/seed/backbone")
        self.file_proofs = dict(self.parent["input_hashes"])
        for name in ("static_config", "historical_policy", "historical_final_config"):
            self.file_proofs[config[name]] = config[name+"_sha256"]
        verify_hashes(root, self.file_proofs)
        self.sc = json.loads((root / config["static_config"]).read_text())
        if config["resolution"] != [1320, 989] or config["pixel_space"] != "PINHOLE":
            raise ValueError("Unsupported native pixel space")
        oof = json.loads((root / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
        self.file_proofs[self.sc["label_manifest"]] = oof["label_manifest_sha256"]
        verify_hashes(root, self.file_proofs)
        self.label_manifest = json.loads((root / self.sc["label_manifest"]).read_text())
        self.experts = {}
        identity = None
        self.source_results = {}
        fits, audits = {}, {}
        for rel, digest in self.parent["input_result_hashes"].items():
            run = root / rel
            if not run.resolve().is_relative_to(root.resolve()) or sha256(run / "result.json") != digest:
                raise ValueError("Changed or escaped source result")
            self.source_results[rel] = digest
            r = json.loads((run / "result.json").read_text())
            if r.get("experiment_id") != "SEM-331-QUALITY-EXPERTS" or r.get("binding", {}).get("expert") != "fiveclass":
                continue
            if r["status"] == "completed_independent_expert_replay":
                audits[r["run_result_sha256"]] = r["binding"]
                continue
            if r["status"] != "completed_expert_fit_pending_independent_audit" or r["trained_steps"] != 3000:
                raise ValueError("Unaccepted expert")
            b = r["binding"]
            if b["identity"]["backbone"] != "scaffold-gs" or b["seed"] != config["seed"] or b["pixel_space"] != "PINHOLE":
                raise ValueError("Wrong backbone or seed")
            if identity is None:
                identity = b["identity"]
            if identity != b["identity"]:
                raise ValueError("Mixed RGB/topology checkpoints")
            if set(r["access"]["train"]) & set(r["access"]["prediction"]):
                raise ValueError("Leaked expert target labels")
            fits[digest] = b
            for name in r["prediction_views"]:
                if name in self.experts:
                    raise ValueError("Duplicate OOF target")
                self.experts[name] = (run, r)
        if len(fits) != 4 or fits != audits:
            raise ValueError("Four exact independently accepted expert sources required")
        self.identity = identity
        split = json.loads((root / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
        halves = partition_training_names(sorted(split["semantic_train_labeled"]), expected_count=219)
        self.roles = {**halves, "development": sorted(split["semantic_validation_labeled"])}
        self.roles = {k: self.roles[k] for k in ("calibration", "check", "development")}
        if [len(self.roles[k]) for k in self.roles] != [108, 111, 74] or set(self.experts) != set(sum(self.roles.values(), [])):
            raise ValueError("Missing target coverage or changed role split")
        for role, names in self.roles.items():
            if sorted(names) != sorted(self.parent["evaluation"][role]["views"]):
                raise ValueError("Parent role mismatch")
        self.allowed_roles = {"calibration"}
        self.access = {r: set() for r in self.roles}
        protocol = json.loads((root / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
        self.frozen = {x["path"]: x["sha256"] for x in json.loads((root / protocol["frozen_manifest"]).read_text())["artifacts"]}
        self.finish()

    def finish(self):
        verify_hashes(self.root, self.file_proofs)
        verify_hashes(self.artifact_root, self.frozen)
        if sha256(self.base / "result.json") != self.config["parent_result_sha256"]:
            raise ValueError("Parent changed")
        for rel, h in self.source_results.items():
            if sha256(self.root / rel / "result.json") != h:
                raise ValueError("Expert evidence changed")

    def labels(self, role):
        if role not in self.allowed_roles:
            raise ValueError("Holdout labels not released after calibration seal")
        return TrainingLabels(self.artifact_root / self.sc["labels"], self.roles[role], self.label_manifest, self.config["resolution"])

    def read(self, role, name):
        if role not in self.allowed_roles or name not in self.roles[role]:
            raise ValueError("Wrong view/role")
        result = {}
        for stage in STAGES:
            rel = f"{role}/{stage}/{name}"
            path = self.base / rel
            if sha256(path) != self.parent["artifacts"][rel]:
                raise ValueError("Changed saved stage image")
            with Image.open(path) as im:
                result[stage] = np.array(im)
        run, r = self.experts[name]
        rel = f"probabilities/{Path(name).stem}.npz"
        if sha256(run / rel) != r["artifacts"][rel]:
            raise ValueError("Changed expert probability")
        with np.load(run / rel, allow_pickle=False) as z:
            prob = z["probabilities"]
        validate_probability(prob, 5, (989, 1320))
        if any(p.shape != (989, 1320) or p.dtype != np.uint8 or p.max() > 4 for p in result.values()):
            raise ValueError("Pixel space/dimension mismatch")
        if not np.array_equal(result["static"] == 4, result["cable"] == 4):
            raise ValueError("Foundation footprint changed before verifier")
        expected = replace_foundation(result["deck"], result["cable"], prob[4], self.parent["thresholds"]["foundation"])
        if not np.array_equal(expected, result["deck"]):
            raise ValueError("Cannot replay parent foundation threshold")
        self.access[role].add(name)
        return result, prob, r["access"]["partition"]
