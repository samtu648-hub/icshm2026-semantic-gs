#!/usr/bin/env python3
"""Independent CPU acceptance of SEM340 records/probabilities/roles and immutability."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status": "auditing", "quality_metrics": None, "new_training": False,
              "script_sha256": sha256(Path(__file__))}
    try:
        probe_root, fresh_root = a.root / "probe-eager-r5", a.root / "fresh-eager-r1"
        probe = json.loads((probe_root / "result.json").read_text())
        fresh = json.loads((fresh_root / "result.json").read_text())
        if (probe["status"] != "completed_two_step_probe_pending_fresh"
                or fresh["status"] != "completed_independent_native_replay"
                or probe["trained_steps"] != 2 or fresh["trained_steps"] != 0
                or probe["binding"] != fresh["binding"] or probe["runtime"] != fresh["runtime"]):
            raise ValueError("Preflight completion/binding/runtime discrepancy")
        binding = probe["binding"]
        cfg_path = "experiments/runs/SEM-340-MASK2FORMER-PREFLIGHT/config.yaml"
        verify_hashes(ROOT, {cfg_path: binding["config_sha256"], **binding["code_sha256"]})
        verify_hashes(a.root, binding["upstream_runtime_sha256"])
        cfg = json.loads((ROOT / cfg_path).read_text())
        for name in ["dataset_manifest", "semantic_split", "oof_split", "label_manifest", "rgb_result"]:
            verify_hashes(ROOT, {cfg[name]: cfg[name + "_sha256"]})
        fold = json.loads((ROOT / cfg["oof_split"]).read_text())["folds"][0]
        name = sorted(fold["train"])[0]
        if (name in fold["oof"] or name in fold["guard"] or binding["view"] != name
                or probe["training_label_views"] != [name] or fresh["training_label_views"]):
            raise ValueError("Training-only sample/read isolation differs")
        if any(r["target_labels_decoded"] or r["rgb_model_imported"] or r["full_training"] for r in (probe, fresh)):
            raise ValueError("Unexpected evaluation or RGB fitting")
        load = probe["pretrained_load"]
        if (load["official_tensors"] != 650 or load["loaded_tensors"] != 647
                or not load["all_loaded_tensors_exact"] or probe["parameters"] != 215451514
                or probe["trainable_parameters"] != probe["parameters"]):
            raise ValueError("Incomplete full pretrained expert")
        for root, data in [(probe_root, probe), (fresh_root, fresh)]:
            verify_hashes(root, data["artifacts"])
            prob = np.load(root / "probability.npy", allow_pickle=False)
            if (prob.shape != (5, 989, 1320) or prob.dtype != np.float32 or not np.isfinite(prob).all()
                    or (prob < 0).any() or (prob > 1).any()
                    or np.max(np.abs(np.sum(prob.astype(np.float64), axis=0)-1)) > 2e-6):
                raise ValueError("Invalid independently checked native probabilities")
        ref = np.load(probe_root / "probability.npy", allow_pickle=False)
        repeated = np.load(fresh_root / "probability.npy", allow_pickle=False)
        if not np.array_equal(ref, repeated) or fresh["replay_max_abs"] != 0:
            raise ValueError("Native replay differs")
        export_root = ROOT / cfg["rgb_export"]
        verify_hashes(export_root, {"result.json": cfg["rgb_export_result_sha256"]})
        export = json.loads((export_root / "result.json").read_text())
        verify_hashes(export_root / "images", {e["name"]: e["sha256"] for e in export["rendered_images"]})
        frozen = json.loads((ROOT / "experiments/runs/SUBMIT-020-SEM161-FREEZE/frozen_manifest.json").read_text())
        verify_hashes(a.artifact_root, {e["path"]: e["sha256"] for e in frozen["artifacts"]})
        if any(not np.isfinite(s["loss"]) or not np.isfinite(s["grad_norm"]) for s in probe["steps"]):
            raise ValueError("Successful step has nonfinite loss/gradient")
        record.update(status="completed_native_preflight_not_quality_validation", training_view=name,
                      probe_result_sha256=sha256(probe_root / "result.json"),
                      fresh_result_sha256=sha256(fresh_root / "result.json"), config_sha256=binding["config_sha256"],
                      compared_values=int(ref.size), native_probability_exact=True,
                      frozen_artifacts_verified=len(frozen["artifacts"]), rgb_exports_unchanged=len(export["rendered_images"]),
                      successful_steps=probe["trained_steps"], amp_skipped_steps=len(probe["amp_skips"]),
                      training_peak_allocated_mib=probe["training_peak_allocated_mib"],
                      training_peak_reserved_mib=probe["training_peak_reserved_mib"],
                      inference_peak_allocated_mib=probe["inference_peak_allocated_mib"],
                      steps_seconds=[s["seconds"] for s in probe["steps"]],
                      native_inference_seconds=[probe["inference_seconds"], fresh["inference_seconds"]])
    except Exception as exc:
        record.update(status="failed_independent_audit", error=repr(exc))
        raise
    finally:
        record["seconds"] = time.monotonic()-started
        write_new_json(a.output / "result.json", record)


if __name__ == "__main__":
    main()
