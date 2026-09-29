#!/usr/bin/env python3
"""SEM343:108-only finite calibration curves; no selection,111 scoring or promotion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels
from src.quality_semantic_audit import independent_confusion, metrics
from src.quality_raw_foundation_gate import apply_gate
from scripts.diagnose_quality_raw_fusion import read_probability, read_image

CONFIG = "experiments/runs/SEM-343-RAW-FOUNDATION-GATE/config.yaml"
CODE = ("scripts/calibrate_quality_raw_foundation_gate.py", "src/quality_raw_foundation_gate.py",
        "scripts/diagnose_quality_raw_fusion.py", "src/quality_protocol.py",
        "src/quality_static_training.py", "src/quality_semantic_audit.py", "src/quality_experts.py")


def compare(cm, baseline):
    m, b = metrics(cm), metrics(baseline)
    delta = {c: {k: None if m["per_class"][c][k] is None or b["per_class"][c][k] is None else
                100*(m["per_class"][c][k]-b["per_class"][c][k]) for k in ("IoU", "precision", "recall")}
             for c in m["per_class"]}
    return {"metrics": m, "delta_miou_pp": 100*(m["mIoU_with_background"]-b["mIoU_with_background"]),
            "delta_per_class_pp": delta}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"outputs/quality/raw_fusion"):
        p.error("Use a new isolated raw_fusion directory")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cfg = json.loads((ROOT/CONFIG).read_text())
    record = {"experiment_id": cfg["experiment_id"], "git_commit": cfg["git_commit"], "seed": cfg["seed"],
              "config_sha256": sha256(ROOT/CONFIG), "code_sha256": {n: sha256(ROOT/n) for n in CODE},
              "started_utc": datetime.now(timezone.utc).isoformat(), "new_training": False,
              "check_candidate_predictions": 0, "check_labels_decoded": 0, "development_labels_decoded": 0,
              "selected_margin": "not_selected_pending_quality_review", "automatic_promotion": False,
              "historically_reused_data": True, "whole_chain_spatial_validation": False}
    try:
        verify_hashes(ROOT, cfg["input_hashes"])
        parent_dir = ROOT/cfg["comparison"]
        parent = json.loads((parent_dir/"result.json").read_text())
        names = parent["roles"]["calibration"]
        if len(names) != 108 or set(names) & set(parent["roles"]["check"]):
            raise ValueError("Historical calibration partition changed")
        pc = json.loads((ROOT/cfg["preflight_config"]).read_text())
        export_dir = ROOT/pc["rgb_export"]
        export = json.loads((export_dir/"result.json").read_text())
        rgb = {e["name"]: e["sha256"] for e in export["rendered_images"]}
        frozen = {e["path"]: e["sha256"] for e in json.loads((ROOT/cfg["frozen_manifest"]).read_text())["artifacts"]}
        def guard():
            if shutil.disk_usage(ROOT).free < 100*1024**3:
                raise OSError("100GiB reserve reached")
            verify_hashes(export_dir/"images", rgb)
            verify_hashes(args.artifact_root, frozen)
        guard()
        sources = {}
        for part in ("oof-0", "oof-1", "oof-2"):
            base = ROOT/cfg["batch"]/("full-"+part)
            rel = str((base/"result.json").relative_to(ROOT))
            verify_hashes(ROOT, {rel: parent["input_result_hashes"][rel]})
            sources[part] = (base, json.loads((base/"result.json").read_text()))
        view_info = json.loads((parent_dir/"calibration/per_view.json").read_text())
        manifest = json.loads((ROOT/pc["label_manifest"]).read_text())
        labels = TrainingLabels(args.artifact_root/"r2/sem_001/labels", names, manifest, [1320,989])
        for name in (CONFIG, *CODE):
            dest = output/"code"/name; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, dest)
        # Written before first target label decode; the grid cannot adapt to results.
        write_new_json(output/"started.json", record)
        candidates = cfg["maximum_margin_grid"]
        groups = {part: np.zeros((len(candidates),5,5), np.int64) for part in sources}
        transitions = {part: np.zeros((len(candidates),5), np.int64) for part in sources}
        per_view = {}
        for index, name in enumerate(names):
            part = view_info[name]["partition"]
            base, source = sources[part]
            if name not in source["access"]["prediction"]:
                raise ValueError("View is not OOF for frozen model")
            maps = {}
            for mode in ("raw", "calibrated"):
                rel = f"calibration/mask2former_6k/{mode}/{name}"
                verify_hashes(parent_dir, {rel: parent["artifacts"][rel]})
                maps[mode] = read_image(parent_dir/rel)
            raw, fused = maps["raw"], maps["calibrated"]
            five = read_probability(base, source, "probabilities/"+Path(name).stem+".npz",5)
            if not np.array_equal(raw, five.argmax(0)):
                raise ValueError("Raw probability argmax changed")
            # Construct all candidates before consulting this target's labels.
            proposals = [apply_gate(raw, fused, five, t) for t in candidates]
            target = labels.read(name)
            baseline = independent_confusion(target, raw)
            if baseline.tolist() != view_info[name]["metrics"]["mask2former_6k/raw"]["confusion"]:
                raise ValueError("Raw baseline differs")
            rows = []
            for j, prediction in enumerate(proposals):
                changed = prediction != raw
                if np.any(changed & ((raw != 3) | (prediction != 4))) or any(not np.array_equal(raw == c, prediction == c) for c in (0,1,2)):
                    raise ValueError("Protected predictions changed")
                counts = np.bincount(target[changed], minlength=5)
                cm = independent_confusion(target, prediction)
                # Independent confusion update from a single3->4 transition.
                reconstructed = baseline.copy(); reconstructed[:,3] -= counts; reconstructed[:,4] += counts
                if not np.array_equal(cm, reconstructed):
                    raise ValueError("Direct recount and algebraic reconstruction differ")
                groups[part][j] += cm; transitions[part][j] += counts
                rows.append({"maximum_margin": candidates[j], "confusion": cm.tolist(), "changed_truth_counts": counts.tolist()})
            per_view[name] = {"partition": part, "curves": rows}
            if index % 24 == 0:
                print(json.dumps({"calibration_views": index+1, "total":108}), flush=True)
        total, changes = sum(groups.values()), sum(transitions.values())
        curves = []
        for j, t in enumerate(candidates):
            result = compare(total[j], total[0])
            by_group = {part: compare(cm[j], cm[0]) for part,cm in groups.items()}
            truth = changes[j]
            qualifies = all(v["delta_miou_pp"] >= 0 and all(v["delta_per_class_pp"][c]["IoU"] >= 0 for c in ("tower", "foundation")) for v in by_group.values())
            curves.append({"maximum_margin": t, **result, "per_partition": by_group,
                           "changed_truth_counts": truth.tolist(), "corrected": int(truth[4]),
                           "harmed": int(truth[3]), "wrong_to_wrong": int(truth[:3].sum()),
                           "spatial_screen_pass": qualifies,
                           "requires_tower_recall_tradeoff_review": bool(truth[3]),
                           "group_changed_truth_counts": {part: arr[j].tolist() for part,arr in transitions.items()}})
        if sorted(labels.accessed) != sorted(names):
            raise ValueError("Incomplete108 calibration coverage")
        guard()
        verify_hashes(ROOT, {CONFIG: record["config_sha256"], **record["code_sha256"], **cfg["input_hashes"]})
        write_new_json(output/"calibration_per_view.json", per_view)
        record.update(status="completed_calibration_curves_pending_quality_review", calibration_views=sorted(labels.accessed),
                      curves=curves, frozen_artifacts_verified=len(frozen), rgb_exports_unchanged=len(rgb),
                      calibration_per_view_sha256=sha256(output/"calibration_per_view.json"))
    except BaseException as exc:
        record.update(status="failed_calibration_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic()-start
        write_new_json(output/"result.json", record)
        print(json.dumps({k:record[k] for k in ("status","seconds")}), flush=True)


if __name__ == "__main__":
    main()
