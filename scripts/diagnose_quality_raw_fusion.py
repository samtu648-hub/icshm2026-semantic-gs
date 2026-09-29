#!/usr/bin/env python3
"""SEM342: reuse SEM341 artifacts, reconstruct fixed fusion and audit raw costs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_experts import validate_probability
from src.quality_static_training import TrainingLabels
from src.quality_mask2former_comparison import independent_map
from src.quality_raw_fusion_diagnostic import transition_cube, describe, provenance_regions

CONFIG = "experiments/runs/SEM-342-RAW-FUSION-DIAG/config.yaml"
CODE = ("scripts/diagnose_quality_raw_fusion.py", "src/quality_raw_fusion_diagnostic.py",
        "src/quality_protocol.py", "src/quality_static_training.py", "src/quality_semantic_audit.py",
        "src/quality_experts.py", "src/quality_mask2former_comparison.py", "src/quality_fusion.py")


def read_image(path):
    with Image.open(path) as image:
        return np.array(image)


def read_probability(base, record, relative, classes):
    verify_hashes(base, {relative: record["artifacts"][relative]})
    with np.load(base/relative, allow_pickle=False) as archive:
        value = archive["probabilities"]
    validate_probability(value, classes, (989, 1320))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=os.environ.get("ICSHM_OUTPUT_ROOT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact_root is None:
        parser.error("Supply --artifact-root or ICSHM_OUTPUT_ROOT")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"outputs/quality"):
        parser.error("Use a new directory inside outputs/quality")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg = json.loads((ROOT/CONFIG).read_text())
    code_hashes = {n: sha256(ROOT/n) for n in CODE}
    record = {"experiment_id": cfg["experiment_id"], "git_commit": cfg["git_commit"],
              "seed": cfg["seed"], "config_sha256": sha256(ROOT/CONFIG), "code_sha256": code_hashes,
              "started_utc": datetime.now(timezone.utc).isoformat(), "new_training": False,
              "new_model_inference": False, "threshold_selection": False,
              "development_labels_decoded": False, "automatic_promotion": False,
              "whole_chain_spatial_validation": False, "historically_reused_data": True}
    try:
        verify_hashes(ROOT, cfg["input_hashes"])
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != cfg["git_commit"]:
            raise ValueError("Repository commit differs")
        comparison = ROOT/cfg["comparison"]
        parent = json.loads((comparison/"result.json").read_text())
        replay = json.loads((ROOT/cfg["replay_result"]).read_text())
        if parent["evaluation"] != replay["evaluation"] or parent["thresholds"] != replay["thresholds"]:
            raise ValueError("Parent independent evaluation differs")
        verify_hashes(ROOT, parent["code_sha256"])
        verify_hashes(ROOT, parent["input_result_hashes"])
        thresholds = parent["thresholds"]["mask2former_6k"]
        if thresholds != cfg["frozen_thresholds"] or thresholds["cable"] is not None:
            raise ValueError("Unexpected frozen fusion; disabled cable branch must stay disabled")
        pc = json.loads((ROOT/cfg["preflight_config"]).read_text())
        rgb_root = ROOT/pc["rgb_export"]
        export = json.loads((rgb_root/"result.json").read_text())
        rgb_hashes = {e["name"]: e["sha256"] for e in export["rendered_images"]}
        frozen = {e["path"]: e["sha256"] for e in json.loads((ROOT/cfg["frozen_manifest"]).read_text())["artifacts"]}
        def guard():
            if shutil.disk_usage(ROOT).free < cfg["minimum_free_gib"]*1024**3:
                raise OSError("100GiB reserve reached; no cleanup or fallback")
            verify_hashes(rgb_root/"images", rgb_hashes)
            verify_hashes(args.artifact_root, frozen)
        guard()
        sources = {}
        for rel in parent["input_result_hashes"]:
            base = (ROOT/rel).parent
            if base.name.startswith(("SEM-330-Scaffold-", "SEM-331-Scaffold-deck-", "full-oof-")):
                r = json.loads((base/"result.json").read_text())
                if r["experiment_id"] == "SEM-341-MASK2FORMER-OOF":
                    kind, part = "five", r["partition"]
                elif base.name.startswith("SEM-330-Scaffold-"):
                    kind, part = "static", r["access"]["partition"]
                elif base.name.startswith("SEM-331-Scaffold-deck-"):
                    kind, part = "deck", r["access"]["partition"]
                else:
                    continue
                sources.setdefault(part, {})[kind] = (base, r)
        if len(sources) != 3 or any(set(s) != {"static", "five", "deck"} for s in sources.values()):
            raise ValueError("Missing exact accepted source folds")
        for n in (CONFIG, *CODE):
            dest = output/"code"/n
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/n, dest)
        write_new_json(output/"started.json", record)
        manifest = json.loads((ROOT/pc["label_manifest"]).read_text())
        evaluation = {}
        for role in ("calibration", "check"):
            views = parent["roles"][role]
            labels = TrainingLabels(args.artifact_root/"r2/sem_001/labels", views, manifest, [1320, 989])
            old_views = json.loads((comparison/role/"per_view.json").read_text())
            pooled = np.zeros((5, 5, 5), np.int64)
            groups, regions, per_view, tiles, confidence = {}, {}, {}, {}, {}
            for idx, name in enumerate(views):
                part = old_views[name]["partition"]
                source = sources[part]
                if name not in source["five"][1]["access"]["prediction"]:
                    raise ValueError("Target not in model OOF prediction role")
                maps = {}
                for mode in ("raw", "calibrated"):
                    rel = f"{role}/mask2former_6k/{mode}/{name}"
                    verify_hashes(comparison, {rel: parent["artifacts"][rel]})
                    maps[mode] = read_image(comparison/rel)
                raw, fused = maps["raw"], maps["calibrated"]
                base, r = source["static"]
                rel = "predictions/"+name
                verify_hashes(base, {rel: r["artifacts"][rel]})
                static = read_image(base/rel)
                five = read_probability(*source["five"], "probabilities/"+Path(name).stem+".npz", 5)
                deck = read_probability(*source["deck"], "probabilities/"+Path(name).stem+".npz", 2)[1]
                rebuilt = independent_map(static, five, np.zeros_like(deck), deck, thresholds)
                if not np.array_equal(raw, five.argmax(0)) or not np.array_equal(fused, rebuilt):
                    raise ValueError("Frozen raw or calibrated output no longer reconstructs")
                target = labels.read(name)
                cube = transition_cube(raw, fused, target)
                result = describe(cube)
                for key, mode in (("raw", "raw"), ("calibrated_fusion", "calibrated")):
                    if result[key]["confusion"] != old_views[name]["metrics"]["mask2former_6k/"+mode]["confusion"]:
                        raise ValueError("Per-view baseline confusion differs")
                pooled += cube
                groups.setdefault(part, np.zeros_like(cube))[:] += cube
                per_view[name] = {"partition": part, **result}
                changed = raw != fused
                ids = raw.astype(np.int64)*25+fused.astype(np.int64)*5+target
                for label, mask in provenance_regions(static, fused).items():
                    regions.setdefault(label, np.zeros_like(cube))[:] += np.bincount(ids[mask], minlength=125).reshape(5,5,5)
                for y in range(3):
                    for x in range(3):
                        sl = ids[y*989//3:(y+1)*989//3, x*1320//3:(x+1)*1320//3]
                        tiles.setdefault(f"row{y}_col{x}", np.zeros_like(cube))[:] += np.bincount(sl.ravel(), minlength=125).reshape(5,5,5)
                for evidence, values in (("raw_confidence", five.max(0)),
                                         ("fusion_class_probability", np.take_along_axis(five, fused[None], 0)[0])):
                    for lo, hi in zip(cfg["descriptive_probability_bins"][:-1], cfg["descriptive_probability_bins"][1:]):
                        mask = changed & (values >= lo) & (values < hi)
                        key = f"{evidence}/{lo}:{hi}"
                        confidence.setdefault(key, np.zeros_like(cube))[:] += np.bincount(ids[mask], minlength=125).reshape(5,5,5)
                if idx % 24 == 0:
                    print(json.dumps({"role": role, "views": idx+1, "total": len(views)}), flush=True)
            total = describe(pooled)
            for key, mode in (("raw", "raw"), ("calibrated_fusion", "calibrated")):
                if total[key] != parent["evaluation"][role]["metrics"]["mask2former_6k/"+mode]:
                    raise ValueError("Aggregate metric reproduction failed")
            if not np.array_equal(sum(groups.values()), pooled) or not np.array_equal(sum(regions.values()), pooled) or not np.array_equal(sum(tiles.values()), pooled):
                raise ValueError("Spatial or provenance partition fails reconciliation")
            evaluation[role] = {"views": sorted(labels.accessed), "total": total,
                                "per_partition": {k: describe(v) for k,v in groups.items()},
                                "provenance": {k: describe(v) for k,v in regions.items()},
                                "image_tiles": {k: describe(v) for k,v in tiles.items()},
                                "descriptive_confidence": {k: describe(v) for k,v in confidence.items()}}
            write_new_json(output/f"{role}_per_view.json", per_view)
        guard()
        verify_hashes(ROOT, {CONFIG: record["config_sha256"], **code_hashes, **cfg["input_hashes"], **parent["input_result_hashes"]})
        record.update(status="completed_readonly_complementarity_diagnostic", evaluation=evaluation,
                      rgb_exports_unchanged=len(rgb_hashes), frozen_artifacts_verified=len(frozen),
                      reconstructed_views=sum(len(v["views"]) for v in evaluation.values()),
                      artifacts={p.name: sha256(p) for p in output.glob("*_per_view.json")})
    except BaseException as exc:
        record.update(status="failed_diagnostic_preserved", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic()-started
        write_new_json(output/"result.json", record)
        print(json.dumps({k: record[k] for k in ("status", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
