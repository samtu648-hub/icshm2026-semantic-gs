#!/usr/bin/env python3
"""Archive paired RGB, native semantic stages and expert evidence without tuning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    from src.quality_diagnostics import diagnose_stages, deck_gap_provenance
    config_path = ROOT / "experiments/runs/QUAL-002-PAIRED-GAP-DIAG/config.yaml"
    config = json.loads(config_path.read_text())
    protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"experiment_id": config["experiment_id"], "config_sha256": sha256(config_path), "seed": config["seed"],
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "status": "running", "views": {}, "pixel_space": "PINHOLE", "geometry_truth_claim": None,
              "new_training": False, "prediction_changes": False, "historically_reused_developer_data": True}
    frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
    frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
    try:
        record["input_hashes"] = verify_hashes(ROOT, protocol["input_hashes"])
        verify_hashes(args.artifact_root, frozen_hashes)
        split = json.loads((ROOT / "experiments/splits/rgb_r1_spatial_v1.json").read_text())["primary_fold"]
        if not set(config["views"]) <= set(split["validation_labeled"]):
            raise ValueError("Unexpected target outside fixed developer role")
        manifest_path = args.artifact_root / config["render_manifest"]
        manifest = json.loads(manifest_path.read_text())
        render_hashes = {v["name"]: v["sha256"] for v in manifest["rendered_images"]}
        record["render_manifest_sha256"] = sha256(manifest_path)
        for view in config["views"]:
            folder = args.output / Path(view).stem
            folder.mkdir()
            inputs = {}
            def archive(path, role):
                digest = sha256(path)
                dest = folder / (role + path.suffix)
                shutil.copy2(path, dest)
                if sha256(dest) != digest:
                    raise ValueError("Archive mismatch")
                inputs[role] = {"source": str(path), "file": dest.name, "sha256": digest}
                return dest
            def png(path, role):
                return np.asarray(Image.open(archive(path, role)))
            target = png(args.artifact_root / config["labels"] / view, "target")
            real = png(args.artifact_root / config["paired_photos"] / view, "real_rgb")
            rendered_path = args.artifact_root / config["rendered_rgb"] / view
            if sha256(rendered_path) != render_hashes[view]:
                raise ValueError("Changed historical RGB")
            rgb = png(rendered_path, "reconstructed_rgb")
            if real.shape != (989, 1320, 3) or rgb.shape != real.shape or target.shape != real.shape[:2]:
                raise ValueError("Inputs do not share native PINHOLE grid")
            stages = {name: png((args.artifact_root if scope == "artifact" else ROOT) / path / view, name)
                      for name, scope, path in config["stages"]}
            result, changes = diagnose_stages(stages, target=target, has_same_camera_photo=True)
            gap, masks = deck_gap_provenance(stages, target)
            mae = np.abs(real.astype(np.float32) / 255 - rgb.astype(np.float32) / 255).mean(-1)
            gap["rgb_mae_association_not_causation"] = {
                name: (float(mae[mask].mean()) if mask.any() else None) for name, mask in masks.items()}
            experts = {}
            for name, path, key in config["experts"]:
                with np.load(archive(args.artifact_root / path / (Path(view).stem + ".npz"), "expert_" + name), allow_pickle=False) as data:
                    probability = data[key].astype(np.float32)
                expected = (5, *target.shape) if name == "fiveclass" else target.shape
                if probability.shape != expected or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                    raise ValueError("Invalid archived expert probabilities")
                # Keep float16 archives unchanged. Measure quantization error, not
                # silently renormalize scores or reinterpret binary as five-class.
                evidence = {"conditional_mean_on_final_deck_gap": None}
                if name == "fiveclass":
                    deviation = float(np.abs(probability.sum(0) - 1).max())
                    if deviation > 0.002:
                        raise ValueError("Unexpected nonnormalized historical five-class archive")
                    evidence["float16_sum_max_deviation"] = deviation
                    evidence["raw_expert_deck_on_final_gap"] = int(((probability.argmax(0) == 1) & masks["final_deck_gap"]).sum())
                    probability = probability[1]
                if masks["final_deck_gap"].any():
                    evidence["conditional_mean_on_final_deck_gap"] = float(probability[masks["final_deck_gap"]].mean())
                if name == "deck":
                    evidence["at_existing_0_99_threshold_on_final_gap"] = int(((probability >= .99) & masks["final_deck_gap"]).sum())
                    evidence["threshold_count_note"] = "quantized float16 archive, not an exact replay of pre-archive float32 threshold decisions"
                    final_gap = masks["final_deck_gap"]
                    eligible = (stages["SEM141_stay_cable"] == 0) & (stages["SEM151_foundation"] == 0)
                    evidence["recovery_eligible_final_gap"] = int((eligible & final_gap).sum())
                    evidence["recovery_blocked_final_gap"] = int((~eligible & final_gap).sum())
                    evidence["pre_foundation_classes_on_final_gap"] = np.bincount(stages["SEM141_stay_cable"][final_gap], minlength=5).tolist()
                experts[name] = evidence
            for index, (name, mask) in enumerate({**changes, **masks}.items()):
                filename = f"mask_{index:02d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / filename)
                inputs["mask/" + name] = {"file": filename, "sha256": sha256(folder / filename)}
            entry = {"stages": result, "deck_gaps": gap, "experts": experts, "inputs": inputs}
            write_new_json(folder / "result.json", entry)
            record["views"][view] = {"result_sha256": sha256(folder / "result.json"), "deck_gaps": gap, "experts": experts}
        verify_hashes(args.artifact_root, frozen_hashes)
        record.update(status="completed_six_paired_view_stage_diagnostics", frozen_verified=len(frozen_hashes))
    except BaseException as exc:
        record.update(status="paired_gap_diagnostics_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        record["code_sha256"] = {n: sha256(ROOT / n) for n in ("scripts/run_paired_gap_diagnostics.py", "src/quality_diagnostics.py", "src/quality_backbone.py", "src/quality_protocol.py", "src/semantic2d.py")}
        write_new_json(args.output / "result.json", record)
        print(json.dumps({"status": record["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
