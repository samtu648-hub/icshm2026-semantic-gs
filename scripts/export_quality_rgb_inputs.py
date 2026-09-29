#!/usr/bin/env python3
"""Prepare expert RGB inputs with the same metadata-only deployment renderer.

Requires a full RGB checkpoint and its actual native adapter probe. No real
photos or labels are decoded. This does not fit an expert, distill GAGS, evaluate
semantics or authorize use of development images as training inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import json
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json
from scripts.run_quality_rgb import source_manifest
from scripts.run_quality_stage_a import verify_run, verify_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partition", choices=("semantic-training", "development"), required=True)
    args = p.parse_args()
    run, source, output = args.run.resolve(), args.source.resolve(), args.output.resolve()
    with (run.parent.parent / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "initializing", "partition": args.partition, "real_rgb_decoded": False,
                  "labels_decoded": False, "camera_metadata": "public metadata; train and evaluation poses may be read",
                  "pixel_space": "PINHOLE", "quality_metrics": None}
        started = time.monotonic()
        try:
            import numpy as np
            from PIL import Image
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import NativeGaussianRenderer
            from src.quality_scaffold import ScaffoldQualityRenderer
            experiment = json.loads((run / "result.json").read_text())["experiment_id"]
            protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
            common = args.artifact_root / protocol["rgb"]["scene"] / "sparse/0/points3D.ply"
            parent = verify_run(run, mode="full", experiment_id=experiment, common_ply_sha256=sha256(common))
            audit = verify_audit(args.audit, run, "completed_rgb_reload_audit")
            if not audit.get("fresh_reloads_exact") or audit.get("cameras") != 74:
                raise ValueError("Full independent RGB audit required")
            proof = verify_audit(args.probe, run, "passed_native_adapter_engineering_probe")
            verify_hashes(ROOT, proof["code_sha256"])
            config = json.loads((ROOT / "experiments/runs" / experiment / "config.yaml").read_text())
            sources = source_manifest(source)
            if sources != json.loads((run / "source_manifest.json").read_text()):
                raise ValueError("Different upstream source/extensions than full training")
            if len(args.extra_pythonpath) != len(parent.get("extra_sources", [])):
                raise ValueError("Exact auxiliary RGB source list required")
            for extra, entry in zip(args.extra_pythonpath, parent.get("extra_sources", [])):
                if (extra.name != entry["name"] or sha256(run / entry["manifest"]) != entry["manifest_sha256"]
                        or source_manifest(extra) != json.loads((run / entry["manifest"]).read_text())):
                    raise ValueError("Auxiliary RGB source differs")
            export_code = ("scripts/export_quality_rgb_inputs.py", "scripts/run_quality_rgb.py", "scripts/run_quality_stage_a.py",
                           "src/quality_protocol.py", *proof["code_sha256"])
            record["code_sha256"] = {n: sha256(ROOT / n) for n in export_code}
            for name in record["code_sha256"]:
                dest = output / "code" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
            verify_hashes(args.artifact_root, frozen_hashes)
            split_path = ROOT / "experiments/splits/sem_001_scaffold_static_v1.json"
            split = json.loads(split_path.read_text())
            names = split["semantic_train_labeled" if args.partition == "semantic-training" else "semantic_validation_labeled"]
            access = json.loads((run / "access.json").read_text())
            allowed = set(access["train" if args.partition == "semantic-training" else "evaluation"])
            if len(names) != len(set(names)) or not set(names) <= allowed:
                raise ValueError("Semantic role is not covered by the parent RGB access")
            cameras = read_camera_requests(run / "scene/sparse/0")
            if config["backbone"] == "scaffold-gs":
                renderer = ScaffoldQualityRenderer(source, run / "model", 30000, train_camera_count=len(access["train"]))
            else:
                renderer = NativeGaussianRenderer(config["backbone"], source, run / "model", 30000,
                    train_camera_metadata=[cameras[n] for n in access["train"]])
            if asdict(renderer.identity) != proof["identity"]:
                raise ValueError("Deployment identity differs from the adapter acceptance probe")
            signature = renderer.rgb_signature()
            (output / "images").mkdir()
            entries, first_rgb = [], None
            for index, name in enumerate(names):
                camera = cameras[name]
                native = renderer.render_pinhole(camera)
                native.validate(camera)
                if first_rgb is None:
                    first_rgb = native.rgb.copy()
                image = np.rint(native.rgb * 255).astype(np.uint8)
                path = output / "images" / name
                Image.fromarray(image).save(path)
                entries.append({"name": name, "sha256": sha256(path), "width": camera.width, "height": camera.height})
                if index == 0 or (index + 1) % 25 == 0:
                    print(json.dumps({"rendered": index + 1, "total": len(names)}), flush=True)
            if not np.array_equal(first_rgb, renderer.render_pinhole(cameras[names[0]]).rgb) or signature != renderer.rgb_signature():
                raise ValueError("RGB tensors or deterministic render changed during export")
            verify_hashes(args.artifact_root, frozen_hashes)
            verify_hashes(run, parent["artifacts"])
            if source_manifest(source) != sources:
                raise ValueError("Upstream source changed during export")
            verify_hashes(ROOT, record["code_sha256"])
            record.update(status="completed_native_expert_rgb_export", parent_experiment=experiment,
                parent_result_sha256=sha256(run / "result.json"), probe_sha256=sha256(args.probe / "result.json"),
                audit_sha256=sha256(args.audit / "result.json"),
                split_sha256=sha256(split_path), parent_seed=parent["seed"], identity=asdict(renderer.identity),
                rendered_images=entries, frozen_rgb_signature=signature, repeated_first_camera_exact=True,
                semantic_training_completed=False, frozen_verified=len(frozen_hashes))
        except BaseException as exc:
            record.update(status="expert_rgb_export_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            record["export_code_sha256"] = sha256(Path(__file__))
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
