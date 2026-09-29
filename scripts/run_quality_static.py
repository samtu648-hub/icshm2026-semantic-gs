#!/usr/bin/env python3
"""Matched static-semantic training on accepted frozen native RGB models.

CPU preflight decodes TRAIN labels only. Smoke/full use the shared nonblocking
GPU lock. A completed fit exports predictions without opening target labels;
independent semantic evaluation and OOF calibration are separate operations.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_quality_rgb import source_manifest
from scripts.run_quality_stage_a import verify_audit, verify_run
from src.quality_protocol import sha256, verify_hashes, write_new_json
from src.quality_static_training import TrainingLabels, fit_static, median_frequency_weights, semantic_access

CODE = ("scripts/run_quality_static.py", "src/quality_static_training.py", "src/quality_semantic_fields.py",
        "scripts/run_quality_rgb.py", "scripts/run_quality_stage_a.py", "src/quality_protocol.py",
        "src/quality_scaffold.py", "src/quality_native_gaussian.py", "src/quality_cameras.py",
        "src/quality_backbone.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py")


def verify_smoke(path, binding, access, code):
    result = json.loads((path / "result.json").read_text())
    if (result.get("status") != "passed_static_semantic_smoke" or result.get("binding") != binding
            or result.get("access") != access or result.get("code_sha256") != code):
        raise ValueError("Full static training requires a matching successful smoke, not another parent/fold/recipe")
    verify_hashes(path, result["artifacts"])
    return sha256(path / "result.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--extra-pythonpath", type=Path, action="append", default=[],
                   help="Same ordered auxiliary sources as RGB parent, verified before import")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("preflight", "smoke", "full"), required=True)
    p.add_argument("--partition", choices=("development", "oof-0", "oof-1", "oof-2"), required=True)
    p.add_argument("--smoke", type=Path, help="Exact separate smoke result directory, required for full")
    args = p.parse_args()
    if (args.mode == "full") != (args.smoke is not None):
        p.error("--smoke is required for full only")
    run, source, output = args.run.resolve(), args.source.resolve(), args.output.resolve()
    lock_path = ROOT / "outputs/quality/gpu.lock"
    with (lock_path.open("a") if args.mode != "preflight" else nullcontext()) as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        record = {"status": "initializing", "experiment_id": "SEM-330-QUALITY-STATIC", "mode": args.mode,
                  "code_sha256": {name: sha256(ROOT / name) for name in CODE},
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "target_labels_decoded": False, "real_rgb_decoded": False, "quality_metrics": None,
                  "trained_steps": 0, "artifacts": {}}
        started = time.monotonic()
        try:
            config_path = ROOT / "experiments/runs/SEM-330-QUALITY-STATIC/config.yaml"
            config = json.loads(config_path.read_text())
            protocol_path = ROOT / config["protocol"]
            protocol = json.loads(protocol_path.read_text())
            input_hashes = {**protocol["input_hashes"], str(config_path.relative_to(ROOT)): sha256(config_path),
                            config["protocol"]: sha256(protocol_path), config["label_manifest"]: sha256(ROOT / config["label_manifest"])}
            verify_hashes(ROOT, input_hashes)
            if shutil.disk_usage(output).free / 2**30 < protocol["execution"]["minimum_free_gib"]:
                raise RuntimeError("Less than registered free-disk reserve; no silent budget/capacity reduction")
            parent_result = json.loads((run / "result.json").read_text())
            experiment = parent_result["experiment_id"]
            if experiment not in config["allowed_rgb_parents"] or parent_result["seed"] != config["parent_rgb_seed"]:
                raise ValueError("Unregistered parent or RGB trial seed")
            parent = verify_run(run, mode="full", experiment_id=experiment,
                common_ply_sha256=sha256(args.artifact_root / protocol["rgb"]["scene"] / "sparse/0/points3D.ply"))
            audit = verify_audit(args.audit, run, "completed_rgb_reload_audit")
            if not audit.get("fresh_reloads_exact") or audit.get("cameras") != 74:
                raise ValueError("RGB independent full reload audit not passed")
            proof = verify_audit(args.probe, run, "passed_native_adapter_engineering_probe")
            changed_probe_code = []
            for name, digest in proof["code_sha256"].items():
                path = ROOT / name
                if not path.resolve().is_relative_to(ROOT):
                    raise ValueError("Probe dependency path escapes project")
                if sha256(path) != digest:
                    changed_probe_code.append(name)
            # A CPU data audit can still check training-only labels, but cannot
            # certify a stale native probe or authorize any GPU fitting.
            record["native_probe_code_changed"] = changed_probe_code
            if changed_probe_code and args.mode != "preflight":
                raise ValueError("Re-run the native adapter probe with current code before semantic smoke/full")
            sources = source_manifest(source)
            if sources != json.loads((run / "source_manifest.json").read_text()):
                raise ValueError("Native source/extensions changed since RGB training")
            if len(args.extra_pythonpath) != len(parent.get("extra_sources", [])):
                raise ValueError("Supply the parent's exact auxiliary source list")
            extras = []
            for extra, entry in zip(args.extra_pythonpath, parent.get("extra_sources", [])):
                manifest_path = run / entry["manifest"]
                if not manifest_path.resolve().is_relative_to(run) or sha256(manifest_path) != entry["manifest_sha256"]:
                    raise ValueError("Auxiliary source manifest differs from RGB parent")
                extra = extra.resolve()
                snapshot = source_manifest(extra)
                if extra.name != entry["name"] or snapshot != json.loads(manifest_path.read_text()):
                    raise ValueError("Auxiliary source/extension bytes changed since RGB training")
                extras.append((extra, snapshot))
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            frozen_hashes = {item["path"]: item["sha256"] for item in frozen["artifacts"]}
            verify_hashes(args.artifact_root, frozen_hashes)
            parent_access = json.loads((run / "access.json").read_text())
            split = json.loads((ROOT / "experiments/splits/sem_001_scaffold_static_v1.json").read_text())
            oof = json.loads((ROOT / "experiments/splits/sem_005_oof_spatial_v1.json").read_text())
            access = semantic_access(split, oof, parent_access, args.partition)
            manifest = json.loads((ROOT / config["label_manifest"]).read_text())
            if sha256(ROOT / config["label_manifest"]) != oof["label_manifest_sha256"]:
                raise ValueError("Label provenance differs from registered OOF split")
            labels = TrainingLabels(args.artifact_root / config["labels"], access["train"], manifest, config["resolution"])
            counts = labels.audit_counts()
            weights = median_frequency_weights(counts)
            binding = {"parent_experiment": experiment, "parent_result_sha256": sha256(run / "result.json"),
                       "audit_result_sha256": sha256(args.audit / "result.json"),
                       "probe_result_sha256": sha256(args.probe / "result.json"), "identity": proof["identity"],
                       "config_sha256": sha256(config_path), "input_hashes": input_hashes,
                       "seed": config["seed"], "source_manifest_sha256": sha256(run / "source_manifest.json"),
                       "extra_sources": parent.get("extra_sources", []),
                       "runtime": {"python": sys.version, "numpy": version("numpy"), "torch": version("torch")}}
            record.update(binding=binding, access=access, train_class_counts=counts.tolist(), class_weights=weights.tolist(),
                          label_hashes={n: labels.records[n]["label_png_sha256"] for n in access["train"]},
                          frozen_verified=len(frozen_hashes))
            for name in (*CODE, str(config_path.relative_to(ROOT))):
                destination = output / "code" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
                if sha256(destination) != sha256(ROOT / name):
                    raise ValueError("Execution source snapshot copy mismatch")
            write_new_json(output / "started.json", record)
            write_new_json(output / "access.json", access)
            if args.mode == "preflight":
                record["status"] = ("completed_cpu_label_preflight_pending_native_probe_refresh" if changed_probe_code
                                    else "passed_static_semantic_cpu_preflight")
            else:
                if args.mode == "full":
                    record["smoke_result_sha256"] = verify_smoke(args.smoke, binding, access, record["code_sha256"])
                backbone = proof["identity"]["backbone"]
                extension = "diff-plane-rasterization" if backbone == "pgsr" else "diff-gaussian-rasterization"
                sys.path[:0] = [str(p) for p in (source, source / "submodules" / extension,
                                                source / "submodules/simple-knn", *[e for e, _ in extras])]
                import numpy as np
                import torch
                from PIL import Image
                from src.quality_cameras import read_camera_requests
                from src.quality_native_gaussian import NativeGaussianRenderer
                from src.quality_scaffold import ScaffoldQualityRenderer
                from src.quality_semantic_fields import PrimitiveSemanticField
                torch.manual_seed(config["seed"])
                np.random.seed(config["seed"])
                random.seed(config["seed"])
                torch.cuda.reset_peak_memory_stats()
                cameras = read_camera_requests(run / "scene/sparse/0")
                if set(cameras) != set(parent_access["train"]) | set(parent_access["evaluation"]):
                    raise ValueError("Camera metadata differs from RGB access")
                if any((c.width, c.height) != tuple(config["resolution"]) or c.model != "PINHOLE" for c in cameras.values()):
                    raise ValueError("Unexpected camera grid; do not resize labels to hide mismatch")
                if backbone == "scaffold-gs":
                    renderer = ScaffoldQualityRenderer(source, run / "model", 30000, train_camera_count=len(parent_access["train"]))
                else:
                    renderer = NativeGaussianRenderer(backbone, source, run / "model", 30000,
                        train_camera_metadata=[cameras[n] for n in parent_access["train"]])
                if asdict(renderer.identity) != proof["identity"]:
                    raise ValueError("Loaded deployment model differs from certified RGB parent")
                signature = renderer.rgb_signature()
                before = {name: renderer.render_pinhole(cameras[name]).rgb.copy() for name in config["frozen_probe_views"]}
                field = PrimitiveSemanticField(renderer)
                budget = config["training"]
                steps = budget["steps" if args.mode == "full" else "smoke_steps"]
                def progress(step, name, loss):
                    record["trained_steps"] = step
                    if step == 1 or step % budget["log_every"] == 0 or step == steps:
                        print(json.dumps({"step": step, "total": steps, "train_view": name, "loss": loss,
                                          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
                fit_started = time.monotonic()
                record["training"] = fit_static(renderer, field, {n: cameras[n] for n in access["train"]}, labels, weights,
                    steps=steps, seed=config["seed"], learning_rate=budget["learning_rate"], weight_decay=budget["weight_decay"], on_step=progress)
                torch.cuda.synchronize()
                record["training_seconds"] = time.monotonic() - fit_started
                checkpoint = output / "semantic_field.pt"
                with checkpoint.open("xb") as stream:
                    torch.save({"binding": binding, "access": access, "trained_steps": steps, "field": field.state_dict()}, stream)
                record["artifacts"][checkpoint.name] = sha256(checkpoint)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if payload["binding"] != binding or payload["access"] != access or payload["trained_steps"] != steps:
                    raise ValueError("Saved semantic attachment provenance changed")
                reloaded = PrimitiveSemanticField(renderer)
                reloaded.load_state_dict(payload["field"], renderer)
                if not torch.equal(field.logits, reloaded.logits):
                    raise ValueError("Semantic checkpoint reload not exact")
                with torch.no_grad():
                    probe_camera = cameras[config["frozen_probe_views"][0]]
                    if not torch.equal(field.render_probabilities(renderer, probe_camera), reloaded.render_probabilities(renderer, probe_camera)):
                        raise ValueError("Reloaded semantic render differs")
                    del field, payload
                    targets = access["prediction"] if args.mode == "full" else access["prediction"][:1]
                    (output / "predictions").mkdir()
                    (output / "probabilities").mkdir()
                    for name in targets:
                        probability = reloaded.render_probabilities(renderer, cameras[name]).cpu().numpy()
                        if (probability.shape != (5, 989, 1320) or not np.isfinite(probability).all()
                                or np.any(probability < 0) or not np.allclose(probability.sum(0), 1, atol=2e-5, rtol=0)):
                            raise ValueError("Invalid native prediction probability")
                        png = output / "predictions" / name
                        npz = output / "probabilities" / (Path(name).stem + ".npz")
                        Image.fromarray(probability.argmax(0).astype(np.uint8)).save(png)
                        np.savez_compressed(npz, probabilities=probability)
                        for path in (png, npz):
                            record["artifacts"][str(path.relative_to(output))] = sha256(path)
                if renderer.rgb_signature() != signature or any(not np.array_equal(rgb, renderer.render_pinhole(cameras[n]).rgb) for n, rgb in before.items()):
                    raise ValueError("Semantic training/export changed frozen RGB tensors or fixed-view pixels")
                record.update(status="completed_static_fit_pending_independent_evaluation" if args.mode == "full" else "passed_static_semantic_smoke",
                              frozen_rgb_signature=signature, fixed_rgb_views_exact=list(before), checkpoint_same_process_reload_exact=True,
                              prediction_views=targets, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, torch_version=str(torch.__version__))
            verify_hashes(args.artifact_root, frozen_hashes)
            verify_hashes(run, parent["artifacts"])
            verify_hashes(labels.root, record["label_hashes"])
            verify_hashes(ROOT, input_hashes)
            verify_hashes(ROOT, record["code_sha256"])
            if source_manifest(source) != sources:
                raise ValueError("Native source changed during static fitting/preflight")
            if any(source_manifest(extra) != snapshot for extra, snapshot in extras):
                raise ValueError("Auxiliary source changed during static fitting/preflight")
            record["training_label_views_decoded"] = sorted(labels.accessed)
        except BaseException as exc:
            record.update(status="static_semantic_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "steps": record["trained_steps"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
