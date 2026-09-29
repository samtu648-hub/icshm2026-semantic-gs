#!/usr/bin/env python3
"""Verify native PGSR/SOF RGB reload, official camera and trainable field contract.

Run in the corresponding isolated upstream PYTHONPATH after native training.
This is an engineering test, not trained semantic or GAGS quality evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    args = p.parse_args()
    run, source, output = args.run.resolve(), args.source.resolve(), args.output.resolve()
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "completed_native_training_pending_independent_reload_and_evaluation":
        raise ValueError("Requires successful native training")
    with (run.parent.parent / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        record = {"experiment_id": result["experiment_id"], "run_result_sha256": sha256(run / "result.json"),
                  "status": "initializing", "semantic_quality_claim": None, "feature_method": "untrained engineering probe"}
        started = time.monotonic()
        try:
            import numpy as np
            import torch
            from PIL import Image
            from src.quality_backbone import array_signature, render_official
            from src.quality_cameras import read_camera_requests
            from src.quality_native_gaussian import NativeGaussianRenderer
            from scripts.run_quality_rgb import source_manifest
            verify_hashes(run, result["artifacts"])
            if source_manifest(source) != json.loads((run / "source_manifest.json").read_text()):
                raise ValueError("Source or native extension changed since training")
            config_path = ROOT / "experiments/runs" / result["experiment_id"] / "config.yaml"
            if sha256(config_path) != result["config_sha256"]:
                raise ValueError("Configuration changed since training")
            config = json.loads(config_path.read_text())
            protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
            frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
            frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
            verify_hashes(args.artifact_root, frozen_hashes)
            steps = protocol["rgb"]["smoke_iterations" if result["mode"] == "smoke" else "iterations"]
            access = json.loads((run / "access.json").read_text())
            cameras = read_camera_requests(run / "scene/sparse/0")
            if set(cameras) != set(access["train"]) | set(access["evaluation"]):
                raise ValueError("Camera metadata universe differs from recorded access")
            train_cameras = [cameras[n] for n in access["train"]]
            target_name = sorted(access["evaluation"])[0]
            camera = cameras[target_name]
            if config["backbone"] == "scaffold-gs":
                from src.quality_scaffold import ScaffoldQualityRenderer
                renderer = ScaffoldQualityRenderer(source, run / "model", steps, train_camera_count=len(train_cameras))
            else:
                renderer = NativeGaussianRenderer(config["backbone"], source, run / "model", steps, train_camera_metadata=train_cameras)
            signature = renderer.rgb_signature()
            before = renderer.render_pinhole(camera)
            before.validate(camera)
            Image.fromarray(np.rint(before.rgb * 255).astype(np.uint8)).save(output / "rgb_before.png")
            # Include signed values: latent features are not limited to RGB range.
            n = renderer.field_positions.shape[0]
            constants = torch.linspace(-.5, .5, 16, device="cuda")
            values = constants.repeat(n, 1).detach().requires_grad_()
            rendered = renderer.render_channels(camera, values)
            with torch.no_grad():
                alpha = renderer.render_channels(camera, torch.ones((n, 1), device="cuda"))
                expected = constants[:, None, None] * alpha
                linear_error = float((rendered - expected).abs().max())
            if not torch.isfinite(rendered).all() or linear_error > 2e-5:
                raise ValueError(f"Native feature compositor failed linearity: {linear_error}")
            rendered.square().mean().backward()
            if values.grad is None or not torch.isfinite(values.grad).all() or not values.grad.abs().sum() > 0:
                raise ValueError("Missing/non-finite feature gradients")
            if any(v.requires_grad or v.grad is not None for v in vars(renderer.model).values() if isinstance(v, torch.Tensor)):
                raise ValueError("Frozen reconstruction acquired gradients")
            with torch.no_grad():
                values -= .01 * values.grad
            after = renderer.render_pinhole(camera)
            if signature != renderer.rgb_signature() or not np.array_equal(before.rgb, after.rgb):
                raise ValueError("Feature update modified frozen RGB")
            # A nonzero-distortion, non-centred target exercises the public adapter,
            # without using a real image or claiming independent geometric truth.
            k = camera.matrix.copy()
            k[0, 2] += 5
            k[1, 2] -= 3
            external = replace(camera, model="SIMPLE_RADIAL", radial_k=.01, matrix=k)
            adapted = render_official(renderer, external)
            Image.fromarray(adapted["rgb"]).save(output / "external_rgb.png")
            verify_hashes(args.artifact_root, frozen_hashes)
            verify_hashes(run, result["artifacts"])
            record.update(status="passed_native_adapter_engineering_probe", identity=asdict(renderer.identity),
                          target_camera=target_name, filter_metadata_count=len(train_cameras),
                          primitive_count=n, feature_channels=16, feature_linearity_max_abs=linear_error,
                          feature_gradients_finite_nonzero=True, frozen_rgb_tensors_exact=True, frozen_rgb_pixels_exact=True,
                          rgb_signature=signature, render_signature=array_signature({"rgb": before.rgb}),
                          official_metadata_input_tested=True, native_depth_is_proxy=True, frozen_verified=len(frozen_hashes),
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        except BaseException as exc:
            record.update(status="adapter_probe_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            record["code_sha256"] = {n: sha256(ROOT / n) for n in
                ("scripts/probe_quality_adapter.py", "src/quality_native_gaussian.py", "src/quality_scaffold.py", "src/quality_backbone.py", "src/quality_cameras.py", "src/camera_adapter.py", "src/scaffold_semantic_sidecar.py")}
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
