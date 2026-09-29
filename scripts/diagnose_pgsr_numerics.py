#!/usr/bin/env python3
"""Read-only checkpoint inventory and bounded native backward diagnostic.

The diagnostic creates an isolated output directory. It is NOT full training or
an exact resumed experiment (the upstream checkpoint omits sampler/RNG state).
No nonfinite values are removed or replaced. Parent artifacts remain read-only.
"""
from __future__ import annotations
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quality_protocol import sha256, verify_hashes, write_new_json


def inventory(path):
    import numpy as np
    from plyfile import PlyData
    vertices = PlyData.read(path)["vertex"].data
    bad_rows = np.zeros(len(vertices), dtype=bool)
    fields = {}
    for name in vertices.dtype.names:
        values = vertices[name]
        bad = ~np.isfinite(values)
        bad_rows |= bad
        if bad.any():
            fields[name] = {"nan": int(np.isnan(values).sum()), "positive_inf": int(np.isposinf(values).sum()),
                            "negative_inf": int(np.isneginf(values).sum())}
    return {"sha256": sha256(path), "primitive_count": len(vertices), "nonfinite_primitives": int(bad_rows.sum()),
            "nonfinite_fields": fields}


class DiagnosticLimit(Exception):
    pass


def worker(output, config, source):
    import torch
    import numpy as np
    from numpy.core.multiarray import scalar as numpy_scalar
    current = {}
    graph = {}
    original_backward = torch.Tensor.backward
    original_step = torch.optim.Adam.step
    steps = 0
    def backward(tensor, *args, **kwargs):
        local = inspect.currentframe().f_back.f_locals
        current.update(iteration=int(local["iteration"]), camera=local["viewpoint_cam"].image_name,
                       losses={name: (float(value.detach()) if torch.isfinite(value).all() else None)
                               for name in ("loss", "image_loss", "normal_loss", "geo_loss", "ncc_loss")
                               if isinstance(value := local.get(name), torch.Tensor)})
        graph.clear()
        graph.update({name: local[name] for name in ("image_loss", "normal_loss", "geo_loss", "ncc_loss", "min_scale_loss")
                      if isinstance(local.get(name), torch.Tensor)})
        write_new_json(output / f"before-backward-{current['iteration']}.json", current)
        # Retain only the current graph so a first bad total gradient can be
        # decomposed without rerendering, changing camera, or changing weights.
        return original_backward(tensor, *args, retain_graph=True, **kwargs)

    def step(optimizer, *args, **kwargs):
        nonlocal steps
        # Inspect before ANY optimizer can consume a bad gradient.
        for group in optimizer.param_groups:
            for value in group["params"]:
                if value.grad is not None and not torch.isfinite(value.grad).all():
                    current["bad_gradient_group"] = group.get("name")
                    current["bad_gradient_counts"] = {"nan": int(torch.isnan(value.grad).sum()),
                        "inf": int(torch.isinf(value.grad).sum())}
                    names, parameters = [], []
                    for param_group in optimizer.param_groups:
                        for param in param_group["params"]:
                            names.append(param_group.get("name"))
                            parameters.append(param)
                    components = {}
                    for loss_name, component in graph.items():
                        # Native train.py calls Adam inside torch.no_grad().
                        # Re-enable only construction of this diagnostic scalar.
                        with torch.enable_grad():
                            gradients = torch.autograd.grad(component.mean(), parameters, retain_graph=True, allow_unused=True)
                        components[loss_name] = {name: {"nan": int(torch.isnan(grad).sum()), "inf": int(torch.isinf(grad).sum()),
                            "finite_abs_max": float(grad[torch.isfinite(grad)].abs().max()) if torch.isfinite(grad).any() else None}
                            for name, grad in zip(names, gradients) if grad is not None}
                    current["component_gradients"] = components
                    raise FloatingPointError(f"Nonfinite gradient before Adam: {group.get('name')}")
        result = original_step(optimizer, *args, **kwargs)
        if any(group.get("name") == "xyz" for group in optimizer.param_groups):
            steps += 1
            for group in optimizer.param_groups:
                for value in group["params"]:
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite parameter after Adam: {group.get('name')}")
            if steps >= config["maximum_diagnostic_steps"]:
                raise DiagnosticLimit("Registered diagnostic step limit reached")
        return result

    torch.Tensor.backward = backward
    torch.optim.Adam.step = step
    torch.autograd.set_detect_anomaly(True, check_nan=True)
    parent = ROOT / config["parent_run"]
    parent_config = json.loads((ROOT / "experiments/runs/RGB-111-PGSR/config.yaml").read_text())
    sys.argv = [str(ROOT / "scripts/quality_upstream_worker.py"), "--source", str(source),
                "--access", str(parent / "access.json"), "--evidence", str(output / "native_worker.json"),
                "--seed", str(config["seed"]), "--", "--source_path", str(output / "scene"),
                "--model_path", str(output / "model"), "--eval", "--resolution", "1", "--iterations", "30000",
                "--test_iterations", "30000", "--save_iterations", "30000", "--checkpoint_iterations", "30000",
                "--quiet", "--detect_anomaly", "--data_device", "cpu", "--start_checkpoint",
                str(parent / f"model/chkpnt{config['resume_iteration']}.pth"), *parent_config["training_arguments"]]
    record = {"status": "running", "quality_claim": None}
    try:
        # Parent is our hash-verified local checkpoint. Keep weights_only=True;
        # explicitly allow only its NumPy scalar scale metadata, not arbitrary pickle.
        with torch.serialization.safe_globals([(numpy_scalar, "numpy.core.multiarray.scalar"),
                                                np.dtype, type(np.dtype("float64"))]):
            runpy.run_path(str(ROOT / "scripts/quality_upstream_worker.py"), run_name="__main__")
        raise RuntimeError("Diagnostic unexpectedly completed full upstream training")
    except DiagnosticLimit:
        record["status"] = "diagnostic_limit_reached_no_nonfinite_observed"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        record.update(status="diagnostic_exception" if current else "diagnostic_setup_failed",
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        record.update(last_observation=current, completed_optimizer_steps=steps,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        write_new_json(output / "backward_result.json", record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--cuda", type=Path)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--config", type=Path, default=ROOT / "experiments/runs/RGB-114-PGSR-NUMERIC-DIAG/config.yaml")
    args = p.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output, source = args.output.resolve(), args.source.resolve()
    if args.worker:
        worker(output, config, source)
        return
    if args.artifact_root is None or args.cuda is None:
        p.error("--artifact-root and --cuda are required for orchestrator")
    from scripts.run_quality_rgb import materialize_scene, source_manifest
    parent = ROOT / config["parent_run"]
    parent_record = json.loads((parent / "result.json").read_text())
    with (parent.parent.parent / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"experiment_id": config["experiment_id"], "config_sha256": sha256(config_path),
                  "parent_result_sha256": sha256(parent / "result.json"), "seed": config["seed"],
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "input_hashes": parent_record["inputs"], "status": "running", "quality_claim": None}
        protocol = json.loads((ROOT / "experiments/runs/QUAL-001-FOUNDATION/config.yaml").read_text())
        frozen = json.loads((ROOT / protocol["frozen_manifest"]).read_text())
        frozen_hashes = {a["path"]: a["sha256"] for a in frozen["artifacts"]}
        try:
            verify_hashes(parent, parent_record["artifacts"])
            verify_hashes(ROOT, parent_record["inputs"])
            verify_hashes(args.artifact_root, frozen_hashes)
            actual_source = source_manifest(source)
            parent_source = json.loads((parent / "source_manifest.json").read_text())
            changed = sorted(n for n in actual_source.keys() | parent_source.keys() if actual_source.get(n) != parent_source.get(n))
            if changed != config.get("allowed_changed_source_files", []):
                raise ValueError(f"Unexpected diagnostic source changes: {changed}")
            if config.get("source_delta_patch"):
                patch = ROOT / config["source_delta_patch"]
                subprocess.run(["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)], check=True)
                record["source_delta_patch_sha256"] = sha256(patch)
            record["source_manifest"] = actual_source
            record["checkpoint_inventory"] = {str(step): inventory(parent / f"model/point_cloud/iteration_{step}/point_cloud.ply")
                                              for step in config["checkpoint_iterations"]}
            write_new_json(output / "inventory.json", record["checkpoint_inventory"])
            if record["checkpoint_inventory"][str(config["resume_iteration"])]["nonfinite_primitives"]:
                raise ValueError("Diagnostic start checkpoint is not finite")
            materialize_scene(parent / "scene", output / "scene", "pgsr")
            step = config["resume_iteration"]
            shutil.copytree(parent / f"model/app_model/iteration_{step}", output / f"model/app_model/iteration_{step}")
            extras = [source, source / "submodules/diff-plane-rasterization", source / "submodules/simple-knn",
                      ROOT, source.parent / "pytorch3d"]
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, extras)), CUDA_HOME=str(args.cuda),
                       OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
            with (output / "backward.log").open("x") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--source", str(source),
                                "--output", str(output), "--config", str(config_path)], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            record["backward_diagnostic"] = json.loads((output / "backward_result.json").read_text())
            verify_hashes(parent, parent_record["artifacts"])
            verify_hashes(args.artifact_root, frozen_hashes)
            record.update(status=("checkpoint_inventory_completed_backward_setup_failed"
                                  if record["backward_diagnostic"]["status"] == "diagnostic_setup_failed"
                                  else "completed_numeric_diagnostic_not_quality_experiment"), frozen_verified=len(frozen_hashes))
        except BaseException as exc:
            record.update(status="numeric_diagnostic_execution_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            record["code_sha256"] = {n: sha256(ROOT / n) for n in
                ("scripts/diagnose_pgsr_numerics.py", "scripts/quality_upstream_worker.py", "scripts/run_quality_rgb.py", "src/quality_protocol.py", "src/quality_pgsr_geometry.py")}
            write_new_json(output / "result.json", record)
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
