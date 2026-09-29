#!/usr/bin/env python3
"""Inject one registered auxiliary RGB target into the native RGB140 trainer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", default="initial")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = ROOT / config["source"]
    sys.path[:0] = [
        str(source),
        str(source / "submodules/diff-gaussian-rasterization"),
        str(source / "submodules/simple-knn"),
        str(ROOT),
    ]

    import numpy as np
    import torch
    from src.quality_protocol import sha256
    from src.standard_rgb_supervision import AuxLossController

    runtime_config = json.loads(json.dumps(config))
    runtime_config["auxiliary_supervision"]["label_root"] = str(
        ROOT / config["auxiliary_supervision"]["label_root"]
    )
    controller = AuxLossController(runtime_config, args.output, args.attempt)
    camera_path = args.output / "camera_sequence.txt"
    binding = {
        "config_sha256": sha256(args.config),
        "worker_sha256": sha256(Path(__file__)),
        "supervision_sha256": sha256(ROOT / "src/standard_rgb_supervision.py"),
    }

    def resume(model, opt, scene):
        if not args.resume_state:
            return (
                0,
                scene.getTrainCameras().copy(),
                list(range(len(scene.getTrainCameras()))),
                0.0,
                0.0,
            )
        state = torch.load(args.resume_state, weights_only=False)
        if state["binding"] != binding:
            raise ValueError("Resume scientific binding mismatch")
        model._exposure.data.copy_(state["exposure"])
        model.restore(state["model"], opt)
        model.exposure_optimizer.load_state_dict(state["exposure_optimizer"])
        cameras = {
            Path(camera.image_name).name: camera for camera in scene.getTrainCameras()
        }
        stack = [cameras[name] for name in state["viewpoint_stack"]]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        return (
            state["iteration"],
            stack,
            state["viewpoint_indices"],
            state["ema_loss"],
            state["ema_depth"],
        )

    def checkpoint(iteration, model, stack, indices, ema_loss, ema_depth, total):
        scheduled = set(config["training"]["recovery_iterations"])
        if iteration not in scheduled and iteration != total:
            return
        state = {
            "binding": binding,
            "iteration": iteration,
            "model": model.capture(),
            "exposure": model._exposure.detach(),
            "exposure_optimizer": model.exposure_optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "viewpoint_stack": [Path(camera.image_name).name for camera in stack],
            "viewpoint_indices": indices,
            "ema_loss": ema_loss,
            "ema_depth": ema_depth,
        }
        folder = args.output / "recovery"
        folder.mkdir(exist_ok=True)
        destination = folder / f"state-{iteration:05d}.pth"
        temporary = destination.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, destination)
        (folder / "latest.json").write_text(
            json.dumps(
                {"path": str(destination), "iteration": iteration, "binding": binding}
            )
            + "\n"
        )

    def observe_camera(iteration, camera):
        with camera_path.open("a") as stream:
            stream.write(f"{iteration}\t{Path(camera.image_name).name}\n")

    original_run_path = runpy.run_path

    def patched_run_path(path, *run_args, **run_kwargs):
        if Path(path).resolve() != (source / "train.py").resolve():
            return original_run_path(path, *run_args, **run_kwargs)
        text = Path(path).read_text()
        replacements = {
            "        viewpoint_cam = viewpoint_stack.pop(rand_idx)\n        vind = viewpoint_indices.pop(rand_idx)\n": "        viewpoint_cam = viewpoint_stack.pop(rand_idx)\n        vind = viewpoint_indices.pop(rand_idx)\n        rgb_camera_observe(iteration, viewpoint_cam)\n",
            "        loss.backward()\n": "        loss = rgb_aux_apply(iteration, viewpoint_cam, image, gt_image, loss, gaussians)\n        loss.backward()\n",
            '    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")': '    first_iter, viewpoint_stack, viewpoint_indices, ema_loss_for_log, ema_Ll1depth_for_log = rgb_resume(gaussians, opt, scene)\n    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")',
            "            if (iteration in checkpoint_iterations):": "            rgb_checkpoint(iteration, gaussians, viewpoint_stack, viewpoint_indices, ema_loss_for_log, ema_Ll1depth_for_log, opt.iterations)\n\n            if (iteration in checkpoint_iterations):",
        }
        for needle, replacement in replacements.items():
            if text.count(needle) != 1:
                raise RuntimeError(f"Native hook location changed: {needle[:40]}")
            text = text.replace(needle, replacement)
        (args.output / "executed_train.py").write_text(text)
        scope = {
            "__name__": "__main__",
            "__file__": str(path),
            "rgb_aux_apply": controller.apply,
            "rgb_resume": resume,
            "rgb_checkpoint": checkpoint,
            "rgb_camera_observe": observe_camera,
        }
        exec(compile(text, str(path), "exec"), scope)
        return scope

    runpy.run_path = patched_run_path
    worker_args = (
        args.worker_args[1:] if args.worker_args[:1] == ["--"] else args.worker_args
    )
    sys.argv = [str(ROOT / "scripts/backbone_all400_worker.py"), *worker_args]
    try:
        original_run_path(sys.argv[0], run_name="__main__")
    finally:
        controller.finalize()


if __name__ == "__main__":
    main()
