"""Read-only native AbsGS trajectory and point-lineage observations."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np


def tensor_hash(value):
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def rng_hashes():
    import torch

    return {
        "python": hashlib.sha256(
            pickle.dumps(random.getstate(), protocol=4)
        ).hexdigest(),
        "numpy": hashlib.sha256(
            pickle.dumps(np.random.get_state(), protocol=4)
        ).hexdigest(),
        "torch_cpu": tensor_hash(torch.get_rng_state()),
        "torch_cuda": tensor_hash(torch.cuda.get_rng_state()),
    }


class Lineage:
    def __init__(self, count):
        self.ids = np.arange(count, dtype=np.int64)
        self.next_id = count
        self.births = []

    def append(self, parents, step, kind):
        parents = np.asarray(parents, dtype=np.int64)
        ids = np.arange(self.next_id, self.next_id + len(parents), dtype=np.int64)
        self.next_id += len(parents)
        self.ids = np.concatenate((self.ids, ids))
        if len(ids):
            self.births.append(
                (
                    ids,
                    parents.copy(),
                    np.full(len(ids), step, np.int32),
                    np.full(len(ids), 1 if kind == "clone" else 2, np.int8),
                )
            )
        return ids

    def prune(self, mask):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(self.ids):
            raise ValueError("Native prune mask and lineage lengths differ")
        removed = self.ids[mask].copy()
        self.ids = self.ids[~mask]
        return removed


class TrajectoryObserver:
    def __init__(self, config, output, scene):
        self.config, self.output, self.scene = config, Path(output), Path(scene)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "events").mkdir()
        (self.output / "snapshots").mkdir()
        self.step = 0
        self.finished = -1
        self.model = None
        self.lineage = None
        self.pending = None
        self.phase = "none"
        self.packet = None
        self.last_gradient = -1
        self.camera_trace = []
        self.image_hashes = {}
        self.camera_metadata = {}
        self.training_args = {}
        self.journals = {
            name: (self.output / f"{name}.jsonl").open("x")
            for name in ("states", "gradients", "events", "probes")
        }
        self.event_counter = 0
        self.prune_settings = None
        self.checkpoints = {}

    def emit(self, name, row):
        self.journals[name].write(json.dumps(row, allow_nan=False) + "\n")
        self.journals[name].flush()

    def install(self):
        import torch
        import gaussian_renderer as renderer

        cls = renderer.GaussianModel
        original = {
            name: getattr(cls, name)
            for name in (
                "training_setup",
                "update_learning_rate",
                "add_densification_stats",
                "densify_and_clone",
                "densify_and_split",
                "densification_postfix",
                "prune_points",
                "densify_and_prune",
                "reset_opacity",
            )
        }
        self.original_render = renderer.render
        owner = self

        def setup(model, args):
            value = original["training_setup"](model, args)
            owner.model = model
            owner.training_args = vars(args).copy()
            (owner.output / "runtime.json").write_text(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(),
                        "threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                        "optimizer_type": model.optimizer_type,
                        "training_args": owner.training_args,
                        "optimizer_groups": [
                            {k: v for k, v in group.items() if k != "params"}
                            for group in model.optimizer.param_groups
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
            owner.lineage = Lineage(len(model.get_xyz))
            step = model.optimizer.step

            def optimizer_step(*a, **kw):
                owner.capture_gradients()
                return step(*a, **kw)

            model.optimizer.step = optimizer_step
            return value

        def learning_rate(model, iteration):
            owner.finish_step(iteration - 1)
            owner.step = iteration
            return original["update_learning_rate"](model, iteration)

        def render(camera, model, *args, **kwargs):
            packet = owner.original_render(camera, model, *args, **kwargs)
            if torch.is_grad_enabled() and model is owner.model:
                name = camera.image_name
                owner.camera_trace.append({"step": owner.step, "camera": name})
                if name not in owner.image_hashes:
                    owner.image_hashes[name] = tensor_hash(camera.original_image)
                    owner.camera_metadata[name] = {
                        "image_width": camera.image_width,
                        "image_height": camera.image_height,
                        "FoVx": camera.FoVx,
                        "FoVy": camera.FoVy,
                        "world_view_transform": camera.world_view_transform.detach()
                        .cpu()
                        .tolist(),
                        "projection_matrix": camera.projection_matrix.detach()
                        .cpu()
                        .tolist(),
                        "full_proj_transform": camera.full_proj_transform.detach()
                        .cpu()
                        .tolist(),
                        "camera_center": camera.camera_center.detach().cpu().tolist(),
                        "render_options": {
                            k: v
                            for k, v in kwargs.items()
                            if isinstance(v, (str, int, float, bool)) or v is None
                        },
                    }
                owner.packet = packet
            return packet

        def stats(model, point_tensor, visible):
            owner.capture_gradients()
            return original["add_densification_stats"](model, point_tensor, visible)

        def clone(model, grads, threshold, extent):
            mask = (torch.norm(grads, dim=-1) >= threshold) & (
                model.get_scaling.max(1).values <= model.percent_dense * extent
            )
            parents = owner.lineage.ids[mask.detach().cpu().numpy()]
            owner.pending = ("clone", parents)
            owner.phase = "clone"
            try:
                return original["densify_and_clone"](model, grads, threshold, extent)
            finally:
                owner.pending = None
                owner.phase = "none"

        def split(model, grads, threshold, extent, N=2):
            padded = torch.zeros(len(model.get_xyz), device=grads.device)
            padded[: len(grads)] = grads.squeeze()
            mask = (padded >= threshold) & (
                model.get_scaling.max(1).values > model.percent_dense * extent
            )
            chosen = mask.detach().cpu().numpy()
            parents = owner.lineage.ids[chosen]
            owner.pending = ("split", np.tile(parents, N))
            owner.phase = "split"
            normal = torch.normal

            def observed_normal(*a, **kw):
                before = rng_hashes()
                value = normal(*a, **kw)
                owner.emit(
                    "events",
                    {
                        "step": owner.step,
                        "type": "split_random",
                        "parents": len(parents),
                        "rng_before": before,
                        "rng_after": rng_hashes(),
                        "samples_sha256": tensor_hash(value),
                    },
                )
                return value

            torch.normal = observed_normal
            try:
                return original["densify_and_split"](model, grads, threshold, extent, N)
            finally:
                torch.normal = normal
                owner.pending = None
                owner.phase = "none"

        def append(model, *args, **kwargs):
            if owner.pending is None:
                raise ValueError("Unknown native append outside clone/split")
            kind, parents = owner.pending
            count = len(args[0])
            if count != len(parents):
                raise ValueError("Native child count and lineage parent count differ")
            ids = owner.lineage.append(parents, owner.step, kind)
            path = owner.output / "events" / f"{owner.step:05d}-{kind}.npz"
            np.savez_compressed(
                path, ids=ids, parents=parents, xyz=args[0].detach().cpu().numpy()
            )
            value = original["densification_postfix"](model, *args, **kwargs)
            if len(model.get_xyz) != len(owner.lineage.ids):
                raise ValueError("Append lineage shape drift")
            owner.emit(
                "events",
                {
                    "step": owner.step,
                    "type": "append",
                    "kind": kind,
                    "children": count,
                    "max_radii_nonzero_after_append": int(
                        torch.count_nonzero(model.max_radii2D)
                    ),
                    "file": str(path.relative_to(owner.output)),
                },
            )
            return value

        def prune(model, mask):
            reasons = None
            if owner.phase == "none" and owner.prune_settings is not None:
                minimum, extent, screen = owner.prune_settings
                reasons = {
                    "low_opacity": int((model.get_opacity < minimum).sum()),
                    "screen_big_at_actual_prune": int(
                        (model.max_radii2D > screen).sum()
                    )
                    if screen
                    else 0,
                    "world_big_at_actual_prune": int(
                        (model.get_scaling.max(1).values > 0.1 * extent).sum()
                    )
                    if screen
                    else 0,
                }
            removed = owner.lineage.prune(mask.detach().cpu().numpy())
            owner.event_counter += 1
            path = (
                owner.output
                / "events"
                / f"{owner.step:05d}-prune-{owner.event_counter:04d}.npz"
            )
            np.savez_compressed(path, removed_ids=removed)
            value = original["prune_points"](model, mask)
            if len(model.get_xyz) != len(owner.lineage.ids):
                raise ValueError("Prune lineage shape drift")
            owner.emit(
                "events",
                {
                    "step": owner.step,
                    "type": "prune",
                    "phase": owner.phase,
                    "removed": len(removed),
                    "remaining": len(model.get_xyz),
                    "file": str(path.relative_to(owner.output)),
                    "reasons": reasons,
                },
            )
            return value

        def densify(
            model, threshold, absolute_threshold, min_opacity, extent, max_screen, radii
        ):
            owner.prune_settings = (min_opacity, extent, max_screen)
            before_rng = rng_hashes()
            signed = torch.nan_to_num(model.xyz_gradient_accum / model.denom, nan=0.0)
            absolute = torch.nan_to_num(
                model.xyz_gradient_accum_abs / model.denom, nan=0.0
            )
            path = owner.output / "events" / f"{owner.step:05d}-scores.npz"
            np.savez_compressed(
                path,
                ids=owner.lineage.ids,
                signed=signed.detach().cpu().numpy().reshape(-1),
                absolute=absolute.detach().cpu().numpy().reshape(-1),
                scale=model.get_scaling.max(1).values.detach().cpu().numpy(),
                max_radii=model.max_radii2D.detach().cpu().numpy(),
            )
            before = len(model.get_xyz)
            screen_before = (
                int((model.max_radii2D > max_screen).sum()) if max_screen else 0
            )
            owner.emit(
                "events",
                {
                    "step": owner.step,
                    "type": "densify_begin",
                    "count": before,
                    "rng": before_rng,
                    "screen_threshold": max_screen,
                    "screen_big_before_append": screen_before,
                    "file": str(path.relative_to(owner.output)),
                },
            )
            value = original["densify_and_prune"](
                model,
                threshold,
                absolute_threshold,
                min_opacity,
                extent,
                max_screen,
                radii,
            )
            owner.emit(
                "events",
                {
                    "step": owner.step,
                    "type": "densify_end",
                    "count": len(model.get_xyz),
                    "rng": rng_hashes(),
                    "screen_nonzero_after": int(torch.count_nonzero(model.max_radii2D)),
                },
            )
            return value

        def reset(model):
            owner.emit(
                "events",
                {
                    "step": owner.step,
                    "type": "opacity_reset",
                    "before_sha256": tensor_hash(model._opacity),
                },
            )
            return original["reset_opacity"](model)

        cls.training_setup, cls.update_learning_rate = setup, learning_rate
        cls.add_densification_stats = stats
        cls.densify_and_clone, cls.densify_and_split = clone, split
        cls.densification_postfix, cls.prune_points = append, prune
        cls.densify_and_prune, cls.reset_opacity = densify, reset
        renderer.render = render
        self.restore = lambda: self._restore(renderer, cls, original)

    def _restore(self, renderer, cls, original):
        renderer.render = self.original_render
        for key, value in original.items():
            setattr(cls, key, value)

    def capture_gradients(self):
        if self.last_gradient == self.step or self.packet is None:
            return
        self.last_gradient = self.step
        if self.step <= self.config["dense_signature_until"] or self.step % 100 == 0:
            import torch

            model = self.model
            self.emit(
                "gradients",
                {
                    "step": self.step,
                    "count": len(model.get_xyz),
                    "xyz_grad_sha256": tensor_hash(model._xyz.grad),
                    "opacity_grad_sha256": tensor_hash(model._opacity.grad),
                    "screen_grad_sha256": tensor_hash(
                        self.packet["viewspace_points"].grad
                    ),
                    "sampled_rgb_sha256": tensor_hash(
                        self.packet["render"][:, ::16, ::16]
                    ),
                    "rng": rng_hashes(),
                },
            )

    def finish_step(self, step):
        if self.model is None or step <= self.finished:
            return
        self.finished = step
        import torch

        model = self.model
        if (
            step <= self.config["dense_signature_until"]
            or step in self.config["state_steps"]
        ):
            row = {
                "step": step,
                "count": len(model.get_xyz),
                "xyz_sha256": tensor_hash(model._xyz),
                "opacity_sha256": tensor_hash(model._opacity),
                "rng": rng_hashes(),
            }
            if step in self.config["state_steps"]:
                row["parameters"] = {
                    k: tensor_hash(getattr(model, k))
                    for k in (
                        "_xyz",
                        "_features_dc",
                        "_features_rest",
                        "_scaling",
                        "_rotation",
                        "_opacity",
                    )
                }
                row["active_sh_degree"] = model.active_sh_degree
                row["spatial_lr_scale"] = float(model.spatial_lr_scale)
            self.emit("states", row)
        if step not in set(self.config["state_steps"]) | set(
            self.config["probe_steps"]
        ):
            self.packet = None
            return
        before = rng_hashes()
        directory = self.output / "snapshots" / f"step-{step:05d}"
        directory.mkdir(exist_ok=False)
        if step in self.config["state_steps"]:
            model.save_ply(str(directory / "point_cloud.ply"))
            np.save(directory / "ids.npy", self.lineage.ids)
            (directory / "state.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "active_sh_degree": model.active_sh_degree,
                        "count": len(model.get_xyz),
                        "next_lineage_id": self.lineage.next_id,
                        "training_args": self.training_args,
                    },
                    indent=2,
                )
                + "\n"
            )
        if step in self.config["full_checkpoint_steps"]:
            torch.save(
                {
                    "model": model.capture(),
                    "step": step,
                    "training_args": self.training_args,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                directory / "capture.pt",
            )
        if step in self.config["probe_steps"]:
            self.probe(step, directory)
        if rng_hashes() != before:
            raise ValueError("Read-only observation changed global RNG")
        self.packet = None

    def probe(self, step, directory):
        import argparse
        import torch
        from PIL import Image
        from arguments import PipelineParams
        from src.quality_cameras import read_camera_requests
        from src.quality_native_gaussian import make_native_view

        if not hasattr(self, "probe_cameras"):
            cameras = read_camera_requests(self.scene / "sparse/0")
            self.probe_cameras = {
                name: make_native_view(cameras[name])
                for name in self.config["probe_views"]
            }
            self.probe_gt = {
                name: np.asarray(
                    Image.open(self.scene / "images" / name).convert("RGB")
                )
                for name in self.probe_cameras
            }
            parser = argparse.ArgumentParser()
            self.pipeline = PipelineParams(parser).extract(parser.parse_args([]))
        with torch.no_grad():
            for name, camera in self.probe_cameras.items():
                value = self.original_render(
                    camera, self.model, self.pipeline, torch.zeros(3, device="cuda")
                )["render"]
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite probe RGB")
                actual = np.rint(
                    value.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                Image.fromarray(actual).save(directory / name)
                mse = np.mean(
                    ((actual.astype(np.float64) - self.probe_gt[name]) / 255.0) ** 2
                )
                self.emit(
                    "probes",
                    {
                        "step": step,
                        "view": name,
                        "psnr_db": float(-10 * np.log10(max(mse, 1e-30))),
                    },
                )

    def finalize(self):
        self.capture_gradients()
        self.finish_step(self.step)
        if self.lineage.births:
            columns = [
                np.concatenate([r[i] for r in self.lineage.births]) for i in range(4)
            ]
        else:
            columns = [np.zeros(0, np.int64) for _ in range(4)]
        np.savez_compressed(
            self.output / "lineage.npz",
            ids=columns[0],
            parents=columns[1],
            birth_step=columns[2],
            kind=columns[3],
            final_ids=self.lineage.ids,
        )
        (self.output / "camera-trace.json").write_text(
            json.dumps(self.camera_trace) + "\n"
        )
        (self.output / "input-tensor-hashes.json").write_text(
            json.dumps(self.image_hashes, indent=2) + "\n"
        )
        (self.output / "camera-tensor-metadata.json").write_text(
            json.dumps(self.camera_metadata, indent=2) + "\n"
        )
        for stream in self.journals.values():
            stream.close()
