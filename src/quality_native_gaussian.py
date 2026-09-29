"""PGSR/SOF adapter using THEIR native rasterizers, never the Scaffold rasterizer.

Run one backend per fresh Python process: upstream packages have colliding names.
Only RGB and generic primitive channels are implemented here; fitting GAGS and
expert sidecars is a separate stage, not implied by this adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from src.camera_adapter import Sim3
from src.quality_backbone import ModelIdentity, NativeRender, array_signature
from src.quality_protocol import sha256


def projection_matrix(camera, near=.01, far=100.):
    """Column-vector positive-Z projection, COLMAP edge-origin pixel centres."""
    k, w, h = camera.matrix, camera.width, camera.height
    p = np.zeros((4, 4), np.float32)
    p[0, 0], p[1, 1] = 2 * k[0, 0] / w, 2 * k[1, 1] / h
    p[0, 2], p[1, 2] = 2 * k[0, 2] / w - 1, 2 * k[1, 2] / h - 1
    p[2, 2], p[2, 3], p[3, 2] = far / (far - near), -far * near / (far - near), 1
    return p


def make_native_view(camera):
    import torch
    r, t = camera.pose.rotation, camera.pose.translation
    world = np.eye(4, dtype=np.float32)
    world[:3, :3], world[:3, 3] = r, t
    world_view = torch.as_tensor(world, device="cuda").T.contiguous()
    projection = torch.as_tensor(projection_matrix(camera), device="cuda").T.contiguous()
    combined = world_view @ projection
    fx, fy, cx, cy = camera.matrix[0, 0], camera.matrix[1, 1], camera.matrix[0, 2], camera.matrix[1, 2]
    view = SimpleNamespace(image_width=camera.width, image_height=camera.height,
                           FoVx=2 * math.atan(camera.width / (2 * fx)), FoVy=2 * math.atan(camera.height / (2 * fy)),
                           world_view_transform=world_view, projection_matrix=projection,
                           full_proj_transform=combined, full_proj_transform_inverse=combined.inverse(),
                           camera_center=world_view.inverse()[3, :3], uid=0, R=r.T, T=t,
                           Fx=fx, Fy=fy, Cx=cx, Cy=cy, focal_x=fx, focal_y=fy)
    view.get_calib_matrix_nerf = lambda scale=1.: (torch.tensor([[fx / scale, 0, cx / scale], [0, fy / scale, cy / scale], [0, 0, 1]], dtype=torch.float32), world_view.T.contiguous())
    return view


class NativeGaussianRenderer:
    @property
    def field_positions(self):
        return self.model.get_xyz

    def __init__(self, backbone, source, model_path, iteration, *, train_camera_metadata=None):
        if backbone not in ("pgsr", "sof"):
            raise ValueError("Native Gaussian adapter supports PGSR/SOF only")
        source, model_path = Path(source).resolve(), Path(model_path).resolve()
        for module in ("scene", "gaussian_renderer", "arguments"):
            loaded = sys.modules.get(module)
            if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
                raise RuntimeError("Upstream namespace collision; use a fresh process")
        sys.path.insert(0, str(source))
        import torch
        arguments = importlib.import_module("arguments")
        renderer = importlib.import_module("gaussian_renderer")
        self.backbone, self.native_render = backbone, renderer.render
        parser = argparse.ArgumentParser(add_help=False)
        pp = arguments.PipelineParams(parser)
        self.pipe = pp.extract(parser.parse_args([]))
        self.model = renderer.GaussianModel(3)
        ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        self.model.load_ply(str(ply))
        self.settings = None
        if backbone == "sof":
            from diff_gaussian_rasterization import ExtendedSettings
            config = model_path / "config.json"
            if not config.is_file() or train_camera_metadata is None:
                raise ValueError("SOF requires saved sorting config and train-only camera metadata")
            self.settings = ExtendedSettings.from_json(str(config))
            # Match author render.py: filter is recomputed from TRAIN cameras,
            # never from target cameras or their RGB. This state is fingerprinted.
            views = []
            for camera in train_camera_metadata:
                if camera.model != "PINHOLE":
                    raise ValueError("SOF filter metadata must be native PINHOLE")
                views.append(make_native_view(camera))
            if not views:
                raise ValueError("Empty SOF filter camera set")
            self.model.compute_3D_filter(views)
        for value in vars(self.model).values():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.requires_grad_(False)
        self.background = torch.zeros(3, device="cuda")
        self.official_to_model = Sim3(1., np.eye(3), np.zeros(3))
        deployment = {"ply_sha256": sha256(ply), "pipe": vars(self.pipe),
                      "settings": None if self.settings is None else self.settings.to_dict(),
                      "official_to_model": "identity", "filter_geometry_sha256": self.geometry_signature()}
        # Bind auxiliary fields to the complete deployment recipe, not just PLY.
        checkpoint = hashlib.sha256(json.dumps(deployment, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.identity = ModelIdentity(backbone, checkpoint, self.geometry_signature())

    def geometry_signature(self):
        names = ("_xyz", "_scaling", "_rotation", "_opacity", "filter_3D")
        return array_signature({n: getattr(self.model, n).detach().cpu().numpy() for n in names if hasattr(self.model, n)})

    def rgb_signature(self):
        names = ("_xyz", "_scaling", "_rotation", "_opacity", "filter_3D", "_features_dc", "_features_rest")
        tensors = {n: getattr(self.model, n).detach().cpu().numpy() for n in names if hasattr(self.model, n)}
        tensors["active_sh_degree"] = np.array(self.model.active_sh_degree, dtype=np.int64)
        return array_signature(tensors)

    def _render(self, view, colors=None):
        kwargs = {"override_color": colors}
        if self.backbone == "pgsr":
            kwargs.update(return_plane=True, return_depth_normal=False)
        else:
            kwargs["splat_args"] = self.settings
        return self.native_render(view, self.model, self.pipe, self.background, **kwargs)

    def render_channels(self, camera, values):
        """N×D per-primitive fields via native 3-channel passes; gradients to values.

        No independent sorting, splat approximation or substitute geometry. The
        caller owns feature normalization; these are raw composited channels.
        """
        import torch
        if camera.model != "PINHOLE":
            raise ValueError("Feature rasterization requires native camera")
        if values.ndim != 2 or values.shape[0] != self.model.get_xyz.shape[0] or values.shape[1] < 1 or not torch.isfinite(values).all():
            raise ValueError("Feature field does not match native primitives")
        if values.device != self.model.get_xyz.device or values.dtype != self.model.get_xyz.dtype:
            raise ValueError("Feature field device/dtype differs from native geometry")
        view, images = make_native_view(camera), []
        for start in range(0, values.shape[1], 3):
            channels = values[:, start:start + 3]
            count = channels.shape[1]
            colors = torch.nn.functional.pad(channels, (0, 3 - count)).contiguous()
            images.append(self._render(view, colors)["render"][:count])
        return torch.cat(images)

    def render_pinhole(self, camera):
        import torch
        if camera.model != "PINHOLE":
            raise ValueError("Native RGB input must be PINHOLE")
        with torch.no_grad():
            raw = self._render(make_native_view(camera))
        rgb = raw["render"][:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        depth, normal = None, None
        if self.backbone == "pgsr":
            depth = raw["plane_depth"].squeeze(0).cpu().numpy()
            normal = raw["rendered_normal"].cpu().numpy()
        return NativeRender(rgb=rgb, depth=depth, normals=normal)
