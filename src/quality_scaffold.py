"""Matched-protocol Scaffold adapter; frozen historical pipeline is not modified.

The per-anchor field is inherited by native generated offsets exactly as in the
existing static baseline. External views use the registered appearance index 0.
"""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from src.camera_adapter import Sim3
from src.quality_backbone import ModelIdentity, NativeRender, array_signature
from src.quality_native_gaussian import make_native_view
from src.quality_protocol import sha256
from src.scaffold_semantic_sidecar import (ScaffoldSettings, load_frozen_scaffold,
    frozen_rgb_named_parameters, frozen_generated_geometry, _render_color_channels, render_frozen_rgb, pipeline)


def saved_namespace(path):
    """Read literal saved author arguments without eval of checkpoint text."""
    tree = ast.parse(Path(path).read_text(), mode="eval").body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != "Namespace" or tree.args or any(k.arg is None for k in tree.keywords):
        raise ValueError("Expected a literal Namespace checkpoint configuration")
    return {k.arg: ast.literal_eval(k.value) for k in tree.keywords}


class ScaffoldQualityRenderer:
    def __init__(self, source, model_path, iteration, *, train_camera_count):
        source, model_path = Path(source).resolve(), Path(model_path).resolve()
        for module in ("scene", "gaussian_renderer", "arguments"):
            loaded = sys.modules.get(module)
            if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
                raise RuntimeError("Use a fresh process for each upstream backend")
        saved = saved_namespace(model_path / "cfg_args")
        defaults = asdict(ScaffoldSettings())
        values = {k: saved.get(k, v) for k, v in defaults.items()}
        values["train_camera_count"] = train_camera_count
        if saved.get("white_background") or saved.get("lod", 0) != 0:
            raise ValueError("Unregistered Scaffold background/LOD recipe")
        settings = ScaffoldSettings(**values)
        directory = model_path / "point_cloud" / f"iteration_{iteration}"
        self.model = load_frozen_scaffold(source, directory / "point_cloud.ply", settings)
        self.pipe = pipeline()
        self.official_to_model = Sim3(1., np.eye(3), np.zeros(3))
        files = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.suffix in (".ply", ".pt")}
        recipe = {"artifacts": files, "settings": values, "appearance_index": 0, "official_to_model": "identity"}
        checkpoint = hashlib.sha256(json.dumps(recipe, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.identity = ModelIdentity("scaffold-gs", checkpoint, self.geometry_signature())

    @property
    def field_positions(self):
        return self.model.get_anchor

    def geometry_signature(self):
        return array_signature({n: p.detach().cpu().numpy() for n, p in frozen_rgb_named_parameters(self.model)
                                if not n.startswith(("mlp_color.", "embedding_appearance."))})

    def rgb_signature(self):
        return array_signature({n: p.detach().cpu().numpy() for n, p in frozen_rgb_named_parameters(self.model)})

    def render_channels(self, camera, values):
        import torch
        if camera.model != "PINHOLE" or values.ndim != 2 or values.shape[0] != len(self.field_positions) or values.shape[1] < 1 or not torch.isfinite(values).all():
            raise ValueError("Field must match native frozen Scaffold anchors")
        if values.dtype != self.field_positions.dtype or values.device != self.field_positions.device:
            raise ValueError("Field device/dtype differs from frozen Scaffold")
        view = make_native_view(camera)
        geometry, visible = frozen_generated_geometry(view, self.model, self.pipe)
        colors = values[visible].repeat_interleave(self.model.n_offsets, dim=0)[geometry[4]]
        images = []
        for start in range(0, values.shape[1], 3):
            channels = colors[:, start:start + 3]
            count = channels.shape[1]
            padded = torch.nn.functional.pad(channels, (0, 3 - count)).contiguous()
            images.append(_render_color_channels(geometry, padded, view, torch.zeros(3, device=values.device))[:count])
        return torch.cat(images)

    def render_pinhole(self, camera):
        if camera.model != "PINHOLE":
            raise ValueError("Native camera must be PINHOLE")
        image = render_frozen_rgb(make_native_view(camera), self.model, self.pipe)
        return NativeRender(image.clamp(0, 1).permute(1, 2, 0).cpu().numpy())
