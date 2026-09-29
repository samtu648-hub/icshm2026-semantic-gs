"""Trainable fields attached to frozen *native* PGSR/SOF primitives.

These are reusable fitting components, not pretrained experts or a reproduced
GAGS granularity mechanism. The experiment runner owns split access, losses,
training duration, calibration and frozen-RGB acceptance.
"""
from dataclasses import asdict, replace

from src.quality_backbone import validate_attachment
from src.quality_protocol import CLASSES


def attachment_metadata(renderer, kind):
    return {"schema_version": 1, **asdict(renderer.identity), "classes": list(CLASSES),
            "geometry_frozen": True, "pixel_space": "PINHOLE", "field_kind": kind}


class PrimitiveSemanticField:
    """Five logits per native primitive, with transparent residual = background."""
    def __init__(self, renderer):
        import torch
        xyz = renderer.field_positions
        self.logits = torch.nn.Parameter(torch.zeros((xyz.shape[0], 5), device=xyz.device, dtype=xyz.dtype))
        self.metadata = attachment_metadata(renderer, "static_fiveclass_per_primitive")

    def parameters(self):
        return [self.logits]

    def render_probabilities(self, renderer, camera):
        import torch
        validate_attachment(self.metadata, renderer.identity)
        rendered = renderer.render_channels(camera, torch.softmax(self.logits, dim=1))
        # Native background is black for generic channels. The unconsumed
        # transmittance goes to class 0, matching the historical static convention.
        mass = rendered.sum(dim=0, keepdim=True)
        tail = torch.zeros_like(rendered[1:])
        residual = torch.cat(((1 - mass).clamp_min(0), tail), dim=0)
        output = (rendered + residual).clamp_min(0)
        return output / output.sum(dim=0, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)

    def state_dict(self):
        return {"metadata": dict(self.metadata), "logits": self.logits.detach().cpu()}

    def load_state_dict(self, payload, renderer):
        import torch
        validate_attachment(payload["metadata"], renderer.identity)
        value = payload["logits"]
        if payload["metadata"].get("field_kind") != self.metadata["field_kind"] or not isinstance(value, torch.Tensor) or value.shape != self.logits.shape or not torch.isfinite(value).all():
            raise ValueError("Invalid static semantic field attachment")
        with torch.no_grad():
            self.logits.copy_(value.to(self.logits))


class PrimitiveFeatureField:
    """16-channel native feature capacity plus 16→128→5 head, before distillation.

Fixed- and learned-granularity training must share this capacity. Teacher feature
preparation and the learned granularity objective are deliberately not fabricated.
"""
    def __init__(self, renderer):
        import torch
        xyz = renderer.field_positions
        self.features = torch.nn.Parameter(torch.zeros((xyz.shape[0], 16), device=xyz.device, dtype=xyz.dtype))
        self.head = torch.nn.Sequential(torch.nn.Conv2d(16, 128, 1), torch.nn.ReLU(), torch.nn.Conv2d(128, 5, 1)).to(xyz)
        self.metadata = attachment_metadata(renderer, "native_16d_feature_head")

    def parameters(self):
        return [self.features, *self.head.parameters()]

    def render_features(self, renderer, camera):
        validate_attachment(self.metadata, renderer.identity)
        return renderer.render_channels(camera, self.features)

    def render_probabilities(self, renderer, camera):
        import torch
        features = self.render_features(renderer, camera)
        return torch.softmax(self.head(features[None])[0], dim=0)

    def state_dict(self):
        return {"metadata": dict(self.metadata), "features": self.features.detach().cpu(),
                "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()}}

    def load_state_dict(self, payload, renderer):
        import torch
        validate_attachment(payload["metadata"], renderer.identity)
        value, head = payload["features"], payload["head"]
        expected = self.head.state_dict()
        if payload["metadata"].get("field_kind") != self.metadata["field_kind"] or not isinstance(value, torch.Tensor) or value.shape != self.features.shape or not torch.isfinite(value).all():
            raise ValueError("Feature field capacity or values changed")
        if head.keys() != expected.keys() or any(not isinstance(v, torch.Tensor) or v.shape != expected[k].shape or not torch.isfinite(v).all() for k, v in head.items()):
            raise ValueError("Feature classifier differs from registered 16→128→5 head")
        with torch.no_grad():
            self.features.copy_(value.to(self.features))
            self.head.load_state_dict(head)


class SemanticFieldRenderer:
    """Attach native probabilities while leaving the RGB renderer unchanged."""
    def __init__(self, rgb_renderer, field):
        validate_attachment(field.metadata, rgb_renderer.identity)
        self.rgb_renderer, self.field = rgb_renderer, field
        self.identity = rgb_renderer.identity
        self.official_to_model = rgb_renderer.official_to_model

    def render_pinhole(self, camera):
        import torch
        rgb = self.rgb_renderer.render_pinhole(camera)
        with torch.no_grad():
            probabilities = self.field.render_probabilities(self.rgb_renderer, camera).cpu().numpy()
        return replace(rgb, probabilities=probabilities)
