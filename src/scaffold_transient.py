"""RGB-092: training-only, structure-protected feature uncertainty weighting.

Independent implementation of WildGaussians equations 8--11, adapted to the
registered soft weight floor. Never imports or mutates a semantic sidecar.
"""
from __future__ import annotations

import json
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@contextmanager
def isolated_rng():
    """Preserve all three RNG streams across third-party feature initialization."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def protection_mask(labels: torch.Tensor, radius: int = 8) -> torch.Tensor:
    if labels.ndim != 2 or labels.numel() == 0 or labels.min() < 0 or labels.max() > 4:
        raise ValueError("Expected a nonempty five-class label map")
    return F.max_pool2d((labels != 0).float()[None, None], 2 * radius + 1,
                        stride=1, padding=radius)[0, 0].bool()


def observation_weights(sigma, protected, iteration, *, warmup=2000,
                        ramp=500, floor=0.2):
    if sigma.shape != protected.shape or not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError("Invalid sigma/protection map")
    strength = min(1.0, max(0.0, (iteration - warmup) / ramp))
    abnormal = (1.0 / (2.0 * sigma.detach().square()) <= 1.0) & ~protected
    return 1.0 - (1.0 - floor) * strength * abnormal.float()


def ssim_map(image, reference):
    """Exact pinned Scaffold 11x11, sigma=1.5, zero-padding SSIM map."""
    from utils.loss_utils import create_window
    window = create_window(11, image.shape[-3]).to(image.device, image.dtype)
    x, y = image[None], reference[None]
    mx = F.conv2d(x, window, padding=5, groups=3)
    my = F.conv2d(y, window, padding=5, groups=3)
    vx = F.conv2d(x * x, window, padding=5, groups=3) - mx.square()
    vy = F.conv2d(y * y, window, padding=5, groups=3) - my.square()
    cov = F.conv2d(x * y, window, padding=5, groups=3) - mx * my
    return (((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)) /
            ((mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)))[0]


def weighted_losses(image, reference, weights):
    if weights.shape != image.shape[-2:] or reference.shape != image.shape:
        raise ValueError("Loss input shape mismatch")
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Weights outside [0,1]")
    weights = weights.detach()
    return ((image - reference).abs() * weights).mean(), ((1 - ssim_map(image, reference)) * weights).mean()


def feature_distance(real, rendered):
    return (2.0 - 2.0 * F.cosine_similarity(real.detach(), rendered.detach(), dim=1)).clamp(0, 1)[:, None]


def uncertainty_loss(sigma, distance, eligible_fraction, prior=0.5):
    return ((distance.detach() / (2 * sigma.square()) + prior * sigma.log()) *
            eligible_fraction).sum() / eligible_fraction.sum().clamp_min(1)


class TransientController:
    def __init__(self, scene, config, labels_root, model_dir):
        from transformers import AutoModel
        self.cfg = config
        self.spec = config["transient"]
        self.output = Path(scene.model_path)
        self.labels_root = Path(labels_root)
        repo = Path(__file__).resolve().parents[1]
        rgb_split = json.loads((repo / config["data"]["split"]).read_text())["primary_fold"]
        sem_split = json.loads((repo / config["data"]["semantic_split"]).read_text())
        self.allowed = set(rgb_split["train_rgb_400"])
        self.labeled = set(sem_split["semantic_train_labeled"])
        actual = {self.name(v.image_name) for v in scene.getTrainCameras()}
        if actual != self.allowed or len(actual) != 303 or len(self.labeled) != 219 or not self.labeled <= actual:
            raise ValueError("Training split does not match 303/219 contract")
        if actual & set(rgb_split["validation_labeled"]):
            raise ValueError("Validation camera reached training controller")
        self.masks = {}
        for name in sorted(self.labeled):
            labels = torch.from_numpy(np.array(Image.open(self.labels_root / name), dtype=np.uint8))
            if tuple(labels.shape) != (989, 1320):
                raise ValueError("Labels are not native PINHOLE 989x1320")
            self.masks[name] = protection_mask(labels, self.spec["protection_radius"])
        # Loading/initializing the feature model and head must not shift Scaffold RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            self.dino = AutoModel.from_pretrained(model_dir, local_files_only=True).cuda().eval()
            self.dino.requires_grad_(False)
            width = self.dino.config.hidden_size
            self.head = torch.nn.Conv2d(width, 1, 1).cuda()
            torch.nn.init.normal_(self.head.weight, std=self.spec["init_std"])
            torch.nn.init.zeros_(self.head.bias)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=self.spec["learning_rate"])
        self.cache = {}
        self.pending = None
        self.records = []
        self.total_masked = 0
        self.total_seen = 0
        self.update_count = 0
        self.log = (self.output / "transient_log.jsonl").open("w")
        self.audit = {"train_names": sorted(actual), "label_names": sorted(self.labeled),
                      "unlabeled_count": len(actual - self.labeled), "validation_masks_read": 0,
                      "label_grid": "PINHOLE", "protected_weight_violations": 0}
        (self.output / "transient_access.json").write_text(json.dumps(self.audit, indent=2))

    @staticmethod
    def name(value):
        return value if value.endswith(".png") else value + ".png"

    @torch.no_grad()
    def features(self, image):
        h, w = image.shape[-2:]
        scale = min(1, self.spec["feature_max_size"] / max(h, w))
        shape = (math.ceil(int(h * scale) / 14) * 14, math.ceil(int(w * scale) / 14) * 14)
        x = F.interpolate(image.detach()[None].clamp(0, 1), size=shape,
                          mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tokens = self.dino(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
        return tokens.transpose(1, 2).reshape(1, -1, shape[0] // 14, shape[1] // 14)

    def losses(self, image, reference, name, iteration):
        from utils.loss_utils import l1_loss, ssim
        name = self.name(name)
        if name not in self.allowed:
            raise ValueError("Non-training image reached transient loss")
        self.pending = None
        if name not in self.labeled or iteration < self.spec["head_start"]:
            return l1_loss(image, reference), 1 - ssim(image, reference)
        if name not in self.cache:
            self.cache[name] = self.features(reference).cpu()
        feat = self.cache[name].cuda()
        protected = self.masks[name].cuda()
        with torch.no_grad():
            small_sigma = F.softplus(self.head(feat)).clamp_min(0.1)
            sigma = F.interpolate(small_sigma, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            weights = observation_weights(sigma, protected, iteration,
                       warmup=self.spec["warmup"], ramp=self.spec["ramp"], floor=self.spec["weight_floor"])
            if not torch.all(weights[protected] == 1):
                raise RuntimeError("Protected supervision changed")
            masked = int((weights < 1).sum())
            self.total_masked += masked
            self.total_seen += weights.numel()
        self.pending = (feat, protected, image.detach(), name, iteration, masked, float(weights.mean()))
        # Preserve the exact original arithmetic throughout warmup.
        if iteration <= self.spec["warmup"]:
            return l1_loss(image, reference), 1 - ssim(image, reference)
        return weighted_losses(image, reference, weights)

    def after_backward(self):
        if self.pending is None:
            return
        feat, protected, image, name, iteration, masked, weight_mean = self.pending
        self.pending = None
        rendered = self.features(image)
        distance = feature_distance(feat, rendered)
        eligible = F.interpolate((~protected).float()[None, None], size=feat.shape[-2:], mode="area")
        # Use a private per-step dropout stream; do not affect growth sampling.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(100000 + iteration)
            dropped = F.dropout(feat, p=self.spec["dropout"], training=True)
            sigma = F.softplus(self.head(dropped)).clamp_min(0.1)
            loss = uncertainty_loss(sigma, distance, eligible, self.spec["log_prior"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite uncertainty loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self.update_count += 1
        if iteration % 100 == 0:
            record = {"iteration": iteration, "name": name, "uncertainty_loss": float(loss.detach()),
                      "masked_pixels": masked, "mean_weight": weight_mean,
                      "masked_fraction_cumulative": self.total_masked / max(1, self.total_seen),
                      "feature_distance_mean": float(distance.mean()), "protected_violations": 0}
            self.log.write(json.dumps(record) + "\n")
            self.log.flush()
        if iteration in (2500, 10000, 20000, 30000):
            self.save(iteration)

    def save(self, iteration):
        state = {"iteration": iteration, "head": self.head.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "config": self.spec, "update_count": self.update_count,
                 "masked_pixels": self.total_masked, "seen_pixels": self.total_seen, "access": self.audit}
        torch.save(state, self.output / f"transient_{iteration}.pth")


def make_controller(scene):
    path = os.environ.get("ICSHM_TRANSIENT_CONFIG")
    if not path:
        return None
    config = json.loads(Path(path).read_text())
    if not config["transient"]["enabled"]:
        return None
    with isolated_rng():
        return TransientController(scene, config, os.environ["ICSHM_TRANSIENT_LABELS"], os.environ["ICSHM_DINO_MODEL"])
