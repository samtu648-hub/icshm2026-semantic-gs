"""Deterministic weak-structure tail sampling for SEM440."""

from __future__ import annotations

from collections.abc import Callable
import random

import numpy as np
from PIL import Image


WEAK_HEIGHT = 384
WEAK_WIDTH = 512
TRACE_KEYS = (
    "base_name",
    "name",
    "top",
    "left",
    "flip",
    "mode",
    "input_height",
    "input_width",
)


def resize_weak_pair(image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a complete 1320x989 RGB/label pair without synthetic labels."""
    if image.shape != (989, 1320, 3) or image.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB input: {image.shape} {image.dtype}")
    if label.shape != (989, 1320) or label.dtype != np.uint8:
        raise ValueError(f"Unexpected label input: {label.shape} {label.dtype}")
    rgb = np.asarray(
        Image.fromarray(image).resize((WEAK_WIDTH, WEAK_HEIGHT), Image.Resampling.BILINEAR)
    )
    target = np.asarray(
        Image.fromarray(label).resize((WEAK_WIDTH, WEAK_HEIGHT), Image.Resampling.NEAREST)
    )
    return np.ascontiguousarray(rgb), np.ascontiguousarray(target)


def tail_plan(
    parent_samples: list[dict], hard_names: list[str], seed: int, start_step: int = 12001
) -> list[dict]:
    """Freeze alternating native and weak full-frame updates."""
    if not hard_names or len(set(hard_names)) != len(hard_names):
        raise ValueError("Hard-view list must be non-empty and unique")
    rng = random.Random(seed)
    rows = []
    for offset, parent in enumerate(parent_samples):
        step = start_step + offset
        weak = step % 2 == 0
        name = rng.choice(hard_names) if weak else parent["name"]
        rows.append(
            {
                "step": step,
                "base_name": parent["name"],
                "name": name,
                "top": parent["top"],
                "left": parent["left"],
                "flip": parent["flip"],
                "mode": "weak_full_frame" if weak else "native_crop",
                "input_height": WEAK_HEIGHT if weak else 768,
                "input_width": WEAK_WIDTH if weak else 768,
            }
        )
    return rows


class WeakStructureTailSampler:
    """Advance the parent RNG every update and replace even updates with hard views."""

    def __init__(self, parent_sampler: Callable, plan: list[dict]):
        self.parent_sampler = parent_sampler
        self.plan = plan
        self.calls = 0
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, rng, names, images, targets, crop=768):
        image, label, parent = self.parent_sampler(rng, names, images, targets, crop)
        if self.calls >= len(self.plan):
            raise IndexError("SEM440 tail sampler exhausted")
        expected = self.plan[self.calls]
        self.calls += 1
        base = {key: parent[key] for key in ("name", "top", "left", "flip")}
        frozen = {
            "name": expected["base_name"],
            "top": expected["top"],
            "left": expected["left"],
            "flip": expected["flip"],
        }
        if base != frozen:
            raise ValueError(f"Parent sampler departed from frozen plan: {base} != {frozen}")
        if expected["mode"] == "native_crop":
            return image, label, {key: expected[key] for key in TRACE_KEYS}
        name = expected["name"]
        if name not in self.cache:
            self.cache[name] = resize_weak_pair(images[name], targets[name])
        image, label = self.cache[name]
        if expected["flip"]:
            image, label = image[:, ::-1], label[:, ::-1]
        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(label),
            {key: expected[key] for key in TRACE_KEYS},
        )
