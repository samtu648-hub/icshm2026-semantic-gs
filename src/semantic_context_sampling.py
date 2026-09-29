"""Deterministic SEM385 native-crop and full-frame context mixture."""

from __future__ import annotations

from collections.abc import Callable
import random

import numpy as np
from PIL import Image


SOURCE_HEIGHT = 989
SOURCE_WIDTH = 1320
CONTEXT_HEIGHT = 768
CONTEXT_WIDTH = 1025
TRACE_KEYS = (
    "name",
    "top",
    "left",
    "flip",
    "mode",
    "input_height",
    "input_width",
)


def trace_for_step(parent_trace: dict, step: int) -> dict:
    """Add the preregistered odd-native/even-context mode to a parent trace."""
    if step <= 0:
        raise ValueError("Context-mix steps are one-based positive integers")
    trace = dict(parent_trace)
    context = step % 2 == 0
    trace.update(
        mode="full_frame_context" if context else "native_crop",
        input_height=CONTEXT_HEIGHT if context else 768,
        input_width=CONTEXT_WIDTH if context else 768,
    )
    return trace


def context_plan(parent_samples: list[dict]) -> list[dict]:
    """Construct the fixed SEM385 plan without changing the parent samples."""
    plan = []
    for expected_step, row in enumerate(parent_samples, start=1):
        if row.get("step") != expected_step:
            raise ValueError("Parent sampling plan has a non-contiguous step sequence")
        plan.append({"step": expected_step, **trace_for_step(row, expected_step)})
    return plan


def resize_full_frame_pair(
    image: np.ndarray, label: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the complete native frame using the frozen paired interpolation."""
    if image.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError(f"Expected native RGB shape, got {image.shape}")
    if label.shape != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError(f"Expected native label shape, got {label.shape}")
    if image.dtype != np.uint8 or label.dtype != np.uint8:
        raise ValueError("Context RGB and labels must be uint8")
    resized_image = np.asarray(
        Image.fromarray(image).resize(
            (CONTEXT_WIDTH, CONTEXT_HEIGHT), resample=Image.Resampling.BILINEAR
        )
    )
    resized_label = np.asarray(
        Image.fromarray(label).resize(
            (CONTEXT_WIDTH, CONTEXT_HEIGHT), resample=Image.Resampling.NEAREST
        )
    )
    return np.ascontiguousarray(resized_image), np.ascontiguousarray(resized_label)


class FullFrameContextSampler:
    """Call the parent sampler every step and replace exactly even-step crops."""

    def __init__(self, parent_sampler: Callable):
        self.parent_sampler = parent_sampler
        self.calls = 0
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(
        self,
        rng: random.Random,
        names: list[str],
        images: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        crop: int = 768,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        image, label, parent_trace = self.parent_sampler(
            rng, names, images, targets, crop
        )
        self.calls += 1
        trace = trace_for_step(parent_trace, self.calls)
        if trace["mode"] == "native_crop":
            return image, label, trace

        name = trace["name"]
        if name not in self.cache:
            self.cache[name] = resize_full_frame_pair(images[name], targets[name])
        image, label = self.cache[name]
        if trace["flip"]:
            image, label = image[:, ::-1], label[:, ::-1]
        return np.ascontiguousarray(image), np.ascontiguousarray(label), trace
