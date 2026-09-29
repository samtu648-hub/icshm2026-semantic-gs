from __future__ import annotations

import random
import unittest

import numpy as np

from src.quality_mask2former_training import sampled_crop
from src.semantic_context_sampling import (
    CONTEXT_HEIGHT,
    CONTEXT_WIDTH,
    FullFrameContextSampler,
    context_plan,
    resize_full_frame_pair,
)


class Sem385ContextSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.label = np.zeros((989, 1320), dtype=np.uint8)
        self.label[:300, :400] = 1
        self.label[:300, -400:] = 2
        self.label[-300:, :400] = 3
        self.label[-300:, -400:] = 4
        self.image = np.repeat(self.label[:, :, None] * 40, 3, axis=2)

    def test_plan_is_exactly_balanced_and_keeps_parent_trace(self) -> None:
        parent = [
            {
                "step": step,
                "name": f"{step % 7:03d}.png",
                "top": step % 222,
                "left": step % 553,
                "flip": step % 3 == 0,
            }
            for step in range(1, 6001)
        ]
        plan = context_plan(parent)
        self.assertEqual(sum(row["mode"] == "full_frame_context" for row in plan), 3000)
        self.assertEqual(sum(row["mode"] == "native_crop" for row in plan), 3000)
        for source, candidate in zip(parent, plan):
            for key in ("step", "name", "top", "left", "flip"):
                self.assertEqual(candidate[key], source[key])

    def test_full_frame_resize_retains_all_four_corners(self) -> None:
        image, label = resize_full_frame_pair(self.image, self.label)
        self.assertEqual(image.shape, (CONTEXT_HEIGHT, CONTEXT_WIDTH, 3))
        self.assertEqual(label.shape, (CONTEXT_HEIGHT, CONTEXT_WIDTH))
        self.assertEqual(
            [label[0, 0], label[0, -1], label[-1, 0], label[-1, -1]],
            [1, 2, 3, 4],
        )
        self.assertTrue(set(np.unique(label)) <= {0, 1, 2, 3, 4})

    def test_sampler_preserves_parent_rng_and_alternates_shapes(self) -> None:
        names = ["001.png"]
        images = {names[0]: self.image}
        targets = {names[0]: self.label}
        parent_rng = random.Random(20260805)
        candidate_rng = random.Random(20260805)
        sampler = FullFrameContextSampler(sampled_crop)
        parent_traces = []
        candidate_traces = []
        shapes = []
        for _ in range(12):
            _, _, parent_trace = sampled_crop(parent_rng, names, images, targets, 768)
            image, label, candidate_trace = sampler(
                candidate_rng, names, images, targets, 768
            )
            parent_traces.append(parent_trace)
            candidate_traces.append(candidate_trace)
            shapes.append((image.shape[:2], label.shape))
        self.assertEqual(parent_rng.getstate(), candidate_rng.getstate())
        for step, (parent, candidate) in enumerate(
            zip(parent_traces, candidate_traces), start=1
        ):
            self.assertEqual(parent, {key: candidate[key] for key in parent})
            expected = (CONTEXT_HEIGHT, CONTEXT_WIDTH) if step % 2 == 0 else (768, 768)
            self.assertEqual(shapes[step - 1], (expected, expected))


if __name__ == "__main__":
    unittest.main()
