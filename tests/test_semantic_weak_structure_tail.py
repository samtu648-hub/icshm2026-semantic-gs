import random

import numpy as np

from src.semantic_weak_structure_tail import (
    TRACE_KEYS,
    WeakStructureTailSampler,
    resize_weak_pair,
    tail_plan,
)


def test_tail_plan_is_alternating_and_deterministic():
    parent = [
        {"name": f"{index:03d}.png", "top": index, "left": index + 1, "flip": bool(index % 2)}
        for index in range(1, 5)
    ]
    hard = ["049.png", "050.png"]
    first = tail_plan(parent, hard, 17)
    assert first == tail_plan(parent, hard, 17)
    assert [row["mode"] for row in first] == [
        "native_crop", "weak_full_frame", "native_crop", "weak_full_frame"
    ]
    assert first[0]["name"] == "001.png"
    assert first[1]["name"] in hard
    assert tuple(key for key in first[0] if key != "step") == TRACE_KEYS


def test_resize_pair_preserves_label_ids_and_shape():
    rgb = np.zeros((989, 1320, 3), dtype=np.uint8)
    label = np.zeros((989, 1320), dtype=np.uint8)
    label[:, 660:] = 2
    image, target = resize_weak_pair(rgb, label)
    assert image.shape == (384, 512, 3)
    assert target.shape == (384, 512)
    assert set(np.unique(target)) == {0, 2}


def test_sampler_advances_parent_and_uses_real_hard_pair():
    calls = []

    def parent(rng, names, images, targets, crop):
        index = len(calls)
        calls.append(index)
        name = names[index]
        return images[name][:768, :768], targets[name][:768, :768], {
            "name": name, "top": index, "left": index + 1, "flip": False
        }

    names = ["001.png", "002.png"]
    hard = ["049.png"]
    images = {name: np.zeros((989, 1320, 3), np.uint8) for name in names + hard}
    targets = {name: np.zeros((989, 1320), np.uint8) for name in names + hard}
    targets["049.png"][:, :] = 1
    plan = tail_plan(
        [
            {"name": "001.png", "top": 0, "left": 1, "flip": False},
            {"name": "002.png", "top": 1, "left": 2, "flip": False},
        ],
        hard,
        3,
    )
    sampler = WeakStructureTailSampler(parent, plan)
    image1, label1, trace1 = sampler(random.Random(1), names, images, targets)
    image2, label2, trace2 = sampler(random.Random(1), names, images, targets)
    assert image1.shape == (768, 768, 3) and not label1.any()
    assert image2.shape == (384, 512, 3) and np.all(label2 == 1)
    assert trace1["mode"] == "native_crop"
    assert trace2["name"] == "049.png"
    assert calls == [0, 1]
