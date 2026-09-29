"""Frozen-native static fitting contracts; no upstream/CUDA import at module load."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.quality_protocol import CLASSES, sha256, unique_names


def semantic_access(split, oof, parent_access, partition):
    """Screening folds share the 303-view RGB parent, NOT independent RGB folds."""
    universe = unique_names(split["semantic_train_labeled"], "semantic train")
    development = unique_names(split["semantic_validation_labeled"], "development")
    rgb_train = unique_names(parent_access["train"], "parent RGB train")
    rgb_eval = unique_names(parent_access["evaluation"], "parent RGB evaluation")
    if universe & development or rgb_train & rgb_eval or not universe <= rgb_train or not development <= rgb_eval:
        raise ValueError("Semantic split does not match RGB parent roles")
    if partition == "development":
        train, target, guard = universe, development, set()
    else:
        choices = {f"oof-{f['fold']}": f for f in oof["folds"]}
        if partition not in choices:
            raise ValueError("Unregistered semantic screening partition")
        fold = choices[partition]
        train, target, guard = (unique_names(fold[k], k) for k in ("train", "oof", "guard"))
        if train | target | guard != universe or train & target or train & guard or target & guard:
            raise ValueError("OOF training, target and guard do not form the parent universe")
    if not train or not target:
        raise ValueError("Empty semantic training or prediction role")
    return {"partition": partition, "train": sorted(train), "prediction": sorted(target), "guard": sorted(guard),
            "class_weight_views": sorted(train), "labels_allowed_during_fit": sorted(train),
            "rgb_training_overlap_with_prediction": sorted(rgb_train & target),
            "whole_chain_spatial_validation": False, "historically_reused_data": True}


class TrainingLabels:
    """Hash-checked role-limited PNG reader. Never uses global manifest counts."""
    def __init__(self, root, names, manifest, resolution):
        self.root = root.resolve()
        self.allowed = unique_names(names, "allowed training labels")
        self.shape = (resolution[1], resolution[0])
        if manifest["class_mapping"] != {n: i for i, n in enumerate(CLASSES)}:
            raise ValueError("Label class mapping differs from official classes")
        entries = manifest["records"]
        if len({r["image"] for r in entries}) != len(entries):
            raise ValueError("Duplicate label manifest entries")
        self.records = {r["image"]: r for r in entries if r["image"] in self.allowed}
        if self.records.keys() != self.allowed:
            raise ValueError("Missing training label provenance")
        self.accessed = set()

    def read(self, name):
        if name not in self.allowed:
            raise ValueError("Forbidden evaluation/guard label read during fitting")
        path = self.root / name
        if not path.resolve().is_relative_to(self.root) or sha256(path) != self.records[name]["label_png_sha256"]:
            raise ValueError("Training label escapes root or differs from its manifest")
        with Image.open(path) as image:
            label = np.array(image)
        if label.shape != self.shape or label.dtype != np.uint8 or int(label.max()) >= len(CLASSES):
            raise ValueError("Expected native single-channel class IDs 0..4, without resize")
        self.accessed.add(name)
        return label

    def audit_counts(self):
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        for name in sorted(self.allowed):
            actual = np.bincount(self.read(name).ravel(), minlength=len(CLASSES))
            expected = [self.records[name]["pixel_counts"].get(str(i), 0) for i in range(len(CLASSES))]
            if not np.array_equal(actual, expected):
                raise ValueError("Training label counts differ from per-view manifest")
            counts += actual
        return counts


def median_frequency_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (5,) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("All five classes require finite positive training-only counts")
    return np.clip(np.median(counts) / counts, .05, 20.).astype(np.float32)


def assert_frozen_renderer(renderer):
    """Gaussian models aren't nn.Modules; inspect direct tensors AND nested MLPs."""
    import torch
    for name, value in vars(renderer.model).items():
        tensors = [(name, value)] if isinstance(value, torch.Tensor) else []
        if isinstance(value, torch.nn.Module):
            tensors += [(f"{name}.{k}", p) for k, p in value.named_parameters()]
        for key, parameter in tensors:
            if parameter.requires_grad or parameter.grad is not None:
                raise ValueError(f"RGB/geometry parameter is not frozen: {key}")


def fit_static(renderer, field, cameras, labels, weights, *, steps, seed, learning_rate, weight_decay, on_step=None):
    """AdamW on semantic logits only; full-image native weighted NLL, no RGB loss."""
    import torch
    if steps <= 0 or set(cameras) != labels.allowed:
        raise ValueError("Training camera/label roles differ or invalid budget")
    assert_frozen_renderer(renderer)
    parameters = list(field.parameters())
    if len(parameters) != 1 or parameters[0] is not field.logits:
        raise ValueError("Only registered primitive/anchor semantic logits may train")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    weights = torch.as_tensor(weights, device=field.logits.device, dtype=field.logits.dtype)
    if weights.shape != (5,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Invalid loss weights")
    sampler = np.random.default_rng(seed)
    names, losses = sorted(labels.allowed), []
    for step in range(1, steps + 1):
        name = names[int(sampler.integers(len(names)))]
        target = torch.from_numpy(labels.read(name).astype(np.int64)).to(field.logits.device)
        probabilities = field.render_probabilities(renderer, cameras[name])
        if probabilities.shape != (5, *target.shape) or not torch.isfinite(probabilities).all():
            raise ValueError(f"Invalid semantic probabilities at step {step}")
        # NLL backward in the selected Torch runtime requires contiguous NCHW
        # input, including when a renderer returns a transposed channel view.
        loss = torch.nn.functional.nll_loss(probabilities.clamp_min(1e-8).log().contiguous()[None], target[None], weight=weights)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if field.logits.grad is None or not torch.isfinite(field.logits.grad).all():
            raise ValueError(f"Missing/nonfinite semantic gradients at step {step}; update refused")
        assert_frozen_renderer(renderer)
        optimizer.step()
        if not torch.isfinite(field.logits).all():
            raise ValueError(f"Nonfinite semantic parameters after step {step}")
        losses.append(float(loss.detach()))
        if on_step is not None:
            on_step(step, name, losses[-1])
    return {"steps": steps, "first_loss": losses[0], "last_loss": losses[-1], "mean_loss": float(np.mean(losses))}
