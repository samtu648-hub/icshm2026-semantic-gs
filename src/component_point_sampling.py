"""Training-only bounded component quotas and read-only point coverage probes."""

from __future__ import annotations

import hashlib
import types
import numpy as np


def full_small_components(label, max_area=1024):
    """Find full-image 8-connected foreground components before any crop."""
    from scipy.ndimage import label as connected, find_objects

    records = []
    for class_id in range(1, 5):
        ids, _ = connected(label == class_id, structure=np.ones((3, 3), bool))
        areas = np.bincount(ids.ravel())
        for cid, slices in enumerate(find_objects(ids), 1):
            if slices is None or areas[cid] > max_area:
                continue
            yy, xx = np.nonzero(ids[slices] == cid)
            records.append(
                {
                    "class_id": class_id,
                    "component_id": cid,
                    "area": int(areas[cid]),
                    "yx": np.stack(
                        (yy + slices[0].start, xx + slices[1].start), 1
                    ).astype(np.int32),
                }
            )
    return records


def crop_regions(records, sample, size=768):
    result = []
    for record in records:
        yx = record["yx"] - np.array([sample["top"], sample["left"]], np.int32)
        yx = yx[((yx >= 0) & (yx < size)).all(1)].copy()
        if not len(yx):
            continue
        if sample["flip"]:
            yx[:, 1] = size - 1 - yx[:, 1]
        result.append({**record, "yx": yx, "visible_area": len(yx)})
    return result


def quotas_for_regions(
    regions, seed, point_count=12544, fraction=0.05, per_component=16, size=768
):
    """Private RNG; round-robin allocation, no replacement within a component."""
    rng = np.random.default_rng(seed)
    result = {}
    for class_id in range(1, 5):
        components = [r for r in regions if r["class_id"] == class_id]
        selected = [
            r["yx"][rng.permutation(len(r["yx"]))[:per_component]] for r in components
        ]
        order = rng.permutation(len(selected))
        yx = []
        cap = int(point_count * fraction)
        for index in range(per_component):
            for comp in order:
                if index < len(selected[comp]) and len(yx) < cap:
                    yx.append(selected[comp][index])
        if yx:
            result[class_id] = (np.asarray(yx, np.float32)[:, ::-1] + 0.5) / size
    return result


def component_hits(coords, region, size=768):
    """Count points with >=0.5 bilinear mass from this specific component."""
    mask = np.zeros((size + 2, size + 2), np.float32)
    mask[region["yx"][:, 0] + 1, region["yx"][:, 1] + 1] = 1
    xy = np.asarray(coords, np.float64) * size - 0.5
    low = np.floor(xy).astype(np.int64)
    frac = xy - low
    low = np.clip(low, -1, size - 1) + 1
    x, y = low[:, 0], low[:, 1]
    fx, fy = frac[:, 0], frac[:, 1]
    weight = (
        (1 - fx) * (1 - fy) * mask[y, x]
        + fx * (1 - fy) * mask[y, x + 1]
        + (1 - fx) * fy * mask[y + 1, x]
        + fx * fy * mask[y + 1, x + 1]
    )
    return int(np.count_nonzero(weight >= 0.5)), float(weight.sum())


class ComponentSampler:
    """Instance-scoped context; upstream model/state/config files are untouched."""

    def __init__(self, enabled=False, seed=418, callback=None):
        self.enabled, self.seed, self.callback = enabled, seed, callback
        self.cache = {}
        self.regions = []
        self.quota = {}
        self.stats = {"loss_calls": 0, "replaced_points": 0, "quota_masks": 0}
        self.sample = None
        self.loss_index = self.match_index = 0

    def set_sample(self, sample, full_label):
        name = sample["name"]
        if name not in self.cache:
            self.cache[name] = full_small_components(full_label)
        self.sample = sample
        self.regions = crop_regions(self.cache[name], sample)
        # Repeating the same crop, including AMP retry, repeats the quota.
        token = repr(
            (self.seed, name, sample["top"], sample["left"], sample["flip"])
        ).encode()
        self.quota = quotas_for_regions(
            self.regions, int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        )
        self.loss_index = self.match_index = 0

    def observe(self, phase, layer, coords, labels, confidence=None):
        if self.callback is None:
            return
        values = coords.detach().cpu().numpy()
        for region in self.regions:
            indices = [i for i, c in enumerate(labels) if c == region["class_id"]]
            if not indices:
                continue
            points = values[0 if len(values) == 1 else indices[0]]
            hits, mass = component_hits(points, region)
            self.callback(
                {
                    "name": self.sample["name"],
                    "phase": phase,
                    "layer": layer,
                    "class_id": region["class_id"],
                    "component_id": region["component_id"],
                    "full_area": region["area"],
                    "visible_area": region["visible_area"],
                    "points": hits,
                    "bilinear_mass": mass,
                    "matched_class_probability": None
                    if confidence is None
                    else confidence[indices[0]],
                }
            )

    def install(self):
        import torch
        import src.quality_mask2former_training as training
        import mask2former.modeling.criterion as criterion
        import mask2former.modeling.matcher as matcher

        old_crop = training.sampled_crop
        old_loss = criterion.SetCriterion.loss_masks
        old_match = matcher.HungarianMatcher.memory_efficient_forward
        owner = self

        def sampled(rng, names, images, targets, crop=768):
            result = old_crop(rng, names, images, targets, crop)
            owner.set_sample(result[2], targets[result[2]["name"]])
            return result

        def loss_masks(instance, outputs, targets, indices, num_masks):
            labels = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(targets, indices)]
            ).tolist()
            sampler = criterion.get_uncertain_point_coords_with_randomness
            layer = owner.loss_index
            owner.loss_index += 1

            def points(*args, **kwargs):
                native = sampler(*args, **kwargs)
                owner.observe("loss", layer, native, labels)
                owner.stats["loss_calls"] += 1
                if not owner.enabled or not owner.quota:
                    return native
                updated = native.clone()
                count = native.shape[1]
                random_count = count - int(count * instance.importance_sample_ratio)
                for row, c in enumerate(labels):
                    if c not in owner.quota:
                        continue
                    q = owner.quota[c]
                    if len(q) > random_count or len(q) > int(0.05 * count):
                        raise ValueError("Quota exceeds registered random point budget")
                    updated[row, -len(q) :] = torch.as_tensor(
                        q, device=native.device, dtype=native.dtype
                    )
                    owner.stats["replaced_points"] += len(q)
                    owner.stats["quota_masks"] += 1
                return updated

            criterion.get_uncertain_point_coords_with_randomness = points
            try:
                return old_loss(instance, outputs, targets, indices, num_masks)
            finally:
                criterion.get_uncertain_point_coords_with_randomness = sampler

        def matching(instance, outputs, targets):
            if owner.callback is None:
                return old_match(instance, outputs, targets)
            sample = matcher.point_sample
            captured = []

            def recording(tensor, coords, *args, **kwargs):
                if not captured:
                    captured.append(coords.detach().clone())
                return sample(tensor, coords, *args, **kwargs)

            matcher.point_sample = recording
            try:
                matched = old_match(instance, outputs, targets)
            finally:
                matcher.point_sample = sample
            if len(targets) != 1:
                raise ValueError("Only registered batch size one is supported")
            source, target = matched[0]
            labels = targets[0]["labels"][target].tolist()
            probability = outputs["pred_logits"][0].float().softmax(-1)
            confidence = [
                float(probability[i, c]) for i, c in zip(source.tolist(), labels)
            ]
            owner.observe("matcher", owner.match_index, captured[0], labels, confidence)
            owner.match_index += 1
            return matched

        training.sampled_crop = sampled
        criterion.SetCriterion.loss_masks = loss_masks
        matcher.HungarianMatcher.memory_efficient_forward = matching
        self.restore = lambda: self._restore(
            training, criterion, matcher, old_crop, old_loss, old_match
        )

    @staticmethod
    def _restore(training, criterion, matcher, crop, loss, match):
        training.sampled_crop = crop
        criterion.SetCriterion.loss_masks = loss
        matcher.HungarianMatcher.memory_efficient_forward = match
