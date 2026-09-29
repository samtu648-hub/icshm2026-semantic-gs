"""SEM379: label-only crop-location mixture on the unchanged native grid."""
import random
import numpy as np


class CableRegionSampler:
    """Keep parent view/flip RNG exactly; use a separate RNG for locations.

    Half the draws retain the parent crop. Guided draws choose an occupied
    64px GT tile uniformly, then a stay_cable pixel uniformly within that tile.
    This avoids weighting the proposal solely by thick cable-mask area.
    A 128px anchor margin is used wherever image borders permit it. No resize,
    label interpolation, image rejection, prediction input, or postprocessing.
    """
    def __init__(self, parent_sampler, seed=20260805, mixture=.5):
        self.parent_sampler = parent_sampler
        self.rng = random.Random(seed)
        self.mixture = mixture
        self.cache = {}

    def index(self, name, target):
        if name not in self.cache:
            yy, xx = np.nonzero(target == 2)
            tile = (yy // 64) * 21 + xx // 64
            order = np.argsort(tile, kind='stable')
            ids, starts = np.unique(tile[order], return_index=True)
            self.cache[name] = (yy[order], xx[order], ids, starts)
        return self.cache[name]

    @staticmethod
    def origin(rng, anchor, size, crop):
        low, high = max(0, anchor-crop+1), min(anchor, size-crop)
        inner_low = max(low, anchor-(crop-1-128))
        inner_high = min(high, anchor-128)
        if inner_low <= inner_high:
            low, high = inner_low, inner_high
        return rng.randint(low, high)

    def __call__(self, rng, names, images, targets, crop=768):
        image, label, trace = self.parent_sampler(rng, names, images, targets, crop)
        trace = dict(trace)
        trace['source'] = 'uniform'
        if self.rng.random() >= self.mixture:
            return image, label, trace
        name = trace['name']
        yy, xx, ids, starts = self.index(name, targets[name])
        if not len(ids):
            trace['source'] = 'uniform_no_cable'
            return image, label, trace
        tile = self.rng.randrange(len(ids))
        end = int(starts[tile+1]) if tile+1 < len(starts) else len(yy)
        pixel = self.rng.randrange(int(starts[tile]), end)
        y, x = int(yy[pixel]), int(xx[pixel])
        top = self.origin(self.rng, y, 989, crop)
        left = self.origin(self.rng, x, 1320, crop)
        image = images[name][top:top+crop, left:left+crop]
        label = targets[name][top:top+crop, left:left+crop]
        if trace['flip']:
            image, label = image[:, ::-1], label[:, ::-1]
        trace.update(source='stay_cable_gt', top=top, left=left,
                     anchor_y=y, anchor_x=x, tile=int(ids[tile]))
        return np.ascontiguousarray(image), np.ascontiguousarray(label), trace
