"""Fixed SEM355 candidate-utility regression. No labels in prediction APIs."""
import hashlib
import numpy as np


def neighbourhood(raw):
    h, w = raw.shape
    count = np.zeros((h, w), np.float64)
    valid = np.zeros((h, w), np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y = slice(max(0, -dy), min(h, h-dy))
            x = slice(max(0, -dx), min(w, w-dx))
            count[y, x] += raw[y, x] == raw[slice(max(0, dy), min(h, h+dy)), slice(max(0, dx), min(w, w+dx))]
            valid[y, x] += 1
    return count / valid


def candidates(raw, proposals):
    return np.stack([np.any(proposals == c, axis=0) & (raw != c) for c in range(5)])


def feature_rows(data, pixels, proposed):
    raw, maps, probs, binary, local = data
    r = raw.ravel()[pixels]
    m = maps.reshape(5, -1)[:, pixels]
    p = [v.reshape(5, -1)[:, pixels].T.astype(np.float64) for v in probs]
    b = binary.reshape(2, -1)[:, pixels].T
    eye = np.eye(5)
    support = (m.T == proposed[:, None]).astype(np.float64)
    ids = np.arange(len(pixels))
    delta = np.column_stack([q[ids, proposed]-q[ids, r] for q in (p[0], p[2])])
    entropy = -np.sum(p[0]*np.log(np.maximum(p[0], 1e-30)), axis=1)
    return np.column_stack([*p, b, eye[m[0]], eye[r], eye[proposed], delta,
                            support.sum(1), entropy, support, local.ravel()[pixels], np.ones(len(pixels))])


def sample_rows(data, seed, fold, name, cap):
    eligible = candidates(data[0], data[1]).reshape(5, -1)
    pixels = np.flatnonzero(eligible.any(0))
    token = hashlib.sha256(f'{seed}/{fold}/{name}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], 'little'))
    if len(pixels) > cap:
        pixels = np.sort(rng.choice(pixels, cap, replace=False))
    cls, pos = np.nonzero(eligible[:, pixels])
    return pixels[pos], cls


def ridge_fit(x, y, weights, penalty):
    weights = np.asarray(weights, np.float64)
    weights = weights / weights.sum()
    mean = weights @ x
    mean[-1] = 0
    scale = np.sqrt(weights @ ((x-mean)**2))
    scale[scale < 1e-12] = 1
    scale[-1] = 1
    z = (x-mean)/scale
    gram = z.T @ (weights[:, None]*z)
    rhs = z.T @ (weights*y)
    regularizer = np.eye(x.shape[1])*penalty
    regularizer[-1, -1] = 0
    coef = np.linalg.solve(gram+regularizer, rhs)
    return dict(mean=mean, scale=scale, coef=coef, gram=gram, rhs=rhs)


def choose(raw, eligible, scores):
    scores = np.where(eligible, scores, -np.inf)
    best = scores.max(0)
    unique = (scores == best[None]).sum(0) == 1
    return np.where((best > 0) & unique, scores.argmax(0), raw).astype(np.uint8)


def predict(data, model, chunk=32768):
    raw = data[0]
    if bool(model['fallback']):
        return raw.copy()
    eligible = candidates(raw, data[1]).reshape(5, -1)
    scores = np.full(eligible.shape, -np.inf)
    for c in range(5):
        pixels = np.flatnonzero(eligible[c])
        for start in range(0, len(pixels), chunk):
            ix = pixels[start:start+chunk]
            x = feature_rows(data, ix, np.full(len(ix), c))
            scores[c, ix] = ((x-model['mean'])/model['scale']) @ model['coef']
    return choose(raw.ravel(), eligible, scores).reshape(raw.shape)
