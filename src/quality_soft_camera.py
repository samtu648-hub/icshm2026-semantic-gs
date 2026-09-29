"""SEM348 expert soft diagnostics; does not define fused-policy probabilities."""
import numpy as np
from src.camera_adapter import redistort_semantic_probabilities


def expert_soft(probabilities, camera):
    if (probabilities.dtype != np.float32 or not np.isfinite(probabilities).all()
            or probabilities.min() < 0
            or not np.allclose(probabilities.sum(0), 1, atol=2e-5, rtol=0)):
        raise ValueError('Expected finite normalized float32 expert probabilities')
    mapped = redistort_semantic_probabilities(probabilities, camera)
    if not np.allclose(mapped.sum(0), 1, atol=2e-6, rtol=0):
        raise ValueError('Mapped simplex invalid; do not invent background border scores')
    logs = np.log(np.maximum(mapped, np.finfo(np.float32).tiny))
    return mapped, logs


def independent_linear(probabilities, maps):
    """Four integer-grid gathers with OpenCV's 1/32 linear interpolation grid."""
    _, height, width = probabilities.shape
    ix, iy = [np.rint(m.astype(np.float64) * 32).astype(np.int64) for m in maps]
    x, y = ix // 32, iy // 32
    fx, fy = (ix % 32).astype(np.float32) / 32, (iy % 32).astype(np.float32) / 32
    result = np.zeros_like(probabilities)
    for dx, dy, weight in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
        xx, yy = x + dx, y + dy
        valid = (xx >= 0) & (yy >= 0) & (xx < width) & (yy < height)
        result[:, valid] += probabilities[:, yy[valid], xx[valid]] * weight[valid]
    return result / np.maximum(result.sum(0, keepdims=True), np.finfo(np.float32).tiny)
