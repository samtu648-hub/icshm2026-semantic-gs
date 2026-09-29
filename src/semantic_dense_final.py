"""SEM326 official-frame output adapter; SEM325 PINHOLE rule is unchanged."""
import numpy as np
import cv2
from src.semantic_dense_review import native_prediction, native_support, review_foundation
from src.semantic2d import confusion_metrics
from src.semantic_foundation_review_final import validate_final_split, segment_groups, final_gates


def inverse_radial(points, camera, independent=False):
    if camera.model != "SIMPLE_RADIAL": raise ValueError("Official SIMPLE_RADIAL required")
    f, cx, cy, k = camera.params
    if independent:
        distorted = (points - [cx, cy]) / f
        out = distorted.copy()
        for _ in range(15): out = distorted / (1 + k * (out ** 2).sum(-1, keepdims=True))
        return out * f + [cx, cy]
    matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
    return cv2.undistortPoints(points.reshape(-1, 1, 2), matrix, np.array([k, 0., 0., 0., 0.]), P=matrix).reshape(points.shape)


def review_official_output(baseline, source_labels, maps, a, b, ca, cb, filters,
                          independent=False, enabled=True):
    """Official distorted baseline is never resampled; no target GT argument."""
    if baseline.shape != (ca.height, ca.width) or source_labels.shape != (cb.height, cb.width):
        raise ValueError("Official output/source-label shape mismatch")
    if not enabled: return baseline.copy(), {"changed_pixels": 0}
    output = baseline.copy()
    stats = {"reliable_pixels": 0, "outside_source_labels": 0, "changed_pixels": 0}
    for start in range(0, ca.height, 64):
        end = min(start + 64, ca.height)
        y, x = np.mgrid[start:end, :ca.width]
        points = inverse_radial(np.stack((x + .5, y + .5), -1), ca, independent)
        source, support = native_support(points, maps, a, b, ca, cb, filters, independent)
        valid = (np.isfinite(source).all(-1) & (source[..., 0] >= 0) & (source[..., 0] < cb.width)
                 & (source[..., 1] >= 0) & (source[..., 1] < cb.height))
        index = np.floor(np.nan_to_num(source)).astype(np.int64)
        sx, sy = np.clip(index[..., 0], 0, cb.width - 1), np.clip(index[..., 1], 0, cb.height - 1)
        if independent:
            labels = np.take(source_labels, sy * cb.width + sx)
            candidate = np.where((baseline[start:end] == 4) & support & valid & (labels == 0), 0, baseline[start:end]).astype(np.uint8)
        else:
            candidate = review_foundation(baseline[start:end], source_labels[sy, sx], support & valid)
        output[start:end] = candidate
        stats["reliable_pixels"] += int(support.sum())
        stats["outside_source_labels"] += int((support & ~valid).sum())
        stats["changed_pixels"] += int((candidate != baseline[start:end]).sum())
    return output, stats


def quality_summaries(views, groups):
    def merge(rows):
        return {a: confusion_metrics(sum((np.array(r[a]) for r in rows), np.zeros((5, 5), np.int64))) for a in ("baseline", "candidate")}
    metrics = merge(views)
    by_name = {r["target"]: r for r in views}
    grouped = {g: merge([by_name[n] for n in names]) for g, names in groups.items()}
    per_view = {n: {a: confusion_metrics(np.array(row[a]))["mIoU_with_background"] for a in ("baseline", "candidate")} for n, row in by_name.items()}
    worst = sorted(per_view, key=lambda n: (per_view[n]["baseline"], n))[:max(1, int(np.ceil(.1 * len(views))))]
    challenge = [n for n in per_view if 140 <= int(n.split(".")[0]) <= 146]
    if len(challenge) != 7: raise ValueError("Missing fixed challenge")
    subsets = {k: {"names": names, **{a + "_mean_miou": float(np.mean([per_view[n][a] for n in names])) for a in ("baseline", "candidate")}}
               for k, names in (("baseline_worst10", worst), ("challenge_140_146", challenge))}
    return metrics, grouped, subsets, per_view
