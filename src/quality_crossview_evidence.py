"""Descriptive source-only evidence. These helpers never accept target truth."""
import numpy as np
from pathlib import Path
import shutil
from src.quality_protocol import sha256
from src.semantic_dense_review import native_support
from src.semantic_dense_geometry import bilinear_sample, fundamental_from_w2c, symmetric_epipolar_pixels

STATES = ('no_first_source', 'no_second_source', 'outside_source_grid',
          'unreliable_correspondence', 'both_deck', 'both_background',
          'both_foundation', 'both_same_other', 'reliable_disagreement')


def snapshot_real_sources(root, output, required, module_files):
    """Ignore virtual module filenames, but never skip an explicitly required file."""
    root = Path(root).resolve()
    paths = {name: sha256(root/name) for name in required}
    for filename in module_files:
        if not isinstance(filename, str):
            continue
        path = Path(filename)
        # torch.ops may advertise a virtual, relative '_ops.py'.
        if not path.is_absolute():
            continue
        path = path.resolve()
        if path.suffix == '.py' and path.is_relative_to(root) and not path.is_relative_to(root/'outputs'):
            paths[str(path.relative_to(root))] = sha256(path)
    for name in paths:
        dest = Path(output)/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, dest)
    return paths


def source_sample(points, source, pose, camera, filters, independent=False, details=False):
    shape = points.shape[:-1]
    if source is None or not points.size:
        result = {'xy': np.full((*shape, 2), np.nan), 'valid': np.zeros(shape, bool),
                  'reliable': np.zeros(shape, bool), 'label': np.full(shape, 255, np.uint8)}
        if details:
            result.update({k: np.full(shape, np.nan) for k in ('forward_confidence', 'reverse_confidence', 'cycle', 'epipolar')})
        return result
    sc = source['camera']
    xy, reliable = native_support(points, source['maps'], pose, source['pose'], camera, sc, filters, independent)
    q = points/[camera.width/2, camera.height/2]-1
    h, w = source['maps'][0][1].shape
    domain = np.isfinite(q).all(-1) & (np.abs(q[..., 0]) <= 1-1/w) & (np.abs(q[..., 1]) <= 1-1/h)
    # The independent sampler pads borders internally; invalid padding is not evidence.
    # Canonicalize its undefined coordinates to the primary sampler's NaN contract.
    xy = np.where(domain[..., None], xy, np.nan)
    valid = np.isfinite(xy).all(-1) & (xy[..., 0] >= 0) & (xy[..., 0] < sc.width) & (xy[..., 1] >= 0) & (xy[..., 1] < sc.height)
    index = np.floor(np.nan_to_num(xy, nan=0, posinf=0, neginf=0)).astype(np.int64)
    x, y = np.clip(index[..., 0], 0, sc.width-1), np.clip(index[..., 1], 0, sc.height-1)
    labels = np.take(source['labels'], y*sc.width+x) if independent else source['labels'][y, x]
    result = {'xy': xy, 'valid': valid, 'reliable': reliable & valid,
              'label': np.where(valid, labels, 255).astype(np.uint8)}
    if not details:
        return result
    def sample(values, grid):
        if not independent:
            return bilinear_sample(values, grid)
        import torch
        import torch.nn.functional as functional
        scalar = values.ndim == 2
        arr = values[..., None] if scalar else values
        h, w = values.shape[:2]
        okay = np.isfinite(grid).all(-1) & (np.abs(grid[..., 0]) <= 1-1/w) & (np.abs(grid[..., 1]) <= 1-1/h)
        tensor = functional.grid_sample(torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None],
            torch.from_numpy(np.ascontiguousarray(np.nan_to_num(grid)))[None], align_corners=False,
            padding_mode='border', mode='bilinear')[0].permute(1, 2, 0).numpy()
        tensor = np.where(okay[..., None], tensor, np.nan)
        return (tensor[..., 0] if scalar else tensor), okay
    (ab, oa), (ba, ob) = source['maps']
    f, vf = sample(ab, q); oc, vo = sample(oa, q)
    back, vb = sample(ba, f); rc, vr = sample(ob, f)
    cycle = np.linalg.norm((back-q)*[camera.width/2, camera.height/2], axis=-1)
    if independent:
        a, b = np.eye(4), np.eye(4)
        for mat, obj in ((a, pose), (b, source['pose'])):
            mat[:3, :3], mat[:3, 3] = obj.rotation_world_to_camera, obj.tvec
        rel = b @ np.linalg.inv(a); tx, ty, tz = rel[:3, 3]
        e = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]]) @ rel[:3, :3]
        def normalized(p, c):
            focal, cx, cy, _ = c.params
            return np.concatenate(((p-[cx, cy])/focal, np.ones((*p.shape[:-1], 1))), -1)
        p, s = normalized(points, camera), normalized(xy, sc)
        lb, la = p @ e.T, s @ e
        residual = np.abs((s*lb).sum(-1))
        epi = np.maximum(camera.params[0]*residual/np.maximum(np.linalg.norm(la[..., :2], axis=-1), 1e-15),
                         sc.params[0]*residual/np.maximum(np.linalg.norm(lb[..., :2], axis=-1), 1e-15))
    else:
        def intrinsics(c):
            f, cx, cy, _ = c.params
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
        matrix = fundamental_from_w2c(pose.rotation_world_to_camera, pose.tvec,
            source['pose'].rotation_world_to_camera, source['pose'].tvec, intrinsics(camera), intrinsics(sc))
        epi = symmetric_epipolar_pixels(points, xy, matrix)
    rebuilt = (valid & vf & vo & vb & vr & np.isfinite(cycle)
               & (oc >= filters['overlap_each_direction_minimum']) & (rc >= filters['overlap_each_direction_minimum'])
               & (cycle <= filters['cycle_maximum_native_pixels']) & (epi <= filters['epipolar_maximum_native_pixels']))
    if not np.array_equal(rebuilt, result['reliable']):
        raise ValueError('Detailed evidence does not reproduce immutable native support')
    result.update(forward_confidence=oc, reverse_confidence=rc, cycle=cycle, epipolar=epi)
    return result


def evidence_states(first, second, first_present, second_present):
    shape = first['label'].shape
    if any(v['label'].shape != shape for v in (first, second)) or (second_present and not first_present):
        raise ValueError('Inconsistent source shapes or order')
    if not first_present:
        return np.zeros(shape, np.uint8)
    if not second_present:
        return np.ones(shape, np.uint8)
    state = np.full(shape, 2, np.uint8)
    valid = first['valid'] & second['valid']
    reliable = valid & first['reliable'] & second['reliable']
    state[valid] = 3
    state[reliable] = 8
    same = reliable & (first['label'] == second['label'])
    state[same] = 7
    for label, code in ((1, 4), (0, 5), (4, 6)):
        state[same & (first['label'] == label)] = code
    return state


def source_status(sample, present):
    """Absent, outside, unreliable, reliable class0..4."""
    out = np.zeros(sample['label'].shape, np.uint8)
    if present:
        out[:] = 1
        out[sample['valid']] = 2
        use = sample['valid'] & sample['reliable']
        out[use] = sample['label'][use]+3
    return out


def counted(states, truth, count):
    states, truth = np.asarray(states).ravel(), np.asarray(truth).ravel()
    if states.shape != truth.shape or np.any(states < 0) or np.any(states >= count) or np.any(truth < 0) or np.any(truth > 4):
        raise ValueError('Invalid descriptive categories')
    primary = np.bincount(states.astype(np.int64)*5+truth, minlength=count*5).reshape(count, 5)
    independent = np.array([[np.count_nonzero((states == s) & (truth == c)) for c in range(5)] for s in range(count)])
    if not np.array_equal(primary, independent):
        raise ValueError('Independent category counts differ')
    return primary


def compare_evidence(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Evidence fields differ')
    for key in expected:
        a, b = expected[key], actual[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Evidence layout differs: '+key)
        okay = np.allclose(a, b, atol=2e-5, rtol=0, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)
        if not okay:
            raise ValueError('Independent evidence differs: '+key)
