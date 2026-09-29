import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.improvedgs_bridge_repair import solve_tangent_constraints, tangent_constraint


def camera(rotation, translation=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        pose=SimpleNamespace(rotation=np.asarray(rotation), translation=np.asarray(translation)),
        matrix=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
    )


def test_multiview_constraints_recover_world_x_axis():
    point = np.array([0.0, 0.0, 10.0])
    cameras = [
        camera(np.eye(3)),
        camera(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]), (10.0, 0.0, 10.0)),
        camera(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]), (0.0, -10.0, 10.0)),
    ]
    world_direction = np.array([1.0, 0.0, 0.0])
    rows = []
    for item in cameras:
        rotation = item.pose.rotation
        camera_point = rotation @ point + item.pose.translation
        camera_direction = rotation @ world_direction
        matrix = item.matrix
        x, y, z = camera_point
        projected = np.array([
            matrix[0, 0] * (camera_direction[0] * z - x * camera_direction[2]) / z**2,
            matrix[1, 1] * (camera_direction[1] * z - y * camera_direction[2]) / z**2,
        ])
        rows.append(tangent_constraint(point, item, math.atan2(projected[1], projected[0])))
    direction, _, residual, eigengap = solve_tangent_constraints(np.stack(rows))
    assert abs(float(direction @ world_direction)) == pytest.approx(1.0, abs=1e-6)
    assert residual < 1e-10
    assert eigengap > 1e6


def test_directional_children_align_and_narrow_cuda_optional():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA optional")
    from src.improvedgs_bridge_repair import directional_children

    xyz = torch.zeros((1, 3), device="cuda")
    scales = torch.tensor([[4.0, 2.0, 1.0]], device="cuda")
    rotation = torch.eye(3, device="cuda").unsqueeze(0)
    tangent = torch.tensor([[0.0, 1.0, 0.0]], device="cuda")
    children, child_scales, quaternions = directional_children(
        xyz, scales, rotation, tangent, split_distance=0.45, transverse_multiplier=0.5
    )
    assert torch.allclose(children[0], torch.tensor([0.0, 5.4, 0.0], device="cuda"))
    assert torch.allclose(children[1], -children[0])
    assert child_scales[0, 0] == pytest.approx(2.2)
    assert child_scales[0, 1] < 1.0 and child_scales[0, 2] < 0.5
    assert torch.isfinite(quaternions).all()

