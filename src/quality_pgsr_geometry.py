"""PGSR cycle loss: select native valid correspondences before differentiating.

The original visibility/occlusion decision is discrete. Recompute only accepted
cycles in autograd, so rejected invalid divisions/norms cannot poison gradients.
No geometry values, thresholds, model parameters or gradients are sanitized.
"""
from __future__ import annotations


def accepted_cycle_noise(points, pixels, cycle, pixel_noise_threshold, occlusion_aware=True):
    import torch
    with torch.no_grad():
        raw_noise, native_mask = cycle(points.detach(), pixels)
        accepted = native_mask & torch.isfinite(raw_noise)
        if occlusion_aware:
            accepted = accepted & (raw_noise < pixel_noise_threshold)
            weights = 1.0 / torch.exp(raw_noise)
        else:
            weights = torch.ones_like(raw_noise)
        weights[~accepted] = 0
    indices = accepted.nonzero(as_tuple=True)[0]
    noise = torch.zeros_like(raw_noise)
    if indices.numel():
        accepted_noise, _ = cycle(points[indices], pixels[indices])
        if not torch.isfinite(accepted_noise).all():
            raise FloatingPointError("Accepted PGSR cycle unexpectedly nonfinite")
        noise = noise.index_copy(0, indices, accepted_noise)
    return noise, accepted, weights


def native_cycle_evidence(model, target_camera, source_camera, target_depth, source_depth,
                          pixels, pixel_noise_threshold, occlusion_aware=True):
    import torch
    def cycle(points, original_pixels):
        in_source = points @ source_camera.world_view_transform[:3, :3] + source_camera.world_view_transform[3, :3]
        map_z, mask = model.get_points_depth_in_depth_map(source_camera, source_depth, in_source)
        in_source = in_source / in_source[:, 2:3]
        in_source = in_source * map_z.reshape(-1, 1)
        rotation = torch.as_tensor(source_camera.R, dtype=torch.float32, device=points.device)
        translation = torch.as_tensor(source_camera.T, dtype=torch.float32, device=points.device)
        world = (in_source - translation) @ rotation.transpose(-1, -2)
        in_target = world @ target_camera.world_view_transform[:3, :3] + target_camera.world_view_transform[3, :3]
        projected = torch.stack([in_target[:, 0] * target_camera.Fx / in_target[:, 2] + target_camera.Cx,
                                 in_target[:, 1] * target_camera.Fy / in_target[:, 2] + target_camera.Cy], -1).float()
        noise = torch.norm(projected - original_pixels, dim=-1)
        return noise, mask
    points = model.get_points_from_depth(target_camera, target_depth)
    return accepted_cycle_noise(points, pixels.reshape(-1, 2), cycle, pixel_noise_threshold, occlusion_aware)
