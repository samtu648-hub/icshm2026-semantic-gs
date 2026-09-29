"""Exact alpha-compositing evidence and dominant-offset attribution."""
import torch

from src.scaffold_semantic_sidecar import _render_color_channels
from src.scaffold_offset_semantic import active_offset_indices


def contributor_evidence(geometry, view, target):
    """Color adjoint equals sum_p T_i alpha_i GT_c; geometry is detached."""
    colors = torch.zeros((len(geometry[0]), 3), device="cuda", requires_grad=True)
    rendered = _render_color_channels(geometry, colors, view, torch.zeros(3, device="cuda"))
    probes = torch.stack((target == 0, target == 1, torch.ones_like(target, dtype=torch.bool))).float()
    evidence = torch.autograd.grad((rendered * probes).sum(), colors)[0].detach()
    if not torch.isfinite(evidence).all() or float(evidence.min()) < 0:
        raise RuntimeError("Invalid pixel contribution evidence")
    return evidence


def dominant_evidence(dominant, maximum, target, count):
    indices = dominant.reshape(-1).long()
    labels = target.reshape(-1)
    weights = maximum.reshape(-1)
    valid = (indices >= 0) & (indices < count)
    output = torch.zeros((count, 3), device=weights.device)
    for c in range(3):
        selected = valid & ((labels == c) if c < 2 else True)
        output[:, c].scatter_add_(0, indices[selected], weights[selected])
    return output


def posterior(evidence):
    counts = evidence[..., :2]
    return (counts + 1) / (counts.sum(-1, keepdim=True) + 2)


def render_table(geometry, visible, table, view, n_offsets=10):
    anchors, offsets = active_offset_indices(visible, geometry[4], n_offsets=n_offsets)
    probabilities = table[anchors, offsets]
    colors = torch.cat((probabilities, torch.zeros_like(probabilities[:, :1])), 1)
    output = _render_color_channels(geometry, colors, view, torch.tensor([1., 0., 0.], device=colors.device))[:2].clamp_min(0)
    return output / output.sum(0, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)
