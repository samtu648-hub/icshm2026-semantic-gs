"""Train-side evidence and Dirichlet helpers for SEM-003 / S2.

This module deliberately does *not* implement a contribution gate.  Frozen RGB
contributions are used only to aggregate a training-side diagnostic evidence
table.  Ordinary five-class semantic supervision continues to reach every
semantic anchor; only anchors selected from that table receive the optional
background/stay-cable concentration regularizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.scaffold_semantic_sidecar import _render_color_channels, frozen_generated_geometry


CLASS_COUNT = 5
BACKGROUND = 0
DECK = 1
STAY_CABLE = 2
TOWER = 3
FOUNDATION = 4
PROTECTED_CLASS_IDS = (DECK, TOWER, FOUNDATION)


def stable_softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """Return ``x`` with ``softplus(x) == value`` without overflowing at x>20."""
    if bool((value <= 0).any()):
        raise ValueError("softplus inverse requires strictly positive evidence")
    # Clamp the unused low-value branch so ``torch.where`` never evaluates an
    # overflowing ``expm1`` for a high-concentration SEM-001 conversion.
    return torch.where(value > 20.0, value, torch.log(torch.expm1(value.clamp_max(20.0))))


@dataclass(frozen=True)
class ConflictPolicy:
    """Deterministic, training-side-only eligibility rule for SEM-003."""

    minimum_support_views: int
    minimum_background_fraction: float
    minimum_stay_cable_fraction: float
    maximum_protected_fraction: float
    conflict_quantile: float

    def validate(self) -> None:
        if self.minimum_support_views < 1:
            raise ValueError("minimum_support_views must be positive")
        for name, value in (
            ("minimum_background_fraction", self.minimum_background_fraction),
            ("minimum_stay_cable_fraction", self.minimum_stay_cable_fraction),
            ("maximum_protected_fraction", self.maximum_protected_fraction),
            ("conflict_quantile", self.conflict_quantile),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def conflict_statistics(
    evidence: torch.Tensor,
    support_views: torch.Tensor,
    policy: ConflictPolicy,
    *,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor | float]:
    """Derive an immutable bg/stay conflict mask from train-side evidence.

    ``evidence[g, c]`` is the sum of frozen RGB contribution mass assigned to
    label class ``c`` over the *training* cameras.  The policy first excludes
    low-support and protected-class anchors, then selects the requested upper
    quantile of the symmetric bg/stay conflict score:

    ``C_bc = 2 min(E_bg, E_stay) / (E_bg + E_stay + eps)``.
    """
    policy.validate()
    if evidence.ndim != 2 or evidence.shape[1] != CLASS_COUNT:
        raise ValueError("evidence must have shape [anchors, 5]")
    if support_views.ndim != 1 or support_views.shape[0] != evidence.shape[0]:
        raise ValueError("support_views must have shape [anchors]")
    if bool((evidence < 0).any()):
        raise ValueError("evidence must be non-negative")

    total = evidence.sum(dim=1)
    safe_total = total.clamp_min(float(epsilon))
    posterior = evidence / safe_total[:, None]
    background_fraction = posterior[:, BACKGROUND]
    stay_fraction = posterior[:, STAY_CABLE]
    protected_fraction = posterior[:, list(PROTECTED_CLASS_IDS)].sum(dim=1)
    bg_stay_mass = evidence[:, BACKGROUND] + evidence[:, STAY_CABLE]
    conflict = 2.0 * torch.minimum(evidence[:, BACKGROUND], evidence[:, STAY_CABLE]) / bg_stay_mass.clamp_min(float(epsilon))
    base = (
        (support_views >= int(policy.minimum_support_views))
        & (background_fraction >= float(policy.minimum_background_fraction))
        & (stay_fraction >= float(policy.minimum_stay_cable_fraction))
        & (protected_fraction <= float(policy.maximum_protected_fraction))
        & (bg_stay_mass > float(epsilon))
    )
    if not bool(base.any()):
        threshold = float("inf")
        eligibility = torch.zeros_like(base)
    else:
        threshold = float(torch.quantile(conflict[base], float(policy.conflict_quantile)).item())
        eligibility = base & (conflict >= threshold)
    return {
        "total_evidence": total,
        "posterior": posterior,
        "background_fraction": background_fraction,
        "stay_cable_fraction": stay_fraction,
        "protected_fraction": protected_fraction,
        "background_stay_conflict": conflict,
        "base_eligibility": base,
        "eligibility": eligibility,
        "conflict_threshold": threshold,
    }


class DirichletAnchorSidecar:
    """A five-class static semantic sidecar parameterized by Dirichlet evidence.

    A direct per-class evidence tensor at very high concentration would make a
    SEM-001 conversion numerically exact but nearly impossible to optimize. We
    therefore parameterize class composition with ordinary trainable logits and
    a separately trainable log concentration.  For each anchor:

    ``alpha = 1 + softplus(raw_evidence) = 1 + exp(log_strength) * softmax(logits)``.

    This realizes the requested alpha form, preserves healthy logits gradients,
    and makes uncertainty concentration an explicit, auditable scalar.
    """

    def __init__(
        self,
        sem001_logits: torch.Tensor,
        *,
        initial_evidence_strength: float,
        device: torch.device | str | None = None,
    ) -> None:
        if sem001_logits.ndim != 2 or sem001_logits.shape[1] != CLASS_COUNT:
            raise ValueError("SEM-001 logits must have shape [anchors, 5]")
        if initial_evidence_strength <= 0.0:
            raise ValueError("initial_evidence_strength must be positive")
        resolved = sem001_logits.device if device is None else torch.device(device)
        self.class_logits = torch.nn.Parameter(sem001_logits.detach().to(device=resolved, dtype=torch.float32).clone())
        initial_log_strength = math.log(float(initial_evidence_strength))
        self.log_evidence_strength = torch.nn.Parameter(
            torch.full((sem001_logits.shape[0],), initial_log_strength, dtype=torch.float32, device=resolved)
        )

    @property
    def anchor_count(self) -> int:
        return int(self.class_logits.shape[0])

    def evidence_strength(self) -> torch.Tensor:
        # Exponentiation is bounded only as a numerical safeguard. The range is
        # recorded in the config and is not a data-dependent tuning mechanism.
        return torch.exp(self.log_evidence_strength.clamp(min=-20.0, max=20.0))

    def raw_evidence(self) -> torch.Tensor:
        composition = torch.softmax(self.class_logits, dim=1)
        return stable_softplus_inverse(self.evidence_strength()[:, None] * composition)

    def alpha(self) -> torch.Tensor:
        return 1.0 + torch.nn.functional.softplus(self.raw_evidence())

    def predictive_probabilities(self) -> torch.Tensor:
        alpha = self.alpha()
        return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(torch.finfo(alpha.dtype).tiny)

    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        return {
            "dirichlet_class_logits": self.class_logits.detach().cpu(),
            "dirichlet_log_evidence_strength": self.log_evidence_strength.detach().cpu(),
        }

    def load_checkpoint_state(self, payload: dict[str, Any]) -> None:
        stored_logits = payload.get("dirichlet_class_logits")
        stored_strength = payload.get("dirichlet_log_evidence_strength")
        if not isinstance(stored_logits, torch.Tensor) or tuple(stored_logits.shape) != tuple(self.class_logits.shape):
            raise ValueError("Dirichlet checkpoint class-logit shape does not match frozen RGB anchors")
        if not isinstance(stored_strength, torch.Tensor) or tuple(stored_strength.shape) != tuple(self.log_evidence_strength.shape):
            raise ValueError("Dirichlet checkpoint concentration shape does not match frozen RGB anchors")
        with torch.no_grad():
            self.class_logits.copy_(stored_logits.to(device=self.class_logits.device, dtype=torch.float32))
            self.log_evidence_strength.copy_(stored_strength.to(device=self.log_evidence_strength.device, dtype=torch.float32))


def masked_concentration_regularizer(
    log_evidence_strength: torch.Tensor,
    eligibility: torch.Tensor,
    *,
    target_evidence_strength: float,
) -> torch.Tensor:
    """Encourage uncertainty only on fixed bg/stay-conflict anchors.

    The mask must be a detached, training-side artifact. The regularizer never
    touches protected-class anchor selection and never gates ordinary CE.
    """
    if target_evidence_strength <= 0.0:
        raise ValueError("target_evidence_strength must be positive")
    if eligibility.ndim != 1 or eligibility.shape != log_evidence_strength.shape:
        raise ValueError("eligibility must match log_evidence_strength")
    selected = eligibility.detach().to(dtype=torch.bool, device=log_evidence_strength.device)
    if not bool(selected.any()):
        return log_evidence_strength.sum() * 0.0
    target = math.log(float(target_evidence_strength))
    return (log_evidence_strength[selected] - target).square().mean()


def render_dirichlet_semantic_probabilities(view: Any, model: Any, sidecar: DirichletAnchorSidecar, pipe: Any) -> torch.Tensor:
    """Render predictive Dirichlet probabilities using frozen Scaffold geometry.

    The two rasterizer calls exactly follow SEM-001's five-class packing.  RGB
    geometry, opacity and color are retrieved under ``no_grad`` by
    ``frozen_generated_geometry``; autograd can therefore reach only Dirichlet
    semantic parameters.
    """
    geometry, visible = frozen_generated_geometry(view, model, pipe)
    active_offsets = geometry[4]
    anchor_probabilities = sidecar.predictive_probabilities()[visible]
    offset_probabilities = anchor_probabilities.repeat_interleave(model.n_offsets, dim=0)[active_offsets]
    rgb_like = _render_color_channels(
        geometry,
        offset_probabilities[:, :3],
        view,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=offset_probabilities.device),
    )
    tail_rgb = torch.cat((offset_probabilities[:, 3:5], torch.zeros_like(offset_probabilities[:, :1])), dim=1)
    tail = _render_color_channels(
        geometry,
        tail_rgb,
        view,
        torch.zeros(3, dtype=torch.float32, device=offset_probabilities.device),
    )
    output = torch.cat((rgb_like, tail[:2]), dim=0).clamp_min(0.0)
    return output / output.sum(dim=0, keepdim=True).clamp_min(torch.finfo(output.dtype).tiny)
