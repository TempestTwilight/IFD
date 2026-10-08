"""
A2: Temporal Grinding Attack

Adversary stays stealthy during warmup rounds, rotating its gradient away from
the honest direction by a bounded angle per round, so the defense's EMA
trajectory is poisoned gradually without a sudden cosine drop in Layer 3.
After warmup, the full target payload is injected at honest-gradient norm.
"""

import math

import torch


class TemporalGrinding:
    def __init__(self, drift_angle_deg: float = 5.0, warmup_rounds: int = 15):
        """
        Args:
            drift_angle_deg: Maximum rotation (degrees) away from the honest
                gradient. The rotation grows by this amount per round.
            warmup_rounds: Rounds (1-indexed, inclusive) over which the
                gradient is ground toward the target before full injection.
        """
        if warmup_rounds < 1:
            raise ValueError("warmup_rounds must be >= 1")
        if drift_angle_deg < 0:
            raise ValueError("drift_angle_deg must be >= 0")
        self.drift_angle_deg = drift_angle_deg
        self.warmup_rounds = warmup_rounds
        self.drift_angle_rad = math.radians(drift_angle_deg)

    def generate_gradient(
        self,
        round_num: int,
        base_gradient: torch.Tensor,
        target_direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            round_num: Current FL round (1-indexed).
            base_gradient: Honest gradient for this round.
            target_direction: Malicious target direction.

        Returns:
            1D adversarial gradient with the same norm as base_gradient.
        """
        g_base = base_gradient.detach().flatten()
        dtype = g_base.dtype if g_base.is_floating_point() else torch.float32
        g_base = g_base.to(dtype)
        v_target = target_direction.detach().flatten().to(g_base.device, dtype)

        norm_base = torch.norm(g_base)
        norm_target = torch.norm(v_target)

        # Degenerate inputs: nothing meaningful to rotate.
        if norm_base < 1e-12 or norm_target < 1e-12:
            return g_base.clone()

        unit_base = g_base / norm_base
        unit_target = v_target / norm_target

        # Post-warmup: full payload at honest norm.
        if round_num >= self.warmup_rounds:
            return unit_target * norm_base

        cos_theta = torch.clamp(torch.dot(unit_base, unit_target), -1.0, 1.0)
        total_angle = math.acos(cos_theta.item())

        # Already aligned: no rotation needed.
        if total_angle < 1e-6:
            return unit_target * norm_base

        # Exactly antipodal: SLERP is undefined, so pick a deterministic
        # orthogonal direction to rotate through.
        if math.pi - total_angle < 1e-6:
            ortho = torch.zeros_like(unit_base)
            ortho[int(torch.argmin(unit_base.abs()))] = 1.0
            ortho = ortho - torch.dot(ortho, unit_base) * unit_base
            ortho = ortho / torch.norm(ortho)
            angle = min(round_num * self.drift_angle_rad, total_angle)
            blended = math.cos(angle) * unit_base + math.sin(angle) * ortho
            return blended / torch.norm(blended) * norm_base

        # SLERP from honest direction toward target by round_num * drift angle.
        angle = min(round_num * self.drift_angle_rad, total_angle)
        t = angle / total_angle
        sin_total = math.sin(total_angle)
        w_base = math.sin((1.0 - t) * total_angle) / sin_total
        w_target = math.sin(t * total_angle) / sin_total

        blended = w_base * unit_base + w_target * unit_target
        blended = blended / (torch.norm(blended) + 1e-12)
        return blended * norm_base
