"""
A3: Spectral Matching Attack

Crafts a malicious gradient that lies inside the top-k centered peer subspace
used by Layer 2 (Spectral LOO Anomaly Detector). The deviation from the peer
mean is the projection of the target's deviation onto that subspace, so the
subspace reconstruction error is zero up to numerical precision.
"""

import torch


class SpectralMatching:
    def __init__(self, gamma: float = 0.95):
        """
        Args:
            gamma: Cumulative variance threshold for choosing subspace
                components. Should match the defended Layer 2's gamma.
        """
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

    def generate_gradient(
        self,
        peer_gradients: list[torch.Tensor],
        target_direction: torch.Tensor,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            peer_gradients: Honest client gradients for the current round.
            target_direction: Desired attack direction.
            top_k: Optional manual override for the number of components.

        Returns:
            1D adversarial gradient: peer_mean + a deviation inside span(Vk),
            with deviation norm equal to the mean peer deviation norm.
            If the target has no component in the subspace, the peer mean is
            returned (the safest in-subspace point).
        """
        peers = torch.stack([g.detach().flatten() for g in peer_gradients])
        if not peers.is_floating_point():
            peers = peers.float()
        N, d = peers.shape
        target = target_direction.detach().flatten().to(peers.device, peers.dtype)

        peer_mean = peers.mean(dim=0)

        # Too few peers to define a subspace.
        if N < 2:
            t_norm = target.norm()
            if t_norm < 1e-12:
                return peer_mean.clone()
            return target * (peer_mean.norm() / t_norm)

        centered = peers - peer_mean
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        var = S**2
        total = var.sum()
        if total < 1e-24:
            # Peers are identical: no subspace, no deviation scale.
            return peer_mean.clone()

        max_k = min(N - 1, d, Vh.shape[0])
        if top_k is None:
            cum = torch.cumsum(var / total, dim=0)
            gamma_t = torch.tensor(self.gamma, device=cum.device, dtype=cum.dtype)
            k = int(torch.searchsorted(cum, gamma_t).item()) + 1
        else:
            if top_k < 1:
                raise ValueError("top_k must be >= 1")
            k = top_k
        k = max(1, min(k, max_k))

        Vk = Vh[:k]  # (k, d)

        # Project the target's deviation from the peer mean onto the subspace.
        centered_target = target - peer_mean
        proj = (centered_target @ Vk.T) @ Vk

        proj_norm = proj.norm()
        if proj_norm < 1e-12:
            return peer_mean.clone()

        # Match the typical peer deviation magnitude.
        avg_peer_dev = centered.norm(dim=1).mean()
        return peer_mean + proj * (avg_peer_dev / proj_norm)
