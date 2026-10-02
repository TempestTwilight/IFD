"""
A1: Oracle White-Box PGD Attack

Adversary knows the full defense state (layer objects, their real thresholds,
peer gradients, Layer 3 per-client history) and uses normalized-gradient PGD to
find a gradient that maximizes alignment with a target direction while passing
the real Layer 1 / Layer 2 / Layer 3 cascade.

eps is RELATIVE: the L2 ball radius is eps * ||mean_honest||.
"""

import torch
import torch.nn.functional as torch_nn_functional


class OracleWhiteBoxPGD:
    def __init__(
        self,
        eps: float = 0.5,
        step_size: float = 0.05,
        num_steps: int = 30,
        penalty_weight: float = 2.0,
        accept_thresh: float = 0.5,
        accept_margin: float = 0.6,
    ):
        """
        Args:
            eps: Relative L2 radius around the mean honest gradient
                (radius = eps * ||mean_honest||).
            step_size: Step length as a fraction of the ball radius.
            num_steps: Number of PGD iterations.
            penalty_weight: Weight of the defense-rejection penalty.
            accept_thresh: A layer "passes" if its acceptance score >= this.
            accept_margin: Penalty activates below this score (> accept_thresh
                gives a safety margin against surrogate/real mismatch).
        """
        self.eps = eps
        self.step_size = step_size
        self.num_steps = num_steps
        self.penalty_weight = penalty_weight
        self.accept_thresh = accept_thresh
        self.accept_margin = accept_margin

    # ------------------------------------------------------------------
    # Layer 3 prediction (differentiable, does NOT mutate layer3 state)
    # ------------------------------------------------------------------
    @staticmethod
    def _layer3_score(g, honest, layer3, client_id):
        zero_one = torch.ones((), device=g.device, dtype=g.dtype)
        state = layer3.get_client_state(client_id)
        if state is None:
            return zero_one  # first round: layer 3 returns a3 = 1

        rounds = state["rounds_seen"] + 1
        if layer3.warmup_rounds > 0 and rounds <= layer3.warmup_rounds:
            return zero_one  # warmup: no anomaly scoring

        traj = state["trajectory"].to(g.device, g.dtype)
        d_local = 1.0 - torch_nn_functional.cosine_similarity(g, traj, dim=0, eps=1e-8)
        s_local = torch.relu(state["cusum_local"] + d_local - (state["mu0_local"] + layer3.k_local))
        a_local = 1.0 - torch.sigmoid(layer3.slope * (s_local - layer3.h_local))

        ref = torch.median(torch.cat([honest, g.unsqueeze(0)], dim=0), dim=0).values
        d_global = 1.0 - torch_nn_functional.cosine_similarity(g, ref, dim=0, eps=1e-8)
        s_global = torch.relu(
            state["cusum_global"] + d_global - (state["mu0_global"] + layer3.k_global)
        )
        a_global = 1.0 - torch.sigmoid(layer3.slope * (s_global - layer3.h_global))

        return torch.minimum(a_local, a_global)

    def _real_scores(self, g, honest, layer1, layer2, layer3, client_id):
        """Run the actual layers on honest + g (no grad). Returns dict of scalars."""
        out = {}
        with torch.no_grad():
            G = torch.cat([honest, g.unsqueeze(0)], dim=0)
            if layer1 is not None:
                out["l1"] = layer1.score(G)[0][-1].item()
            if layer2 is not None and G.shape[0] >= 3:
                out["l2"] = layer2.score(G)[0][-1].item()
            if layer3 is not None:
                out["l3"] = self._layer3_score(g, honest, layer3, client_id).item()
        return out

    def attack(
        self,
        honest_gradients: list[torch.Tensor],
        target_direction: torch.Tensor,
        layer1=None,
        layer2=None,
        layer3=None,
        client_id: str = "adv_0",
        return_info: bool = False,
    ):
        """
        Craft adversarial gradient g_adv.

        Returns g_adv, or (g_adv, info) if return_info=True. info contains
        the real per-layer acceptance scores, whether all layers passed,
        and the cosine alignment with the target.
        """
        honest = torch.stack([g.detach().flatten() for g in honest_gradients])
        device, dtype = honest.device, honest.dtype
        target = target_direction.detach().flatten().to(device, dtype)
        target = target / (target.norm() + 1e-8)

        mean_honest = honest.mean(dim=0)
        mean_norm = mean_honest.norm()
        radius = self.eps * mean_norm
        lr = self.step_size * radius

        def project(g):
            d = g - mean_honest
            n = d.norm()
            return mean_honest + d * (radius / n) if n > radius else g

        # Layer 2 surrogate subspace: computed once, using the layer's own gamma.
        Vk = None
        if layer2 is not None and honest.shape[0] >= 3:
            peer_mat = honest - mean_honest
            _, S, Vh = torch.linalg.svd(peer_mat, full_matrices=False)
            var = S**2
            if var.sum() > 0:
                cum = torch.cumsum(var / var.sum(), dim=0)
                gamma = torch.tensor(layer2.gamma, device=cum.device, dtype=cum.dtype)
                k = min(int(torch.searchsorted(cum, gamma).item()) + 1, Vh.shape[0])
                Vk = Vh[:k]

        # Feasible start point.
        g_adv = project(target * mean_norm).clone().detach().requires_grad_(True)

        zero = torch.zeros((), device=device, dtype=dtype)
        best_ok, best_ok_cos = None, -2.0
        best_fb, best_fb_viol = None, float("inf")

        for _ in range(self.num_steps + 1):
            # ---- evaluate current iterate against the REAL layers ----
            gc = g_adv.detach()
            real = self._real_scores(gc, honest, layer1, layer2, layer3, client_id)
            cos_t = torch_nn_functional.cosine_similarity(gc, target, dim=0).item()
            viol = sum(max(0.0, self.accept_thresh - v) for v in real.values())
            if viol == 0.0 and cos_t > best_ok_cos:
                best_ok, best_ok_cos = gc.clone(), cos_t
            if viol < best_fb_viol:
                best_fb, best_fb_viol = gc.clone(), viol

            if _ == self.num_steps:
                break

            # ---- differentiable surrogate loss ----
            if g_adv.grad is not None:
                g_adv.grad.zero_()

            loss_target = -torch_nn_functional.cosine_similarity(g_adv, target, dim=0, eps=1e-8)
            penalty = zero

            if layer1 is not None:
                a1 = layer1.score(torch.cat([honest, g_adv.unsqueeze(0)], dim=0))[0][-1]
                penalty = penalty + torch.relu(self.accept_margin - a1)

            if Vk is not None:
                d = g_adv - mean_honest
                recon_err = (d - (d @ Vk.T) @ Vk).norm() / (mean_norm + 1e-8)
                penalty = penalty + recon_err

            if layer3 is not None:
                a3 = self._layer3_score(g_adv, honest, layer3, client_id)
                penalty = penalty + torch.relu(self.accept_margin - a3)

            (loss_target + self.penalty_weight * penalty).backward()

            # ---- normalized-gradient step + projection ----
            with torch.no_grad():
                grad = g_adv.grad
                if grad is not None:
                    g_adv -= lr * grad / (grad.norm() + 1e-12)
                    g_adv.copy_(project(g_adv))

        g_final = best_ok if best_ok is not None else best_fb
        assert g_final is not None  # at least one of best_ok/best_fb is set
        if not return_info:
            return g_final

        real = self._real_scores(g_final, honest, layer1, layer2, layer3, client_id)
        info = {
            "scores": real,
            "passed_all": all(v >= self.accept_thresh for v in real.values()),
            "cos_target": torch_nn_functional.cosine_similarity(g_final, target, dim=0).item(),
            "l2_dist_ratio": ((g_final - mean_honest).norm() / (mean_norm + 1e-8)).item(),
        }
        return g_final, info
