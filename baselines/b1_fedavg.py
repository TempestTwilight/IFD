"""
B1: FedAvg Baseline
"""

import torch


def fedavg(gradients: list[torch.Tensor]) -> torch.Tensor:
    """Arithmetic mean of client gradients."""
    stacked = torch.stack([g.flatten().float() for g in gradients])
    return stacked.mean(dim=0)
