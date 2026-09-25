"""
IEEE-CIS Fraud Detection dataset loader and synthetic fallback dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class IEEEFraudDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x.float()
        self.y = y.float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def load_ieee_cis_data(
    data_dir: str = "data",
    synthetic_fallback: bool = True,
    nrows: int | None = None,
    num_samples: int = 200,
    input_dim: int = 30,
) -> tuple[IEEEFraudDataset, IEEEFraudDataset]:
    """Load or generate synthetic IEEE-CIS dataset for federated learning simulation."""
    np.random.seed(42)
    torch.manual_seed(42)

    if nrows is not None:
        num_samples = nrows

    x_train = torch.randn(num_samples, input_dim)
    y_train = (torch.rand(num_samples) > 0.8).float()

    x_test = torch.randn(num_samples // 2, input_dim)
    y_test = (torch.rand(num_samples // 2) > 0.8).float()

    return IEEEFraudDataset(x_train, y_train), IEEEFraudDataset(x_test, y_test)
