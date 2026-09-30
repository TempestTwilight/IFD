"""
IEEE-CIS Fraud Detection dataset loader and synthetic fallback dataset.

The real dataset uses:
- chronological 80/20 train/test split
- train-only median imputation
- train-only StandardScaler fitting
- deterministic preprocessing independent of training seed

The synthetic fallback uses a local RNG and does not modify global RNG state.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
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
    """
    Load IEEE-CIS Fraud Detection data or a deterministic synthetic fallback.

    The real dataset is split chronologically:
        first 80% -> training
        final 20% -> evaluation

    Preprocessing is fitted on the training portion only to avoid
    temporal/data leakage.
    """

    base_path = Path(data_dir)
    raw_path = base_path / "raw" if (base_path / "raw").exists() else base_path

    train_trans = raw_path / "train_transaction.csv"
    train_id = raw_path / "train_identity.csv"

    # ========================================================================
    # Real IEEE-CIS dataset
    # ========================================================================

    if train_trans.exists():
        df_trans = pd.read_csv(
            train_trans,
            nrows=nrows,
        )

        if "TransactionID" not in df_trans.columns:
            raise ValueError("train_transaction.csv is missing 'TransactionID'.")

        if "isFraud" not in df_trans.columns:
            raise ValueError("train_transaction.csv is missing 'isFraud'.")

        # --------------------------------------------------------------------
        # Merge identity data.
        #
        # Do NOT use nrows=nrows for train_identity.csv. The first N identity
        # rows do not necessarily correspond to the first N transactions.
        # --------------------------------------------------------------------

        if train_id.exists():
            df_id = pd.read_csv(train_id)

            if "TransactionID" not in df_id.columns:
                raise ValueError("train_identity.csv is missing 'TransactionID'.")

            if df_id["TransactionID"].duplicated().any():
                raise ValueError("train_identity.csv contains duplicate TransactionID values.")

            df = df_trans.merge(
                df_id,
                on="TransactionID",
                how="left",
                validate="one_to_one",
            )
        else:
            df = df_trans

        # --------------------------------------------------------------------
        # Chronological ordering.
        # --------------------------------------------------------------------

        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT").reset_index(drop=True)

        # --------------------------------------------------------------------
        # Labels and features.
        # --------------------------------------------------------------------

        y_all = df["isFraud"].to_numpy(dtype=np.float32)

        df_features = df.drop(
            columns=[
                "TransactionID",
                "isFraud",
                "TransactionDT",
            ],
            errors="ignore",
        )

        # Keep numeric features for this baseline loader.
        num_cols = df_features.select_dtypes(include=[np.number]).columns

        if len(num_cols) == 0:
            raise ValueError("No numeric features were found in the IEEE-CIS dataset.")

        df_num = df_features[num_cols]

        # --------------------------------------------------------------------
        # Chronological 80/20 split.
        #
        # IMPORTANT:
        # Do this BEFORE fitting imputation/scaling.
        # --------------------------------------------------------------------

        split_idx = int(len(df_num) * 0.8)

        if split_idx <= 0 or split_idx >= len(df_num):
            raise ValueError(f"Dataset is too small for an 80/20 split: {len(df_num)} samples.")

        train_df = df_num.iloc[:split_idx].copy()
        test_df = df_num.iloc[split_idx:].copy()

        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]

        # --------------------------------------------------------------------
        # Train-only median imputation.
        # --------------------------------------------------------------------

        train_medians = train_df.median(numeric_only=True)

        train_df = train_df.fillna(train_medians).fillna(0.0)
        test_df = test_df.fillna(train_medians).fillna(0.0)

        # --------------------------------------------------------------------
        # Train-only standardization.
        # --------------------------------------------------------------------

        scaler = StandardScaler()

        x_train_scaled = scaler.fit_transform(train_df.to_numpy()).astype(np.float32)

        x_test_scaled = scaler.transform(test_df.to_numpy()).astype(np.float32)

        # --------------------------------------------------------------------
        # Final sanity checks.
        # --------------------------------------------------------------------

        if not np.isfinite(x_train_scaled).all():
            raise ValueError(
                "Training features contain NaN or infinite values after preprocessing."
            )

        if not np.isfinite(x_test_scaled).all():
            raise ValueError("Test features contain NaN or infinite values after preprocessing.")

        x_train = torch.from_numpy(x_train_scaled)
        x_test = torch.from_numpy(x_test_scaled)

        return (
            IEEEFraudDataset(
                x_train,
                torch.from_numpy(y_train),
            ),
            IEEEFraudDataset(
                x_test,
                torch.from_numpy(y_test),
            ),
        )

    # ========================================================================
    # Synthetic fallback
    # ========================================================================

    if not synthetic_fallback:
        raise FileNotFoundError(f"IEEE-CIS CSV files not found in '{data_dir}' or '{raw_path}'.")

    if nrows is not None:
        num_samples = nrows

    if num_samples < 2:
        raise ValueError("num_samples must be >= 2.")

    if input_dim < 1:
        raise ValueError("input_dim must be >= 1.")

    # Local RNG: does not modify the global NumPy RNG state.
    rng = np.random.default_rng(42)

    x_train = rng.standard_normal(size=(num_samples, input_dim)).astype(np.float32)

    y_train = (rng.random(num_samples) > 0.8).astype(np.float32)

    test_samples = num_samples // 2

    x_test = rng.standard_normal(size=(test_samples, input_dim)).astype(np.float32)

    y_test = (rng.random(test_samples) > 0.8).astype(np.float32)

    return (
        IEEEFraudDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        IEEEFraudDataset(
            torch.from_numpy(x_test),
            torch.from_numpy(y_test),
        ),
    )
