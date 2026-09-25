"""
Metric Tracking & Evaluation Utilities (FPR, TPR, ROC-AUC, PR-AUC)
"""

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_eval_metrics(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """
    Compute comprehensive fraud detection evaluation metrics.

    Args:
        y_true: 1D ground truth binary labels (0 = honest transaction, 1 = fraud)
        y_pred_prob: 1D predicted probability scores in [0, 1]
        threshold: Binary decision threshold

    Returns:
        Dict of computed metrics: roc_auc, pr_auc, precision, recall, f1, fpr, tpr
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)

    # ROC-AUC
    if len(np.unique(y_true)) < 2:
        roc_auc = 0.5
    else:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))

    # PR-AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = float(auc(recall_curve, precision_curve))
    except Exception:
        pr_auc = 0.0

    # Binary metrics at decision threshold
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))

    # Confusion matrix elements for FPR / TPR
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))

    tpr = float(tp / max(1, tp + fn))
    fpr = float(fp / max(1, fp + tn))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
    }


class MetricTracker:
    """Accumulates round-by-round defense metrics."""

    def __init__(self):
        self.history: list[dict[str, float]] = []

    def log_round(self, round_num: int, metrics: dict[str, float]):
        record = {"round": round_num, **metrics}
        self.history.append(record)

    def summary(self) -> dict[str, float]:
        if not self.history:
            return {}
        keys = set().union(*(d.keys() for d in self.history))
        summary_dict = {}
        for k in sorted(keys):
            if k == "round":
                continue
            vals = [h[k] for h in self.history if k in h]
            if vals:
                summary_dict[f"mean_{k}"] = float(np.nanmean(vals))
                summary_dict[f"final_{k}"] = float(vals[-1])
        return summary_dict
