"""
metrics.py — Authentication metrics: FAR, FRR, EER, and ROC curve utilities.
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc

log = logging.getLogger(__name__)


def compute_far_frr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """Compute False Accept Rate and False Reject Rate.

    Convention:
        genuine = 0  (positive class = legitimate user)
        imposter = 1

    FAR = imposters accepted / total imposters
    FRR = genuines rejected / total genuines
    """
    genuine_mask = y_true == 0
    imposter_mask = y_true == 1

    n_genuine = genuine_mask.sum()
    n_imposter = imposter_mask.sum()

    # False Reject: genuine users classified as imposter (pred=1)
    frr = (y_pred[genuine_mask] == 1).sum() / max(n_genuine, 1)

    # False Accept: imposters classified as genuine (pred=0)
    far = (y_pred[imposter_mask] == 0).sum() / max(n_imposter, 1)

    return float(far), float(frr)


def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER) and corresponding threshold.

    Args:
        y_true: Binary ground truth (0=genuine, 1=imposter).
        scores: Continuous anomaly scores (higher → more likely imposter).

    Returns:
        eer: Equal Error Rate.
        eer_threshold: Score threshold at which FAR ≈ FRR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the point where |FPR - FNR| is minimised
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.0

    return eer, eer_thresh


def compute_roc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC.

    Returns:
        fpr, tpr arrays and the AUC scalar.
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc


def format_results_table(results: dict) -> str:
    """Pretty-print a dict of {model_name: {FAR, FRR, EER, AUC}} as a table."""
    header = f"{'Model':<20} {'FAR':>8} {'FRR':>8} {'EER':>8} {'AUC':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for name, m in results.items():
        lines.append(
            f"{name:<20} {m.get('FAR', 0):.4f}   {m.get('FRR', 0):.4f}   "
            f"{m.get('EER', 0):.4f}   {m.get('AUC', 0):.4f}"
        )
    lines.append(sep)
    return "\n".join(lines)
