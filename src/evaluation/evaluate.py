"""
evaluate.py — End-to-end evaluation of mouse-only, keystroke-only, and fused models.

Computes FAR, FRR, EER for each model, prints a formatted table, and saves
ROC curve plots to /results/.
"""

import logging
import os
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import (
    compute_far_frr,
    compute_eer,
    compute_roc,
    format_results_table,
)
from src.models.lstm_ae import (
    LSTMAutoencoder,
    load_checkpoint,
    reconstruction_mse,
)
from src.fusion.fuse import (
    build_meta_features,
    load_fusion_model,
)
from src.utils.config_loader import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_data(
    manifest_path: str, eval_cfg: dict, include_imposter: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Load test-split genuine windows + all imposter windows.

    Returns:
        X: concatenated windows (N, seq, feat)
        y: labels (0=genuine, 1=imposter)
    """
    manifest = pd.read_csv(manifest_path)

    genuine = manifest[manifest["label"] == "genuine"].reset_index(drop=True)
    imposter = manifest[manifest["label"] == "imposter"].reset_index(drop=True)

    n = len(genuine)
    n_train = max(1, int(n * eval_cfg["train_ratio"]))
    n_val = max(1, int(n * eval_cfg["val_ratio"]))
    test_genuine = genuine.iloc[n_train + n_val :]

    arrays, labels = [], []

    for _, row in test_genuine.iterrows():
        data = np.load(row["npy_path"])
        if data.shape[0] > 0:
            arrays.append(data)
            labels.append(np.zeros(data.shape[0], dtype=np.int64))

    if include_imposter:
        for _, row in imposter.iterrows():
            data = np.load(row["npy_path"])
            if data.shape[0] > 0:
                arrays.append(data)
                labels.append(np.ones(data.shape[0], dtype=np.int64))

    if not arrays:
        return np.empty((0,)), np.empty((0,))

    X = np.concatenate(arrays, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _evaluate_single_modality(
    model: LSTMAutoencoder,
    threshold: float,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
) -> dict:
    """Evaluate one modality and return metrics dict."""
    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    errors = reconstruction_mse(model, tensor, reduction="per_sample").cpu().numpy()

    preds = (errors > threshold).astype(np.int64)
    far, frr = compute_far_frr(y, preds)
    eer, _ = compute_eer(y, errors)
    fpr, tpr, roc_auc = compute_roc(y, errors)

    return {
        "FAR": far,
        "FRR": frr,
        "EER": eer,
        "AUC": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "errors": errors,
        "preds": preds,
    }


def _plot_roc(fpr, tpr, roc_auc, title: str, save_path: str) -> None:
    """Save a ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("ROC curve saved → %s", save_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_evaluation(cfg: dict | None = None) -> dict:
    """Evaluate all three models and produce results.

    Returns dict with per-model metrics.
    """
    if cfg is None:
        cfg = load_config()

    eval_cfg = cfg["evaluation"]
    processed_dir = cfg["paths"]["processed_data_dir"]
    ckpt_dir = cfg["paths"]["model_checkpoint_dir"]
    results_dir = cfg["results_dir"] if "results_dir" in cfg else cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    # ---- Mouse-only evaluation ----
    log.info("=== Mouse-only evaluation ===")
    mouse_ckpt = os.path.join(ckpt_dir, cfg["model_mouse"]["checkpoint_file"])
    mouse_model, mouse_extra = load_checkpoint(mouse_ckpt, device)
    mouse_threshold = mouse_extra.get("threshold", float("inf"))

    mouse_manifest = os.path.join(processed_dir, "mouse_manifest.csv")
    X_m, y_m = _load_test_data(mouse_manifest, eval_cfg)

    if X_m.shape[0] > 0:
        m_res = _evaluate_single_modality(mouse_model, mouse_threshold, X_m, y_m, device)
        all_results["Mouse-only"] = {k: m_res[k] for k in ("FAR", "FRR", "EER", "AUC")}
        _plot_roc(
            m_res["fpr"],
            m_res["tpr"],
            m_res["AUC"],
            "ROC — Mouse LSTM-AE",
            os.path.join(results_dir, "roc_mouse.png"),
        )
    else:
        log.warning("No mouse test data available for evaluation.")
        all_results["Mouse-only"] = {"FAR": 0, "FRR": 0, "EER": 0, "AUC": 0}

    # ---- Keystroke-only evaluation ----
    log.info("=== Keystroke-only evaluation ===")
    ks_ckpt = os.path.join(ckpt_dir, cfg["model_keystroke"]["checkpoint_file"])
    ks_model, ks_extra = load_checkpoint(ks_ckpt, device)
    ks_threshold = ks_extra.get("threshold", float("inf"))

    ks_manifest = os.path.join(processed_dir, "keystroke_manifest.csv")
    X_k, y_k = _load_test_data(ks_manifest, eval_cfg)

    if X_k.shape[0] > 0:
        k_res = _evaluate_single_modality(ks_model, ks_threshold, X_k, y_k, device)
        all_results["Keystroke-only"] = {k: k_res[k] for k in ("FAR", "FRR", "EER", "AUC")}
        _plot_roc(
            k_res["fpr"],
            k_res["tpr"],
            k_res["AUC"],
            "ROC — Keystroke LSTM-AE",
            os.path.join(results_dir, "roc_keystroke.png"),
        )
    else:
        log.warning("No keystroke test data available for evaluation.")
        all_results["Keystroke-only"] = {"FAR": 0, "FRR": 0, "EER": 0, "AUC": 0}

    # ---- Fused evaluation ----
    log.info("=== Fused model evaluation ===")
    fusion_model_path = os.path.join(ckpt_dir, "fusion_rf.pkl")
    if os.path.isfile(fusion_model_path):
        rf_clf = load_fusion_model(fusion_model_path)

        # Rebuild test meta-features
        mouse_mf = pd.read_csv(mouse_manifest)
        ks_mf = pd.read_csv(ks_manifest)

        # Filter to test + imposter sessions
        n_m = len(mouse_mf[mouse_mf["label"] == "genuine"])
        n_k = len(ks_mf[ks_mf["label"] == "genuine"])
        m_train_end = max(1, int(n_m * (eval_cfg["train_ratio"] + eval_cfg["val_ratio"])))
        k_train_end = max(1, int(n_k * (eval_cfg["train_ratio"] + eval_cfg["val_ratio"])))

        test_mouse = pd.concat([
            mouse_mf[mouse_mf["label"] == "genuine"].iloc[m_train_end:],
            mouse_mf[mouse_mf["label"] == "imposter"],
        ])
        test_ks = pd.concat([
            ks_mf[ks_mf["label"] == "genuine"].iloc[k_train_end:],
            ks_mf[ks_mf["label"] == "imposter"],
        ])

        X_fused, y_fused, _ = build_meta_features(
            test_mouse, test_ks,
            mouse_model, ks_model,
            mouse_threshold, ks_threshold,
            cfg, device,
        )

        if X_fused.shape[0] > 0:
            fused_preds = rf_clf.predict(X_fused)
            fused_scores = rf_clf.predict_proba(X_fused)[:, 1]

            far, frr = compute_far_frr(y_fused, fused_preds)
            eer, _ = compute_eer(y_fused, fused_scores)
            fpr, tpr, roc_auc = compute_roc(y_fused, fused_scores)

            all_results["Fused (RF)"] = {
                "FAR": far, "FRR": frr, "EER": eer, "AUC": roc_auc
            }
            _plot_roc(
                fpr, tpr, roc_auc,
                "ROC — Fused (Random Forest)",
                os.path.join(results_dir, "roc_fused.png"),
            )
        else:
            log.warning("No overlapping test buckets for fused evaluation.")
            all_results["Fused (RF)"] = {"FAR": 0, "FRR": 0, "EER": 0, "AUC": 0}
    else:
        log.warning("Fusion model not found at %s — skipping.", fusion_model_path)
        all_results["Fused (RF)"] = {"FAR": 0, "FRR": 0, "EER": 0, "AUC": 0}

    # ---- Print & save results table ----
    table = format_results_table(all_results)
    print("\n" + table + "\n")
    log.info("Evaluation results:\n%s", table)

    table_path = os.path.join(results_dir, "evaluation_results.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    log.info("Results table saved → %s", table_path)

    return all_results
