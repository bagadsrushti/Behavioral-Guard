"""
fuse.py — Decision-level fusion of mouse + keystroke modality outputs.

Pipeline:
    1. Time-align mouse and keystroke windows into fixed-time buckets.
    2. Build a meta-feature vector per bucket:
       [mouse_recon_error, mouse_binary_label,
        keystroke_recon_error, keystroke_binary_label]
    3. Train a Random Forest on these meta-features to output final
       genuine / imposter decision.

NOTE: This is basic fixed-window time-bucket alignment.
      CTGAN / TimeGAN alignment is deferred to Phase 2.
"""

import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from src.models.lstm_ae import LSTMAutoencoder, reconstruction_mse, load_checkpoint
from src.utils.config_loader import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time-bucket alignment
# ---------------------------------------------------------------------------

def _assign_time_buckets(
    manifest: pd.DataFrame,
    raw_dir: str,
    modality: str,
    bucket_seconds: float,
) -> pd.DataFrame:
    """Assign each window an integer time-bucket index based on the raw
    session's start timestamp and the window's ordinal position.

    We estimate each window's centre time from the raw CSV's first timestamp
    plus the window index × approximate inter-window interval.
    """
    records = []
    for _, row in manifest.iterrows():
        npy_path = row["npy_path"]
        session_id = row["session_id"]
        label = row["label"]
        n_windows = row["num_windows"]

        # Derive raw CSV path to get the session's first timestamp
        csv_name = f"{modality}_{session_id}.csv"
        csv_path = os.path.join(raw_dir, csv_name)
        if os.path.isfile(csv_path):
            raw_df = pd.read_csv(csv_path, nrows=2)
            t0 = raw_df["timestamp"].iloc[0]
            # Rough session duration from last event
            raw_full = pd.read_csv(csv_path, usecols=["timestamp"])
            t_end = raw_full["timestamp"].iloc[-1]
            duration = t_end - t0
        else:
            t0 = 0.0
            duration = n_windows * bucket_seconds  # fallback estimate

        for wi in range(n_windows):
            # Approximate centre time of this window
            if n_windows > 1:
                t_centre = t0 + (wi / (n_windows - 1)) * duration
            else:
                t_centre = t0
            bucket = int((t_centre - t0) // bucket_seconds)
            records.append(
                {
                    "session_id": session_id,
                    "label": label,
                    "npy_path": npy_path,
                    "window_idx": wi,
                    "t_centre": t_centre,
                    "bucket": bucket,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Build meta-feature matrix
# ---------------------------------------------------------------------------

def build_meta_features(
    mouse_manifest: pd.DataFrame,
    keystroke_manifest: pd.DataFrame,
    mouse_model: LSTMAutoencoder,
    keystroke_model: LSTMAutoencoder,
    mouse_threshold: float,
    keystroke_threshold: float,
    cfg: dict,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute per-bucket meta-features and ground-truth labels.

    Returns:
        X_meta: (N_buckets, 4) — [mouse_err, mouse_pred, ks_err, ks_pred]
        y_meta: (N_buckets,)   — 0 = genuine, 1 = imposter
        info:   DataFrame with session_id, bucket, label
    """
    raw_dir = cfg["paths"]["raw_data_dir"]
    bucket_sec = cfg["fusion"]["time_bucket_seconds"]

    mouse_buckets = _assign_time_buckets(mouse_manifest, raw_dir, "mouse", bucket_sec)
    ks_buckets = _assign_time_buckets(keystroke_manifest, raw_dir, "keystroke", bucket_sec)

    # Pre-compute reconstruction errors for every window
    mouse_model.eval()
    keystroke_model.eval()

    def _errors_for_manifest(manifest_df, model):
        all_errors = {}
        for npy_path, grp in manifest_df.groupby("npy_path"):
            data = np.load(npy_path)
            if data.shape[0] == 0:
                continue
            tensor = torch.tensor(data, dtype=torch.float32).to(device)
            errs = reconstruction_mse(model, tensor, reduction="per_sample").cpu().numpy()
            for _, row in grp.iterrows():
                wi = row["window_idx"]
                if wi < len(errs):
                    all_errors[(row["session_id"], row["bucket"], row["window_idx"])] = errs[wi]
        return all_errors

    mouse_errors = _errors_for_manifest(mouse_buckets, mouse_model)
    ks_errors = _errors_for_manifest(ks_buckets, keystroke_model)

    # Aggregate by (session_id, bucket): take mean error in each bucket
    def _bucket_agg(bucket_df, errors_dict):
        agg = {}
        for _, row in bucket_df.iterrows():
            key = (row["session_id"], row["bucket"], row["window_idx"])
            bid = (row["session_id"], row["bucket"])
            err = errors_dict.get(key, np.nan)
            if bid not in agg:
                agg[bid] = {"errors": [], "label": row["label"]}
            agg[bid]["errors"].append(err)
        result = {}
        for bid, info in agg.items():
            result[bid] = {
                "mean_error": float(np.nanmean(info["errors"])),
                "label": info["label"],
            }
        return result

    mouse_agg = _bucket_agg(mouse_buckets, mouse_errors)
    ks_agg = _bucket_agg(ks_buckets, ks_errors)

    # Join on shared (session_id, bucket)
    shared_keys = set(mouse_agg.keys()) & set(ks_agg.keys())
    if not shared_keys:
        log.warning("No overlapping time-buckets between modalities.")

    rows = []
    for key in sorted(shared_keys):
        m = mouse_agg[key]
        k = ks_agg[key]
        mouse_pred = 1 if m["mean_error"] > mouse_threshold else 0
        ks_pred = 1 if k["mean_error"] > keystroke_threshold else 0
        label_val = 0 if m["label"] == "genuine" else 1
        rows.append(
            {
                "session_id": key[0],
                "bucket": key[1],
                "mouse_error": m["mean_error"],
                "mouse_pred": mouse_pred,
                "keystroke_error": k["mean_error"],
                "keystroke_pred": ks_pred,
                "label": label_val,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return np.empty((0, 4)), np.empty((0,)), df

    X = df[["mouse_error", "mouse_pred", "keystroke_error", "keystroke_pred"]].values
    y = df["label"].values
    return X.astype(np.float32), y.astype(np.int64), df


# ---------------------------------------------------------------------------
# Train Random Forest meta-model
# ---------------------------------------------------------------------------

def train_fusion_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
) -> RandomForestClassifier:
    """Train the Random Forest meta-classifier."""
    rf_cfg = cfg["fusion"]["random_forest"]
    clf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        random_state=rf_cfg["random_state"],
    )
    clf.fit(X_train, y_train)
    log.info(
        "Fusion RF trained — n_estimators=%d, samples=%d",
        rf_cfg["n_estimators"],
        len(y_train),
    )
    return clf


def save_fusion_model(clf: RandomForestClassifier, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    log.info("Fusion model saved → %s", path)


def load_fusion_model(path: str) -> RandomForestClassifier:
    with open(path, "rb") as f:
        clf = pickle.load(f)
    log.info("Fusion model loaded ← %s", path)
    return clf
