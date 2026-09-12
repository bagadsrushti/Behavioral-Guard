"""
mouse_features.py — Convert raw mouse event logs into windowed feature sequences.

Features per event (20 total):
    event_type_encoded, x, y, delta_t, delta_x, delta_y,
    d (Euclidean distance), v, v_x, v_y, acc, acc_x, acc_y,
    jerk, jerk_x, jerk_y, theta, direction, k (curvature), r (radius)

The module produces fixed-length sliding windows from consecutive events,
handling first-event-in-session edge cases by zero-filling derivatives.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

log = logging.getLogger(__name__)

# Mapping event_type strings → numeric codes
EVENT_TYPE_MAP = {
    "move": 0,
    "left_down": 1,
    "left_up": 2,
    "right_down": 3,
    "right_up": 4,
}


# ---------------------------------------------------------------------------
# Per-event feature computation
# ---------------------------------------------------------------------------

def _compute_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-event kinematic features from raw mouse log columns.

    Expects columns: timestamp, event_type, x, y
    Returns a DataFrame with 20 feature columns (float32).
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Encode event type
    df["event_type_enc"] = (
        df["event_type"].map(EVENT_TYPE_MAP).fillna(-1).astype(np.float32)
    )

    # Temporal & spatial deltas
    df["delta_t"] = df["timestamp"].diff().fillna(0).astype(np.float32)
    df["delta_x"] = df["x"].diff().fillna(0).astype(np.float32)
    df["delta_y"] = df["y"].diff().fillna(0).astype(np.float32)

    # Euclidean distance
    df["d"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2).astype(np.float32)

    # Velocity (guard against zero delta_t)
    safe_dt = df["delta_t"].replace(0, np.nan)
    df["v"] = (df["d"] / safe_dt).fillna(0).astype(np.float32)
    df["v_x"] = (df["delta_x"] / safe_dt).fillna(0).astype(np.float32)
    df["v_y"] = (df["delta_y"] / safe_dt).fillna(0).astype(np.float32)

    # Acceleration
    dv = df["v"].diff().fillna(0)
    dv_x = df["v_x"].diff().fillna(0)
    dv_y = df["v_y"].diff().fillna(0)
    df["acc"] = (dv / safe_dt).fillna(0).astype(np.float32)
    df["acc_x"] = (dv_x / safe_dt).fillna(0).astype(np.float32)
    df["acc_y"] = (dv_y / safe_dt).fillna(0).astype(np.float32)

    # Jerk
    da = df["acc"].diff().fillna(0)
    da_x = df["acc_x"].diff().fillna(0)
    da_y = df["acc_y"].diff().fillna(0)
    df["jerk"] = (da / safe_dt).fillna(0).astype(np.float32)
    df["jerk_x"] = (da_x / safe_dt).fillna(0).astype(np.float32)
    df["jerk_y"] = (da_y / safe_dt).fillna(0).astype(np.float32)

    # Angle theta (bearing in radians)
    df["theta"] = np.arctan2(df["delta_y"], df["delta_x"]).fillna(0).astype(np.float32)

    # Direction change (difference of consecutive theta)
    df["direction"] = df["theta"].diff().fillna(0).astype(np.float32)

    # Curvature k = |d(theta)/ds| where ds = d
    d_theta = df["theta"].diff().fillna(0)
    safe_d = df["d"].replace(0, np.nan)
    df["k"] = (d_theta.abs() / safe_d).fillna(0).astype(np.float32)

    # Radius of curvature r = 1 / k (cap at large value to avoid inf)
    safe_k = df["k"].replace(0, np.nan)
    df["r"] = (1.0 / safe_k).fillna(0).clip(-1e6, 1e6).astype(np.float32)

    # Replace any remaining inf / NaN with 0
    feature_cols = [
        "event_type_enc", "x", "y", "delta_t", "delta_x", "delta_y",
        "d", "v", "v_x", "v_y", "acc", "acc_x", "acc_y",
        "jerk", "jerk_x", "jerk_y", "theta", "direction", "k", "r",
    ]
    features = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding-window segmentation
# ---------------------------------------------------------------------------

def _sliding_windows(
    features: np.ndarray, window_size: int, stride: int
) -> np.ndarray:
    """Segment a 2-D feature array into overlapping windows.

    Returns shape (num_windows, window_size, num_features).
    """
    n = len(features)
    if n < window_size:
        log.warning(
            "Feature array length (%d) < window_size (%d); skipping.",
            n,
            window_size,
        )
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32)

    starts = range(0, n - window_size + 1, stride)
    windows = np.array(
        [features[s : s + window_size] for s in starts], dtype=np.float32
    )
    return windows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_mouse_features(
    raw_csv_path: str,
    cfg: dict | None = None,
) -> Tuple[np.ndarray, dict]:
    """End-to-end: raw CSV → windowed feature array.

    Returns:
        windows: ndarray of shape (N, window_size, 20)
        meta: dict with session_id, label, num_events, num_windows
    """
    if cfg is None:
        cfg = load_config()

    window_size = cfg["features"]["mouse"]["window_size"]
    stride = cfg["features"]["mouse"]["window_stride"]

    log.info("Loading raw mouse log: %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    required = {"timestamp", "event_type", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {raw_csv_path}: {required - set(df.columns)}"
        )

    session_id = df["session_id"].iloc[0] if "session_id" in df.columns else "unknown"
    label = df["label"].iloc[0] if "label" in df.columns else "unknown"

    features = _compute_event_features(df)
    windows = _sliding_windows(features.values, window_size, stride)

    meta = {
        "session_id": session_id,
        "label": label,
        "num_events": len(df),
        "num_windows": windows.shape[0],
    }
    log.info(
        "Mouse features extracted — session=%s, events=%d, windows=%d",
        session_id,
        meta["num_events"],
        meta["num_windows"],
    )
    return windows, meta


def process_all_mouse_sessions(
    raw_dir: str,
    output_dir: str,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Process every mouse_*.csv in *raw_dir* and save .npy feature files.

    Returns a manifest DataFrame listing (session_id, label, npy_path, num_windows).
    """
    if cfg is None:
        cfg = load_config()

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.startswith("mouse_") or not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(raw_dir, fname)
        try:
            windows, meta = extract_mouse_features(csv_path, cfg)
        except Exception as exc:
            log.error("Failed to process %s: %s", csv_path, exc)
            continue

        npy_name = fname.replace(".csv", ".npy")
        npy_path = os.path.join(output_dir, npy_name)
        np.save(npy_path, windows)

        records.append(
            {
                "session_id": meta["session_id"],
                "label": meta["label"],
                "npy_path": npy_path,
                "num_windows": meta["num_windows"],
            }
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(
        os.path.join(output_dir, "mouse_manifest.csv"), index=False
    )
    log.info("Mouse feature extraction complete — %d sessions processed.", len(manifest))
    return manifest
