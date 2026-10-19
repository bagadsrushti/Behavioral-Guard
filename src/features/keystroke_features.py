"""
keystroke_features.py — Convert raw keystroke event logs into windowed feature
sequences.

Features per keystroke (6 total):
    hold_time, flight_time, digraph_latency, trigraph_latency,
    typing_speed, backspace_freq

The module produces fixed-length sliding windows over consecutive keystrokes.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-keystroke feature computation
# ---------------------------------------------------------------------------

def _compute_keystroke_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-keystroke timing and frequency features.

    Input columns: timestamp, key, event_type (key_down / key_up)
    Returns a DataFrame with 6 float32 feature columns, one row per key-press.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Separate press and release events
    presses = df[df["event_type"] == "key_down"].copy().reset_index(drop=True)
    releases = df[df["event_type"] == "key_up"].copy().reset_index(drop=True)

    # Build a dict: key → list of release timestamps (for hold-time matching)
    release_map: dict[str, list[float]] = {}
    for _, row in releases.iterrows():
        release_map.setdefault(row["key"], []).append(row["timestamp"])

    # --- Hold time (dwell) ---
    hold_times = []
    for _, row in presses.iterrows():
        key = row["key"]
        press_ts = row["timestamp"]
        rel_list = release_map.get(key, [])
        # Find the earliest release *after* this press
        matched = [r for r in rel_list if r >= press_ts]
        if matched:
            ht = matched[0] - press_ts
            rel_list.remove(matched[0])  # consume match
        else:
            ht = 0.0
        hold_times.append(ht)

    presses = presses.copy()
    presses["hold_time"] = np.array(hold_times, dtype=np.float32)

    # --- Flight time (inter-key interval between consecutive presses) ---
    presses["flight_time"] = (
        presses["timestamp"].diff().fillna(0).astype(np.float32)
    )

    # --- Digraph latency (time between press[i] and press[i+1]) ---
    # Same as flight_time for consecutive pairs; included for clarity.
    presses["digraph_latency"] = presses["flight_time"]

    # --- Trigraph latency (time spanning 3 consecutive presses) ---
    presses["trigraph_latency"] = (
        (presses["timestamp"] - presses["timestamp"].shift(2))
        .fillna(0)
        .astype(np.float32)
    )

    # --- Typing speed (keys/sec in a local rolling window of 5 presses) ---
    roll = 5
    ts_diff = (
        presses["timestamp"] - presses["timestamp"].shift(roll - 1)
    ).fillna(0)
    safe_diff = ts_diff.replace(0, np.nan)
    presses["typing_speed"] = (
        ((roll - 1) / safe_diff).fillna(0).astype(np.float32)
    )

    # --- Backspace / correction frequency (rolling ratio in window of 10) ---
    is_backspace = presses["key"].str.lower().isin(
        ["backspace", "delete", "back"]
    ).astype(np.float32)
    presses["backspace_freq"] = (
        is_backspace.rolling(window=10, min_periods=1).mean().astype(np.float32)
    )

    feature_cols = [
        "hold_time",
        "flight_time",
        "digraph_latency",
        "trigraph_latency",
        "typing_speed",
        "backspace_freq",
    ]

    features = presses[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# Sliding-window segmentation
# ---------------------------------------------------------------------------

def _sliding_windows(
    features: np.ndarray, window_size: int, stride: int
) -> np.ndarray:
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

def extract_keystroke_features(
    raw_csv_path: str,
    cfg: dict | None = None,
) -> Tuple[np.ndarray, dict]:
    """End-to-end: raw CSV → windowed feature array.

    Returns:
        windows: ndarray of shape (N, window_size, 6)
        meta: dict with session_id, label, num_events, num_windows
    """
    if cfg is None:
        cfg = load_config()

    window_size = cfg["features"]["keystroke"]["window_size"]
    stride = cfg["features"]["keystroke"]["window_stride"]

    log.info("Loading raw keystroke log: %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    if "key_category" in df.columns and "key" not in df.columns:
        df["key"] = df["key_category"]

    required = {"timestamp", "key", "event_type"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {raw_csv_path}: {required - set(df.columns)}"
        )

    session_id = df["session_id"].iloc[0] if "session_id" in df.columns else "unknown"
    label = df["label"].iloc[0] if "label" in df.columns else "unknown"

    features = _compute_keystroke_features(df)
    windows = _sliding_windows(features.values, window_size, stride)

    meta = {
        "session_id": session_id,
        "label": label,
        "num_events": len(df),
        "num_windows": windows.shape[0],
    }
    log.info(
        "Keystroke features extracted — session=%s, events=%d, windows=%d",
        session_id,
        meta["num_events"],
        meta["num_windows"],
    )
    return windows, meta


def process_all_keystroke_sessions(
    raw_dir: str,
    output_dir: str,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Process every keystroke_*.csv in *raw_dir* and save .npy feature files."""
    if cfg is None:
        cfg = load_config()

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.startswith("keystroke_") or not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(raw_dir, fname)
        try:
            windows, meta = extract_keystroke_features(csv_path, cfg)
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
        os.path.join(output_dir, "keystroke_manifest.csv"), index=False
    )
    log.info(
        "Keystroke feature extraction complete — %d sessions processed.",
        len(manifest),
    )
    return manifest
