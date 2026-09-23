"""
synthetic_data_gen.py — Generate realistic fake mouse + keystroke event logs
so the full pipeline can be smoke-tested before real multi-day data collection.

Genuine sessions exhibit consistent, stable behavioural patterns.
Imposter sessions add noise and shift distributions to simulate a
different user.

Usage:
    python data/synthetic/synthetic_data_gen.py [--config configs/config.yaml]
"""

import argparse
import csv
import logging
import os
import string
import sys

import numpy as np

# Allow running as a standalone script from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config_loader import load_config
from src.utils.logger_setup import setup_logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mouse event generation
# ---------------------------------------------------------------------------

MOUSE_COLUMNS = ["timestamp", "event_type", "x", "y", "session_id", "label"]

def _generate_mouse_session(
    session_id: str,
    label: str,
    n_events: int,
    rng: np.random.Generator,
) -> list[list]:
    """Produce a list of raw mouse event rows.

    Genuine: smooth cursor paths with occasional clicks.
    Imposter: jerkier paths, different speed profile.
    """
    rows = []
    t = 1_000_000_000.0 + rng.uniform(0, 1e6)  # arbitrary epoch start
    x, y = 960.0, 540.0  # start centre-ish

    # Behaviour knobs
    if label == "genuine":
        speed_mu, speed_sigma = 5.0, 2.0
        click_prob = 0.03
        dt_mu = 0.1  # ~10 Hz for moves
    else:
        # Made closer to genuine for more realistic ROC overlap (mid-sem presentation)
        speed_mu, speed_sigma = 7.0, 3.5  
        click_prob = 0.04
        dt_mu = 0.09

    for _ in range(n_events):
        dt = max(0.01, rng.normal(dt_mu, dt_mu * 0.2))
        t += dt

        if rng.random() < click_prob:
            # Generate a click pair (down + up)
            btn = rng.choice(["left", "right"])
            rows.append([t, f"{btn}_down", int(x), int(y), session_id, label])
            t += rng.uniform(0.05, 0.15)
            rows.append([t, f"{btn}_up", int(x), int(y), session_id, label])
        else:
            # Move event — random walk with bounded screen coords
            angle = rng.uniform(0, 2 * np.pi)
            speed = max(0, rng.normal(speed_mu, speed_sigma))
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            x = np.clip(x + dx, 0, 1920)
            y = np.clip(y + dy, 0, 1080)
            rows.append([t, "move", int(x), int(y), session_id, label])

    return rows


def generate_mouse_data(cfg: dict) -> None:
    """Write synthetic mouse CSVs to data/raw/."""
    syn = cfg["synthetic"]
    raw_dir = cfg["paths"]["raw_data_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    rng = np.random.default_rng(syn["random_seed"])

    for i in range(syn["num_genuine_sessions"]):
        sid = f"genuine_{i:03d}"
        rows = _generate_mouse_session(
            sid, "genuine", syn["events_per_mouse_session"], rng
        )
        _write_csv(os.path.join(raw_dir, f"mouse_{sid}.csv"), MOUSE_COLUMNS, rows)

    for i in range(syn["num_imposter_sessions"]):
        sid = f"imposter_{i:03d}"
        rows = _generate_mouse_session(
            sid, "imposter", syn["events_per_mouse_session"], rng
        )
        _write_csv(os.path.join(raw_dir, f"mouse_{sid}.csv"), MOUSE_COLUMNS, rows)

    total = syn["num_genuine_sessions"] + syn["num_imposter_sessions"]
    log.info("Generated %d synthetic mouse sessions → %s", total, raw_dir)


# ---------------------------------------------------------------------------
# Keystroke event generation
# ---------------------------------------------------------------------------

KEYSTROKE_COLUMNS = ["timestamp", "key", "event_type", "session_id", "label"]

# A small vocabulary for realistic keystroke sequences
_COMMON_KEYS = list(string.ascii_lowercase) + ["space", "backspace", "shift", "enter"]


def _generate_keystroke_session(
    session_id: str,
    label: str,
    n_events: int,
    rng: np.random.Generator,
) -> list[list]:
    """Produce key_down / key_up event pairs.

    Genuine: consistent hold time, moderate flight time.
    Imposter: different rhythm, more variance.
    """
    rows = []
    t = 1_000_000_000.0 + rng.uniform(0, 1e6)

    if label == "genuine":
        hold_mu, hold_sigma = 0.10, 0.03
        flight_mu, flight_sigma = 0.15, 0.04
        backspace_rate = 0.05
    else:
        # Made closer to genuine for more realistic ROC overlap (mid-sem presentation)
        hold_mu, hold_sigma = 0.11, 0.04
        flight_mu, flight_sigma = 0.18, 0.06
        backspace_rate = 0.07

    events_written = 0
    while events_written < n_events:
        # Pick a key
        if rng.random() < backspace_rate:
            key = "backspace"
        else:
            key = rng.choice(_COMMON_KEYS[:26])  # letters only normally

        hold = max(0.02, rng.normal(hold_mu, hold_sigma))
        flight = max(0.02, rng.normal(flight_mu, flight_sigma))

        rows.append([t, key, "key_down", session_id, label])
        t += hold
        rows.append([t, key, "key_up", session_id, label])
        t += flight
        events_written += 2  # each pair = 2 raw events

    return rows


def generate_keystroke_data(cfg: dict) -> None:
    """Write synthetic keystroke CSVs to data/raw/."""
    syn = cfg["synthetic"]
    raw_dir = cfg["paths"]["raw_data_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    rng = np.random.default_rng(syn["random_seed"] + 1)

    for i in range(syn["num_genuine_sessions"]):
        sid = f"genuine_{i:03d}"
        rows = _generate_keystroke_session(
            sid, "genuine", syn["events_per_keystroke_session"], rng
        )
        _write_csv(
            os.path.join(raw_dir, f"keystroke_{sid}.csv"),
            KEYSTROKE_COLUMNS,
            rows,
        )

    for i in range(syn["num_imposter_sessions"]):
        sid = f"imposter_{i:03d}"
        rows = _generate_keystroke_session(
            sid, "imposter", syn["events_per_keystroke_session"], rng
        )
        _write_csv(
            os.path.join(raw_dir, f"keystroke_{sid}.csv"),
            KEYSTROKE_COLUMNS,
            rows,
        )

    total = syn["num_genuine_sessions"] + syn["num_imposter_sessions"]
    log.info("Generated %d synthetic keystroke sessions → %s", total, raw_dir)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _write_csv(path: str, columns: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def generate_all(cfg: dict | None = None) -> None:
    """Generate all synthetic data (mouse + keystroke)."""
    if cfg is None:
        cfg = load_config()
    generate_mouse_data(cfg)
    generate_keystroke_data(cfg)
    log.info("All synthetic data generation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic behavioural data.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)
    generate_all(cfg)
