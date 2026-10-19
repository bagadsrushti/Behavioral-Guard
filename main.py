"""
main.py — Orchestrates the full Behavioral Guard pipeline end-to-end.

Stages:
    1. Generate synthetic data (if no raw data exists)
    2. Extract features (mouse + keystroke)
    3. Train LSTM-AE models (mouse + keystroke)
    4. Build fusion meta-features & train Random Forest
    5. Evaluate all three models and produce results

Usage:
    python main.py [--config configs/config.yaml] [--stage STAGE]

Stages: all | generate | features | train | fuse | evaluate
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

from src.utils.config_loader import load_config
from src.utils.logger_setup import setup_logging

log = logging.getLogger(__name__)

VALID_STAGES = {"all", "generate", "features", "train", "fuse", "evaluate"}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_generate(cfg: dict, force: bool = False) -> None:
    """Stage 1: Generate synthetic data (development/testing only).

    Safeguard: If raw CSV data already exists in the configured raw_data_dir,
    generation is skipped to prevent overwriting existing data unless force=True.
    """
    from data.synthetic.synthetic_data_gen import generate_all

    log.info("=" * 60)
    log.info("STAGE: Synthetic Data Generation [SYNTHETIC / DEVELOPMENT ONLY]")
    log.info("=" * 60)

    raw_dir = cfg["paths"]["raw_data_dir"]
    if os.path.isdir(raw_dir):
        existing_csvs = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if existing_csvs and not force:
            log.warning(
                "[SAFEGUARD] Raw data directory '%s' already contains %d CSV files. "
                "Skipping synthetic data generation to preserve existing data. "
                "Use --force to regenerate and overwrite.",
                raw_dir,
                len(existing_csvs),
            )
            return

    generate_all(cfg, force=force)


def stage_features(cfg: dict) -> None:
    """Stage 2: Feature extraction for both modalities."""
    from src.features.mouse_features import process_all_mouse_sessions
    from src.features.keystroke_features import process_all_keystroke_sessions

    log.info("=" * 60)
    log.info("STAGE: Feature Engineering")
    log.info("=" * 60)

    raw_dir = cfg["paths"]["raw_data_dir"]
    proc_dir = cfg["paths"]["processed_data_dir"]

    process_all_mouse_sessions(raw_dir, proc_dir, cfg)
    process_all_keystroke_sessions(raw_dir, proc_dir, cfg)


def stage_train(cfg: dict) -> dict:
    """Stage 3: Train both LSTM-AE models.

    Returns dict with both modality results for downstream stages.
    """
    from src.models.train_mouse import train_mouse_model
    from src.models.train_keystroke import train_keystroke_model

    log.info("=" * 60)
    log.info("STAGE: Model Training")
    log.info("=" * 60)

    mouse_result = train_mouse_model(cfg)
    keystroke_result = train_keystroke_model(cfg)

    return {"mouse": mouse_result, "keystroke": keystroke_result}


def stage_fuse(cfg: dict) -> None:
    """Stage 4: Build fusion meta-features and train the Random Forest."""
    from src.models.lstm_ae import load_checkpoint
    from src.fusion.fuse import (
        build_meta_features,
        train_fusion_model,
        save_fusion_model,
    )

    log.info("=" * 60)
    log.info("STAGE: Fusion")
    log.info("=" * 60)

    processed_dir = cfg["paths"]["processed_data_dir"]
    ckpt_dir = cfg["paths"]["model_checkpoint_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_cfg = cfg["evaluation"]

    # Load models
    mouse_ckpt = os.path.join(ckpt_dir, cfg["model_mouse"]["checkpoint_file"])
    ks_ckpt = os.path.join(ckpt_dir, cfg["model_keystroke"]["checkpoint_file"])
    mouse_model, mouse_extra = load_checkpoint(mouse_ckpt, device)
    ks_model, ks_extra = load_checkpoint(ks_ckpt, device)
    mouse_threshold = mouse_extra.get("threshold", float("inf"))
    ks_threshold = ks_extra.get("threshold", float("inf"))

    # Load manifests — use train+val split for fusion training
    mouse_manifest = pd.read_csv(
        os.path.join(processed_dir, "mouse_manifest.csv")
    )
    ks_manifest = pd.read_csv(
        os.path.join(processed_dir, "keystroke_manifest.csv")
    )

    # For fusion training, use train+val genuine + train imposter
    n_m_genuine = len(mouse_manifest[mouse_manifest["label"] == "genuine"])
    n_k_genuine = len(ks_manifest[ks_manifest["label"] == "genuine"])
    m_train_end = max(1, int(n_m_genuine * (eval_cfg["train_ratio"] + eval_cfg["val_ratio"])))
    k_train_end = max(1, int(n_k_genuine * (eval_cfg["train_ratio"] + eval_cfg["val_ratio"])))

    imposter_train_ratio = cfg.get("fusion", {}).get("imposter_train_ratio", 0.6)
    all_imp_sessions = mouse_manifest[mouse_manifest["label"] == "imposter"]["session_id"].unique()
    n_imp = len(all_imp_sessions)
    imp_train_end = max(1, int(n_imp * imposter_train_ratio))
    train_imp_sessions = all_imp_sessions[:imp_train_end]

    import json
    with open(os.path.join(processed_dir, "fusion_train_imp_sessions.json"), "w") as f:
        json.dump(list(train_imp_sessions), f)
    log.info("Saved %d imposter sessions for fusion training.", len(train_imp_sessions))

    train_mouse = pd.concat([
        mouse_manifest[mouse_manifest["label"] == "genuine"].iloc[:m_train_end],
        mouse_manifest[mouse_manifest["session_id"].isin(train_imp_sessions)],
    ])
    train_ks = pd.concat([
        ks_manifest[ks_manifest["label"] == "genuine"].iloc[:k_train_end],
        ks_manifest[ks_manifest["session_id"].isin(train_imp_sessions)],
    ])

    X_meta, y_meta, info_df = build_meta_features(
        train_mouse, train_ks,
        mouse_model, ks_model,
        mouse_threshold, ks_threshold,
        cfg, device,
    )

    if X_meta.shape[0] == 0:
        log.error("No meta-feature samples produced — cannot train fusion model.")
        return

    clf = train_fusion_model(X_meta, y_meta, cfg)
    save_fusion_model(clf, os.path.join(ckpt_dir, "fusion_rf.pkl"))


def stage_evaluate(cfg: dict) -> None:
    """Stage 5: Evaluate all models."""
    from src.evaluation.evaluate import run_evaluation

    log.info("=" * 60)
    log.info("STAGE: Evaluation")
    log.info("=" * 60)

    run_evaluation(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Guard — Continuous Authentication Pipeline"
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=sorted(VALID_STAGES),
        help="Run a specific pipeline stage (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing raw datasets during synthetic data generation",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    log.info("Behavioral Guard pipeline starting — stage=%s, force=%s", args.stage, args.force)

    stage = args.stage

    try:
        if stage in ("all", "generate"):
            stage_generate(cfg, force=args.force)
        if stage in ("all", "features"):
            stage_features(cfg)
        if stage in ("all", "train"):
            stage_train(cfg)
        if stage in ("all", "fuse"):
            stage_fuse(cfg)
        if stage in ("all", "evaluate"):
            stage_evaluate(cfg)
    except Exception:
        log.exception("Pipeline failed during stage '%s'", stage)
        sys.exit(1)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
