# Behavioral Guard

**Continuous authentication via behavioral biometrics (mouse dynamics + keystroke dynamics), using LSTM-Autoencoder anomaly detection and decision-level fusion.** This project implements a single-user continuous authentication pipeline that learns the genuine user's behavioural signature from mouse and keystroke patterns, then flags anomalous (imposter) behaviour in real time. The architecture adapts concepts from Li et al.'s MBBFAuth framework — specifically feature engineering, LSTM-Autoencoder reconstruction-based anomaly detection, and decision-level fusion — to a two-modality (mouse + keystroke) desktop setting.

---

## Reference

> Li, W., et al., **"MBBFAuth: Multimodal Behavioral Biometrics Fusion for Continuous Authentication on Non-Portable Devices,"** *IEEE Transactions on Information Forensics and Security (TIFS)*, 2024.

This project adapts their feature engineering pipeline, LSTM-Autoencoder one-class anomaly detection, and decision-level fusion concepts to a two-modality continuous authentication prototype.

---

## Project Structure

```
Behavioral Guard/
├── configs/
│   └── config.yaml              # Single config file — all hyperparameters
├── data/
│   ├── raw/                     # Logged / generated raw event CSVs
│   ├── processed/               # Extracted windowed feature arrays (.npy)
│   └── synthetic/
│       └── synthetic_data_gen.py # Generates fake mouse + keystroke data
├── models_checkpoints/          # Saved model weights + fusion RF
├── results/                     # Evaluation tables, ROC plots, pipeline log
├── src/
│   ├── logging/
│   │   ├── mouse_logger.py      # Background mouse event capture
│   │   └── keystroke_logger.py  # Background keystroke event capture
│   ├── features/
│   │   ├── mouse_features.py    # 20-feature kinematic engineering
│   │   └── keystroke_features.py# 6-feature timing engineering
│   ├── models/
│   │   ├── lstm_ae.py           # Shared LSTM-AE architecture class
│   │   ├── train_mouse.py       # One-class training on mouse data
│   │   └── train_keystroke.py   # One-class training on keystroke data
│   ├── fusion/
│   │   └── fuse.py              # Time-bucket alignment + RF meta-classifier
│   ├── evaluation/
│   │   ├── metrics.py           # FAR, FRR, EER, ROC utilities
│   │   └── evaluate.py          # End-to-end evaluation pipeline
│   └── utils/
│       ├── config_loader.py     # YAML config loader
│       └── logger_setup.py      # Python logging setup
├── main.py                      # Full pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Safe Execution & Safeguards

> [!IMPORTANT]
> **Data Overwrite Safeguard:** By default, running `python main.py` or `python main.py --stage generate` will **never** overwrite existing raw data in `data/raw/`. If existing CSVs are detected, generation is automatically skipped. To explicitly regenerate synthetic data and overwrite existing files, pass the `--force` flag.

> [!NOTE]
> **Keystroke Privacy Guarantee:** The keystroke logger records only event timing and coarse key categories (`letter`, `digit`, `space`, `backspace`, `enter`, `tab`, `modifier`, `punctuation`, `other`). It **never** logs or persists actual typed characters or passwords.

### Running Pipeline Stages Individually

```bash
# 1. Generate synthetic data (skipped if data/raw already has files; use --force to overwrite)
python main.py --stage generate
python main.py --stage generate --force

# 2. Extract features from raw CSVs into windowed .npy arrays (safe, non-destructive to raw data)
python main.py --stage features

# 3. Train LSTM-AE models on genuine sessions
python main.py --stage train

# 4. Build fusion meta-features and train Random Forest classifier
python main.py --stage fuse

# 5. Evaluate models and produce ROC plots in results/
python main.py --stage evaluate

# Run all stages end-to-end (skips generation if data/raw exists)
python main.py
```

### Running Safety & Regression Tests

```bash
# Run the Phase 0 safety and baseline test suite
python -m unittest tests/test_phase0_safety.py -v
```

### Live Data Collection (Privacy-Preserving)

```bash
# Record mouse events (Ctrl+C to stop)
python -m src.logging.mouse_logger --session_id s01 --label genuine

# Record keystroke events — records key categories only, no plaintext (Ctrl+C to stop)
python -m src.logging.keystroke_logger --session_id s01 --label genuine
```

### Custom Config

```bash
python main.py --config path/to/custom_config.yaml
```

---

## Configuration

All hyperparameters, paths, window sizes, and thresholds are controlled by `configs/config.yaml`. Key sections:

| Section            | Controls                                              |
|--------------------|-------------------------------------------------------|
| `paths`            | Data directories, checkpoint dir, results dir         |
| `collection`       | Mouse sampling rate, output format                    |
| `features`         | Window size and stride for each modality              |
| `model_mouse`      | LSTM-AE architecture, LR, batch size, epochs (mouse)  |
| `model_keystroke`  | LSTM-AE architecture, LR, batch size, epochs (keys)   |
| `fusion`           | Time bucket size, RF hyperparameters                  |
| `evaluation`       | Train/val/test split ratios                           |
| `synthetic`        | Number of sessions, events per session, random seed   |
| `logging`          | Log level, log file path, format string               |

---

## Evaluation Output

After running the full pipeline, `results/` contains:

- **`evaluation_results.txt`** — Formatted table with FAR, FRR, EER, AUC for each model
- **`roc_mouse.png`** — ROC curve for the mouse-only LSTM-AE
- **`roc_keystroke.png`** — ROC curve for the keystroke-only LSTM-AE
- **`roc_fused.png`** — ROC curve for the fused Random Forest model
- **`pipeline.log`** — Full pipeline execution log

---

## Implemented in This Phase (Phase 1)

- [x] Mouse event logger (background capture, ~10 Hz moves, event-triggered clicks)
- [x] Keystroke event logger (background capture, key_down/key_up)
- [x] Mouse feature engineering (20 kinematic features, sliding windows)
- [x] Keystroke feature engineering (6 timing features, sliding windows)
- [x] Shared LSTM-Autoencoder architecture (Encoder: 2 LSTM + FC, Decoder: FC + 2 LSTM + FC)
- [x] One-class training on genuine data (MSE reconstruction loss)
- [x] Per-modality threshold setting (95th percentile of genuine validation error)
- [x] Basic fixed-time-window fusion with Random Forest meta-classifier
- [x] Evaluation with FAR, FRR, EER metrics and ROC curves
- [x] Synthetic data generator for end-to-end smoke testing
- [x] Centralised YAML config (no hardcoded magic numbers)
- [x] Python logging throughout the pipeline
- [x] Checkpoint save/load for all models

---

## Planned for Phase 2

- [ ] **Keystroke-mouse temporal alignment via CTGAN / TimeGAN** — learned alignment of asynchronous modality streams to replace the fixed-time-bucket approach
- [ ] **Full 8-model stacking comparison** — exhaustive meta-model comparison (LR, SVM, KNN, DT, RF, XGBoost, LightGBM, MLP) with cross-validated selection
- [ ] **Adaptive learning module** — online model updates to track gradual behavioural drift over weeks/months
- [ ] **Imitation-attack evaluation** — adversarial testing with targeted mimicry attacks to assess robustness under realistic threat models

---

## License

This project is for academic / research use.
