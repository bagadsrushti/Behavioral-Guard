# 🛡️ Behavioral Guard — Complete Project Deep Analysis

> **Adaptive Multimodal Behavioral Biometrics for Continuous Authentication on Single-User Laptops/PCs**

**Reference Paper:** Li, W., et al., *"MBBFAuth: Multimodal Behavioral Biometrics Fusion for Continuous Authentication on Non-Portable Devices,"* IEEE TIFS, 2024.

**Project Goals:**
1. Working prototype of continuous behavioral authentication
2. IEEE/Scopus publication
3. Patent filing on the adaptive learning mechanism

---

## 📌 Table of Contents

1. [What is the Project?](#1-what-is-the-project)
2. [Problem Statement](#2-problem-statement)
3. [Core Concept — One-Class Anomaly Detection](#3-core-concept--one-class-anomaly-detection)
4. [Technology Stack — What We Used](#4-technology-stack--what-we-used)
5. [Complete Project Architecture](#5-complete-project-architecture)
6. [Project File Structure](#6-project-file-structure)
7. [Data Pipeline — How It Works End-to-End](#7-data-pipeline--how-it-works-end-to-end)
8. [Stage 1: Data Collection & Synthetic Generation](#8-stage-1-data-collection--synthetic-generation)
9. [Stage 2: Feature Engineering — All Formulas](#9-stage-2-feature-engineering--all-formulas)
10. [Stage 3: LSTM-Autoencoder Model — Architecture & Formulas](#10-stage-3-lstm-autoencoder-model--architecture--formulas)
11. [Stage 4: Decision-Level Fusion](#11-stage-4-decision-level-fusion)
12. [Stage 5: Evaluation — Metrics & Formulas](#12-stage-5-evaluation--metrics--formulas)
13. [Current Evaluation Results (Phase 1)](#13-current-evaluation-results-phase-1)
14. [Configuration & Hyperparameters](#14-configuration--hyperparameters)
15. [Current Project Status](#15-current-project-status)
16. [Phase 2 — Planned Work (End-Sem)](#16-phase-2--planned-work-end-sem)
17. [Novel Contributions & Patent Claim](#17-novel-contributions--patent-claim)
18. [Dataset Summary](#18-dataset-summary)
19. [Saved Model Checkpoints](#19-saved-model-checkpoints)
20. [How to Run the Project](#20-how-to-run-the-project)

---

## 1. What is the Project?

**Behavioral Guard** is a **continuous authentication system** that passively verifies a user's identity in real-time by analyzing their behavioral biometrics — specifically **mouse movement dynamics** and **keystroke typing patterns**. Unlike traditional authentication (passwords, PINs, fingerprints) that verifies identity only at login, Behavioral Guard continuously monitors the session and flags if a different person takes over the device.

The system adapts concepts from the **MBBFAuth framework** (Li et al., IEEE TIFS 2024) — specifically:
- Feature engineering pipelines for mouse & keystroke modalities
- LSTM-Autoencoder reconstruction-based anomaly detection
- Decision-level fusion via meta-classifier stacking

Applied to a **two-modality (mouse + keystroke) desktop setting** for single-user laptops/PCs.

---

## 2. Problem Statement

| Problem | Impact |
|---------|--------|
| Traditional login-time authentication verifies identity **only once** | After login, the session is trusted indefinitely |
| If someone else takes over the keyboard/mouse, the system has no way to detect it | Complete security gap post-login |
| Static biometrics (fingerprint, face) require active user participation | Not passive, not continuous |
| User behavior naturally drifts over time (fatigue, injury, new habits) | Static profiles degrade, causing increased false rejections |

**Our Solution:** Treat continuous authentication as a **one-class anomaly detection** problem. The system learns "what normal behavior looks like" for the genuine user and flags any deviation as potentially suspicious — without ever needing imposter data at training time.

---

## 3. Core Concept — One-Class Anomaly Detection

The fundamental principle:
- **Only genuine user data is available** at training time (an imposter is, by definition, unknown in advance)
- The model learns to **reconstruct** genuine behavior patterns
- When it encounters **different** behavior (imposter), the reconstruction will be **poor** — high error
- High reconstruction error = anomaly = flag as imposter

```
If MSE(new_sequence) > threshold  →  IMPOSTER (anomaly)
Else                               →  GENUINE (normal)
```

---

## 4. Technology Stack — What We Used

### 4.1 Programming Language & Core Libraries

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Primary language |
| **PyTorch** | >= 2.0.0 | Deep learning framework (LSTM-Autoencoder) |
| **NumPy** | >= 1.24.0 | Numerical computation, array operations |
| **Pandas** | >= 2.0.0 | Data manipulation, CSV handling, manifests |
| **scikit-learn** | >= 1.3.0 | Random Forest classifier, StandardScaler, ROC/AUC metrics |
| **Matplotlib** | >= 3.7.0 | ROC curve plots, visualization |
| **PyYAML** | >= 6.0 | Configuration file parsing |
| **pynput** | >= 1.7.6 | Real-time mouse & keyboard event capture |

### 4.2 How Each Technology is Used

| Technology | How It Is Used |
|-----------|----------------|
| **PyTorch** | Defines the `LSTMAutoencoder` neural network architecture (Encoder: 2 LSTM + FC, Decoder: FC + 2 LSTM + FC). Handles training loops, backpropagation with `Adam` optimizer, MSE loss computation, GPU/CPU device management, model checkpoint save/load |
| **NumPy** | Feature array operations, sliding window segmentation, `.npy` file I/O for processed features, percentile-based threshold computation, random number generation for synthetic data |
| **Pandas** | Loading raw CSV event logs, feature DataFrame manipulation, train/val/test manifest management, time-bucket alignment in fusion |
| **scikit-learn** | `StandardScaler` for mouse feature normalization (fit on train only), `RandomForestClassifier` as the fusion meta-model, `roc_curve()` and `auc()` for evaluation metrics |
| **Matplotlib** | Generates ROC curve plots (`roc_mouse.png`, `roc_keystroke.png`, `roc_fused.png`) saved to `results/` |
| **PyYAML** | Parses the single `configs/config.yaml` file that controls ALL hyperparameters, paths, and settings |
| **pynput** | `mouse.Listener` captures real-time mouse events (move, click) at ~10 Hz; `keyboard.Listener` captures key_down/key_up events for live data collection |
| **pickle** | Serializes the trained Random Forest fusion model (`fusion_rf.pkl`) and StandardScaler (`mouse_scaler.pkl`) |

### 4.3 Data Formats Used

| Format | Purpose |
|--------|---------|
| **CSV** | Raw event logs (mouse moves/clicks, keystrokes) |
| **NPY** (NumPy binary) | Processed windowed feature arrays |
| **PT** (PyTorch checkpoint) | Saved LSTM-AE model weights + metadata |
| **PKL** (pickle) | Saved Random Forest model + StandardScaler |
| **YAML** | Configuration file |
| **PNG** | ROC curve plots |
| **TXT** | Evaluation results table |
| **LOG** | Pipeline execution log |

---

## 5. Complete Project Architecture

```
+---------------------------------------------------------------------+
|                    BEHAVIORAL GUARD PIPELINE                         |
+---------------------------------------------------------------------+
|                                                                      |
|  +----------------------+    +----------------------+                |
|  |  MOUSE EVENT LOGGER  |    |  KEYSTROKE LOGGER    |   <- Stage 1  |
|  |  (pynput ~10Hz)      |    |  (pynput key_up/dn)  |               |
|  +----------+-----------+    +----------+-----------+                |
|             | raw CSV                    | raw CSV                   |
|             v                            v                           |
|  +----------------------+    +----------------------+                |
|  |  MOUSE FEATURE ENG.  |    |  KEYSTROKE FEAT. ENG |   <- Stage 2  |
|  |  20 kinematic feats  |    |  6 timing features   |               |
|  |  + sliding windows   |    |  + sliding windows   |               |
|  +----------+-----------+    +----------+-----------+                |
|             | .npy arrays                | .npy arrays               |
|             v                            v                           |
|  +----------------------+    +----------------------+                |
|  |  MOUSE LSTM-AE       |    |  KEYSTROKE LSTM-AE   |   <- Stage 3  |
|  |  One-class training   |    |  One-class training  |               |
|  |  MSE reconstruction   |    |  MSE reconstruction  |               |
|  |  95th %ile threshold  |    |  95th %ile threshold |               |
|  +----------+-----------+    +----------+-----------+                |
|             | recon_error                | recon_error               |
|             v                            v                           |
|  +---------------------------------------------------------+        |
|  |             DECISION-LEVEL FUSION                        | <- S4  |
|  |  Time-bucket alignment -> meta-features:                 |        |
|  |  [mouse_err, mouse_pred, ks_err, ks_pred]               |        |
|  |  -> Random Forest Meta-Classifier                        |        |
|  +----------------------------+----------------------------+        |
|                               |                                      |
|                               v                                      |
|  +---------------------------------------------------------+        |
|  |              EVALUATION ENGINE                           | <- S5  |
|  |  FAR, FRR, EER, AUC + ROC Curves                       |        |
|  |  Per-model: Mouse-only / Keystroke-only / Fused         |        |
|  +---------------------------------------------------------+        |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## 6. Project File Structure

```
Behavioral-Guard/
├── configs/
│   └── config.yaml                  # ALL hyperparameters in one place
├── data/
│   ├── raw/                         # 90 raw CSV event logs
│   │   ├── mouse_genuine_000..029.csv       (30 genuine mouse sessions)
│   │   ├── mouse_imposter_000..014.csv      (15 imposter mouse sessions)
│   │   ├── keystroke_genuine_000..029.csv   (30 genuine keystroke sessions)
│   │   └── keystroke_imposter_000..014.csv  (15 imposter keystroke sessions)
│   ├── processed/                   # 90 processed .npy + 2 manifests + 1 JSON
│   │   ├── mouse_genuine_000..029.npy       (windowed feature arrays)
│   │   ├── mouse_imposter_000..014.npy
│   │   ├── keystroke_genuine_000..029.npy
│   │   ├── keystroke_imposter_000..014.npy
│   │   ├── mouse_manifest.csv               (index of all processed sessions)
│   │   ├── keystroke_manifest.csv
│   │   └── fusion_train_imp_sessions.json   (tracks imposter train/test split)
│   ├── real/                        # For real data collection (empty, ready)
│   │   ├── raw/
│   │   └── processed/
│   └── synthetic/
│       └── synthetic_data_gen.py    # Generates fake data for smoke testing
├── models_checkpoints/
│   ├── mouse_lstm_ae.pt             # Trained mouse LSTM-AE (133 KB)
│   ├── keystroke_lstm_ae.pt         # Trained keystroke LSTM-AE (509 KB)
│   ├── mouse_scaler.pkl             # StandardScaler fitted on mouse train data
│   └── fusion_rf.pkl                # Trained Random Forest meta-classifier (128 KB)
├── results/
│   ├── evaluation_results.txt       # FAR/FRR/EER/AUC table
│   ├── roc_mouse.png                # ROC curve for mouse-only model
│   ├── roc_keystroke.png            # ROC curve for keystroke-only model
│   ├── roc_fused.png                # ROC curve for fused RF model
│   └── pipeline.log                 # Full execution log (232 KB)
├── src/
│   ├── logging/
│   │   ├── mouse_logger.py          # pynput-based background mouse capture
│   │   └── keystroke_logger.py      # pynput-based background keystroke capture
│   ├── features/
│   │   ├── mouse_features.py        # 20-feature kinematic engineering
│   │   └── keystroke_features.py    # 6-feature timing engineering
│   ├── models/
│   │   ├── lstm_ae.py               # Shared LSTM-AE architecture class
│   │   ├── train_mouse.py           # One-class training on mouse data
│   │   └── train_keystroke.py       # One-class training on keystroke data
│   ├── fusion/
│   │   └── fuse.py                  # Time-bucket alignment + RF meta-classifier
│   ├── evaluation/
│   │   ├── metrics.py               # FAR, FRR, EER, ROC computation
│   │   └── evaluate.py              # End-to-end evaluation pipeline
│   └── utils/
│       ├── config_loader.py         # YAML config loader
│       └── logger_setup.py          # Python logging setup
├── scripts/
│   └── start_real_session.py        # Convenience: starts both loggers together
├── docs/
│   └── data_collection_protocol.md  # Instructions for real data collection
├── main.py                          # Full pipeline orchestrator (5 stages)
├── requirements.txt                 # Python dependencies
├── 01_Plan_File.md                  # Complete project plan with all formulas
├── 02_Master_Build_Prompt.md        # Build instructions document
├── 03_Step_Wise_Phase_Build.md      # Step-by-step phase build guide
├── MID_SEM_PRESENTATION.md          # Mid-semester presentation content
└── README.md                        # Project README
```

---

## 7. Data Pipeline — How It Works End-to-End

The pipeline is orchestrated by `main.py` and runs 5 sequential stages:

```
python main.py                       # Run all stages
python main.py --stage generate      # Run only stage 1
python main.py --stage features      # Run only stage 2
python main.py --stage train         # Run only stage 3
python main.py --stage fuse          # Run only stage 4
python main.py --stage evaluate      # Run only stage 5
```

### Pipeline Flow:

```
Stage 1: GENERATE  -> Synthetic mouse + keystroke CSVs in data/raw/
         |
Stage 2: FEATURES  -> Windowed .npy arrays in data/processed/ + manifests
         |
Stage 3: TRAIN     -> LSTM-AE models saved to models_checkpoints/
         |
Stage 4: FUSE      -> Random Forest meta-classifier trained and saved
         |
Stage 5: EVALUATE  -> FAR/FRR/EER/AUC results + ROC plots in results/
```

### Data Split Strategy:

```
Genuine Sessions (30 total):
├── TRAIN: 60% = 18 sessions  (used to train LSTM-AEs — one-class)
├── VAL:   20% = 6 sessions   (used for threshold setting + validation loss)
└── TEST:  20% = 6 sessions   (used for final evaluation)

Imposter Sessions (15 total):
├── FUSION TRAIN: 60% = 9 sessions  (used to train Random Forest — needs both classes)
└── FUSION TEST:  40% = 6 sessions  (held out for fused model evaluation)
```

---

## 8. Stage 1: Data Collection & Synthetic Generation

### 8.1 Real-Time Data Loggers

**Mouse Logger** (`src/logging/mouse_logger.py`):
- Uses `pynput.mouse.Listener`
- Captures: `timestamp, event_type, x, y, session_id, label`
- Event types: `move`, `left_down`, `left_up`, `right_down`, `right_up`
- Move events are **rate-limited to ~10 Hz** (configurable) to avoid excessive data
- Click events are captured **immediately** (event-triggered)
- Output: append-only CSV file

**Keystroke Logger** (`src/logging/keystroke_logger.py`):
- Uses `pynput.keyboard.Listener`
- Captures: `timestamp, key, event_type, session_id, label`
- Event types: `key_down`, `key_up`
- Every key press produces a pair of events (down + up)
- Output: append-only CSV file

### 8.2 Synthetic Data Generator

**Purpose:** Smoke-testing the pipeline before real multi-day data collection. Synthetic results must **never** be reported as actual evaluation results.

**Mouse Simulation:**
- Genuine sessions: smooth cursor random walk, speed mu=5.0 sigma=2.0, dt~0.1s (~10Hz), click prob=3%
- Imposter sessions: jerkier paths, speed mu=7.0 sigma=3.5, dt~0.09s, click prob=4%
- Screen bounds: x in [0,1920], y in [0,1080], starting at (960, 540)

**Keystroke Simulation:**
- Genuine sessions: hold time mu=0.10s sigma=0.03s, flight time mu=0.15s sigma=0.04s, backspace rate=5%
- Imposter sessions: hold time mu=0.11s sigma=0.04s, flight time mu=0.18s sigma=0.06s, backspace rate=7%
- Key vocabulary: 26 lowercase letters + space/backspace/shift/enter

**Volume Generated (per config):**
- 30 genuine sessions x 2000 mouse events/session = 60,000 mouse events
- 30 genuine sessions x 1000 keystroke events/session = 30,000 keystroke events
- 15 imposter sessions x 2000 mouse events/session = 30,000 mouse events
- 15 imposter sessions x 1000 keystroke events/session = 15,000 keystroke events
- **Total: ~135,000 raw events across 90 CSV files**

---

## 9. Stage 2: Feature Engineering — All Formulas

### 9.1 Mouse Features (20 features per event)

Given a raw event sequence with timestamp `t`, position `(x, y)`:

#### Time & Spatial Deltas
```
delta_t_i = t_i - t_{i-1}                     (time delta)
delta_x_i = x_i - x_{i-1}                     (horizontal displacement)
delta_y_i = y_i - y_{i-1}                     (vertical displacement)
```

#### Distance
```
d_i = sqrt(delta_x_i^2 + delta_y_i^2)         (Euclidean distance moved)
```

#### Velocity (1st-order kinematics)
```
v_i   = d_i / delta_t_i                       (speed / magnitude of velocity)
v_x,i = delta_x_i / delta_t_i                 (horizontal velocity component)
v_y,i = delta_y_i / delta_t_i                 (vertical velocity component)
```

#### Acceleration (2nd-order kinematics)
```
acc_i   = (v_i - v_{i-1}) / delta_t_i         (acceleration magnitude)
acc_x,i = (v_x,i - v_x,{i-1}) / delta_t_i    (horizontal acceleration)
acc_y,i = (v_y,i - v_y,{i-1}) / delta_t_i    (vertical acceleration)
```

#### Jerk (3rd-order kinematics — rate of change of acceleration)
```
jerk_i   = (acc_i - acc_{i-1}) / delta_t_i    (jerk magnitude)
jerk_x,i = (acc_x,i - acc_x,{i-1}) / delta_t_i
jerk_y,i = (acc_y,i - acc_y,{i-1}) / delta_t_i
```

#### Angular Features
```
theta_i     = atan2(delta_y_i, delta_x_i)      (movement bearing in radians, range [-pi, pi])
direction_i = theta_i - theta_{i-1}            (turning angle: >0 CCW, <0 CW, 0 straight)
```

#### Curvature
```
k_i = |theta_i - theta_{i-1}| / d_i           (curvature = angular change per unit distance)
r_i = 1 / k_i                                 (radius of curvature, capped at +/-1e6)
```

**Complete feature vector (20 dimensions per event):**

| # | Feature | Symbol | Description |
|---|---------|--------|-------------|
| 1 | event_type_enc | — | Encoded: move=0, left_down=1, left_up=2, right_down=3, right_up=4 |
| 2 | x | x_i | Cursor x-coordinate |
| 3 | y | y_i | Cursor y-coordinate |
| 4 | delta_t | delta_t_i | Time since previous event |
| 5 | delta_x | delta_x_i | Horizontal displacement |
| 6 | delta_y | delta_y_i | Vertical displacement |
| 7 | d | d_i | Euclidean distance moved |
| 8 | v | v_i | Speed (scalar) |
| 9 | v_x | v_x,i | Horizontal velocity |
| 10 | v_y | v_y,i | Vertical velocity |
| 11 | acc | acc_i | Acceleration magnitude |
| 12 | acc_x | acc_x,i | Horizontal acceleration |
| 13 | acc_y | acc_y,i | Vertical acceleration |
| 14 | jerk | jerk_i | Jerk magnitude |
| 15 | jerk_x | jerk_x,i | Horizontal jerk |
| 16 | jerk_y | jerk_y,i | Vertical jerk |
| 17 | theta | theta_i | Movement bearing (radians) |
| 18 | direction | — | Direction change (angular difference) |
| 19 | k | k_i | Curvature |
| 20 | r | r_i | Radius of curvature |

**Why these features matter:** Velocity, acceleration, and jerk profiles capture the *physical dynamics* of how someone moves a mouse — smoothness, overshoot, correction habits — which differ meaningfully between individuals and are extremely difficult to consciously imitate over a full session.

---

### 9.2 Keystroke Features (6 features per keystroke)

Given key-press (`key_down`) and key-release (`key_up`) events:

#### Hold Time (Dwell Time)
```
hold_time_i = t_up(key_i) - t_down(key_i)
```
How long a key is held down. Encodes finger dexterity and force.

#### Flight Time (Inter-Key Interval)
```
flight_time_i = t_down(key_i) - t_down(key_{i-1})
```
Time between consecutive key presses. Encodes typing rhythm.

#### Digraph Latency
```
digraph_latency_i = t_down(key_i) - t_down(key_{i-1})
```
Same as flight time for consecutive pairs — included explicitly for the model.

#### Trigraph Latency
```
trigraph_latency_i = t_down(key_i) - t_down(key_{i-2})
```
Time spanning 3 consecutive key presses. Captures longer-range timing patterns.

#### Typing Speed (Rolling Window)
```
typing_speed_i = (N - 1) / (t_down(key_i) - t_down(key_{i-N+1}))
```
Keys per second over a rolling window of N=5 presses. Captures instantaneous throughput.

#### Backspace Frequency
```
backspace_freq_i = mean(is_backspace) over rolling window of 10
```
Correction rate — how often the user makes and corrects mistakes.

**Complete feature vector (6 dimensions per keystroke):**

| # | Feature | Formula | Description |
|---|---------|---------|-------------|
| 1 | hold_time | t_up - t_down | Key dwell time (seconds) |
| 2 | flight_time | t_down(i) - t_down(i-1) | Inter-key interval |
| 3 | digraph_latency | Same as flight_time | Two-key latency |
| 4 | trigraph_latency | t_down(i) - t_down(i-2) | Three-key latency |
| 5 | typing_speed | (N-1) / delta_t | Local keys/second (N=5) |
| 6 | backspace_freq | Rolling mean (window=10) | Error correction rate |

**Why these features matter:** Dwell and flight times encode finger dexterity and muscle memory that is extremely stable for a given person over short-to-medium timeframes, and is one of the most well-studied behavioral biometrics in the literature.

---

### 9.3 Sliding Window Segmentation

Raw features are segmented into fixed-length overlapping windows:

```
window[j] = features[ j * stride : j * stride + window_size ]
```

| Parameter | Mouse | Keystroke |
|-----------|-------|-----------|
| `window_size` | 50 events | 30 keystrokes |
| `window_stride` | 25 events | 15 keystrokes |
| Overlap | 50% | 50% |
| Output shape per window | (50, 20) | (30, 6) |

This converts irregular event streams into fixed-shape tensors suitable for sequence models.

---

## 10. Stage 3: LSTM-Autoencoder Model — Architecture & Formulas

### 10.1 LSTM Cell — Core Recurrent Unit

For input `B_t` at time step `t`, previous hidden state `h_{t-1}`, previous cell state `C_{t-1}`:

```
f_t = sigma(W_f * [h_{t-1}, B_t] + b_f)          <- Forget Gate
i_t = sigma(W_i * [h_{t-1}, B_t] + b_i)          <- Input Gate
C~_t = tanh(W_c * [h_{t-1}, B_t] + b_c)          <- Candidate Cell State
C_t = f_t . C_{t-1} + i_t . C~_t                 <- New Cell State
o_t = sigma(W_o * [h_{t-1}, B_t] + b_o)          <- Output Gate
h_t = o_t . tanh(C_t)                             <- New Hidden State
```

Where:
- `sigma` = Sigmoid activation function
- `tanh` = Hyperbolic tangent activation function
- `.` = Element-wise (Hadamard) product
- `W`, `b` = Learnable weight matrices and bias vectors

**Why LSTM?** Behavioral sequences have **long-term temporal dependencies** (a typing rhythm or mouse-movement style persists across many timesteps) that plain feed-forward networks can't capture.

---

### 10.2 LSTM-Autoencoder Architecture

```
ENCODER:
  Input: x of shape (batch, seq_length, input_dim)
    |
    v
  [2 Stacked LSTM Layers] -> output: (batch, seq, hidden_dim)
    |                        h_n: (num_layers, batch, hidden_dim)
    v
  Take h_n[-1] (last layer's final hidden state)
    |
    v
  [FC Layer: hidden_dim -> embedding_dim]
    |
    v
  z of shape (batch, embedding_dim)     <- BOTTLENECK

DECODER:
  z of shape (batch, embedding_dim)
    |
    v
  [FC Layer: embedding_dim -> hidden_dim]
    |
    v
  Repeat across seq_length timesteps -> (batch, seq_length, hidden_dim)
    |
    v
  [2 Stacked LSTM Layers] -> (batch, seq_length, hidden_dim)
    |
    v
  [FC Layer: hidden_dim -> input_dim]
    |
    v
  x_hat of shape (batch, seq_length, input_dim)  <- RECONSTRUCTION
```

### 10.3 Model Hyperparameters

| Parameter | Mouse Model | Keystroke Model |
|-----------|------------|-----------------|
| `input_dim` | 20 | 6 |
| `seq_length` | 50 | 30 |
| `embedding_dim` (bottleneck) | 16 | 8 |
| `hidden_dim` (LSTM units) | 64 | 32 |
| `num_lstm_layers` | 2 | 2 |
| `dropout` | 0.1 | 0.1 |
| `learning_rate` | 0.001 | 0.001 |
| `batch_size` | 128 | 128 |
| `epochs` | 50 | 50 |
| `optimizer` | Adam | Adam |
| `threshold_percentile` | 95th | 95th |

### 10.4 Training Process (One-Class)

**Key principle:** Both models are trained on **genuine data ONLY** (one-class learning). They never see imposter data during training.

```
Loss Function:  MSE = (1/N) * Sum (B_t - B_hat_t)^2

Training:       Minimize reconstruction error on GENUINE sequences
                using Adam optimizer with LR = 0.001

Threshold:      tau = 95th percentile of MSE over genuine VALIDATION set
                (only 5% of genuine sequences should exceed this threshold)

Decision Rule:  If MSE(new_sequence) > tau  ->  IMPOSTER
                Else                         ->  GENUINE
```

**Mouse-specific: StandardScaler normalization**
- A `StandardScaler` is fitted on the training data only (preventing data leakage)
- Applied to validation and test data using the same fitted scaler
- Scaler saved to `models_checkpoints/mouse_scaler.pkl`

### 10.5 Reconstruction Error Formula

```
MSE_per_sample = mean over (seq_length * input_dim) of (x - x_hat)^2

Per-sample MSE:   e of shape (batch,)     — mean over sequence & features
Full-batch MSE:   e scalar                 — scalar mean over everything
```

---

## 11. Stage 4: Decision-Level Fusion

### 11.1 Why Decision-Level (Not Feature-Level) Fusion

Each modality is evaluated independently first, so the system remains functional even if one modality's data is temporarily unavailable (e.g., user is only using the mouse, not typing). Feature-level fusion doesn't offer this robustness.

### 11.2 Time-Bucket Alignment

Mouse (~10 Hz continuous) and keystroke (bursty, only while typing) have different, uncorrelated sampling patterns. They are aligned using **fixed-time-window bucketing**:

```
bucket_index = floor((t_centre - t_0) / bucket_seconds)

Where:
  t_0 = first timestamp of the session
  t_centre = estimated centre time of each window
  bucket_seconds = 10 (configurable)
```

Windows from both modalities that fall into the same time bucket are grouped together.

### 11.3 Meta-Feature Vector

For each time bucket, the following 4-dimensional meta-feature vector is constructed:

```
X_meta = [mouse_error, mouse_pred, keystroke_error, keystroke_pred]

Where:
  mouse_error     = mean MSE reconstruction error from Mouse LSTM-AE (float)
  mouse_pred      = 1 if mouse_error > mouse_threshold, else 0 (binary)
  keystroke_error = mean MSE reconstruction error from Keystroke LSTM-AE (float)
  keystroke_pred  = 1 if keystroke_error > keystroke_threshold, else 0 (binary)
```

### 11.4 Random Forest Meta-Classifier

```
Configuration:
  n_estimators = 100 (number of decision trees)
  max_depth    = 10
  random_state = 42 (reproducibility)

Training Data:
  X = meta-feature vectors from train+val genuine + train imposter sessions
  y = 0 (genuine) or 1 (imposter)

Output:
  Probability score for imposter class -> used for ROC/EER
  Binary prediction -> used for FAR/FRR
```

---

## 12. Stage 5: Evaluation — Metrics & Formulas

### 12.1 Convention

```
Genuine  = 0 (positive class = legitimate user)
Imposter = 1 (negative class = unauthorized user)

TP = Imposter correctly flagged (True Positive)
TN = Genuine correctly accepted (True Negative)
FP = Genuine wrongly rejected (False Positive)
FN = Imposter wrongly accepted (False Negative)
```

### 12.2 False Acceptance Rate (FAR)

```
FAR = Imposters accepted as genuine / Total imposters
    = FN / (FN + TP)
    = (y_pred[imposter_mask] == 0).sum() / total_imposters
```
**Interpretation:** Security metric — what fraction of imposters get through? **Lower is better.**

### 12.3 False Rejection Rate (FRR)

```
FRR = Genuine users rejected / Total genuine users
    = FP / (FP + TN)
    = (y_pred[genuine_mask] == 1).sum() / total_genuine
```
**Interpretation:** Usability metric — how often is the real user locked out? **Lower is better.**

### 12.4 Equal Error Rate (EER)

```
EER = The point on the ROC curve where FAR ≈ FRR

Computed by:
  1. Generate ROC curve: fpr, tpr, thresholds = roc_curve(y_true, scores)
  2. fnr = 1 - tpr
  3. Find index where |fpr - fnr| is minimized
  4. EER = (fpr[idx] + fnr[idx]) / 2
```
**Interpretation:** Single-number summary of security/usability tradeoff. **Lower is better.** An EER of 0% means perfect separation.

### 12.5 ROC Curve & AUC

```
ROC = plot of TPR (True Positive Rate) vs FPR (False Positive Rate)
      across all possible threshold values

AUC = Area Under the ROC Curve (computed via trapezoidal integration)
      Range: [0, 1], where 1.0 = perfect separability
```

### 12.6 Evaluation Protocol

Three separate evaluations are run:

| Model | Input | Scoring |
|-------|-------|---------|
| **Mouse-only** | Test genuine + all imposter mouse windows | MSE reconstruction error vs. mouse threshold |
| **Keystroke-only** | Test genuine + all imposter keystroke windows | MSE reconstruction error vs. keystroke threshold |
| **Fused (RF)** | Test meta-features (test genuine + test imposters only) | RF probability score |

---

## 13. Current Evaluation Results (Phase 1)

### Results Table (Synthetic Data)

```
--------------------------------------------------------
Model                  FAR      FRR      EER      AUC
--------------------------------------------------------
Mouse-only           0.0008   0.0556   0.0246   0.9940
Keystroke-only       0.4021   0.0781   0.1323   0.9071
Fused (RF)           0.0111   0.0000   0.0120   0.9999
--------------------------------------------------------
```

### Interpretation of Results

| Model | Analysis |
|-------|----------|
| **Mouse-only** | Excellent performance. FAR=0.08% (almost no imposters get through), FRR=5.56% (occasionally locks out genuine user), AUC=0.9940 (near-perfect). The 20 kinematic features are highly discriminative. |
| **Keystroke-only** | Weaker standalone performance. FAR=40.21% (many imposters accepted), FRR=7.81%, EER=13.23%. This is expected because the synthetic genuine/imposter keystroke distributions overlap significantly (mu difference is only 0.01s for hold time). |
| **Fused (RF)** | **Best overall.** FAR=1.11%, FRR=0.00% (zero genuine rejections!), EER=1.20%, AUC=0.9999 (near-perfect). Fusion dramatically compensates for keystroke's weakness by combining both modality signals. |

> **Important Note:** These results are on **synthetic data** (designed for pipeline smoke-testing). Synthetic data has artificially tunable separability. Real-world results with actual human data are expected to be different and should be the basis for any publication.

---

## 14. Configuration & Hyperparameters

All settings are centralized in `configs/config.yaml` — **no hardcoded magic numbers anywhere in the code.**

| Section | Parameter | Value | Purpose |
|---------|-----------|-------|---------|
| **Paths** | raw_data_dir | `data/raw` | Raw CSV event logs |
| | processed_data_dir | `data/processed` | Processed .npy features |
| | model_checkpoint_dir | `models_checkpoints` | Saved models |
| | results_dir | `results` | Evaluation outputs |
| **Collection** | mouse.sampling_rate_hz | 10 | Mouse move rate limit |
| **Features** | mouse.window_size | 50 | Events per mouse window |
| | mouse.window_stride | 25 | Overlap stride |
| | keystroke.window_size | 30 | Keystrokes per window |
| | keystroke.window_stride | 15 | Overlap stride |
| **Mouse Model** | input_dim | 20 | Feature dimensions |
| | seq_length | 50 | Must match window_size |
| | embedding_dim | 16 | Bottleneck dimension |
| | hidden_dim | 64 | LSTM hidden units |
| | num_lstm_layers | 2 | LSTM stack depth |
| | dropout | 0.1 | Regularization |
| | learning_rate | 0.001 | Adam LR |
| | batch_size | 128 | Training batch size |
| | epochs | 50 | Training epochs |
| | threshold_percentile | 95 | Anomaly threshold |
| **Keystroke Model** | input_dim | 6 | Feature dimensions |
| | seq_length | 30 | Must match window_size |
| | embedding_dim | 8 | Bottleneck dimension |
| | hidden_dim | 32 | LSTM hidden units |
| | (rest same as mouse) | | |
| **Fusion** | time_bucket_seconds | 10 | Alignment bucket size |
| | imposter_train_ratio | 0.6 | Fraction for fusion training |
| | n_estimators | 100 | RF trees |
| | max_depth | 10 | RF tree depth |
| **Evaluation** | train_ratio | 0.60 | Training split |
| | val_ratio | 0.20 | Validation split |
| | test_ratio | 0.20 | Test split |
| **Synthetic** | num_genuine_sessions | 30 | Synthetic genuine sessions |
| | num_imposter_sessions | 15 | Synthetic imposter sessions |
| | events_per_mouse_session | 2000 | Events per session |
| | events_per_keystroke_session | 1000 | Events per session |
| | random_seed | 42 | Reproducibility |

---

## 15. Current Project Status

### Phase 1 — COMPLETE (Mid-Semester)

| Component | Status | File |
|-----------|--------|------|
| Mouse event logger (background capture, ~10 Hz) | DONE | `src/logging/mouse_logger.py` |
| Keystroke event logger (background capture) | DONE | `src/logging/keystroke_logger.py` |
| Mouse feature engineering (20 kinematic features) | DONE | `src/features/mouse_features.py` |
| Keystroke feature engineering (6 timing features) | DONE | `src/features/keystroke_features.py` |
| Sliding window segmentation (both modalities) | DONE | Integrated in feature modules |
| Shared LSTM-Autoencoder architecture | DONE | `src/models/lstm_ae.py` |
| One-class training — mouse model | DONE | `src/models/train_mouse.py` |
| One-class training — keystroke model | DONE | `src/models/train_keystroke.py` |
| StandardScaler normalization (mouse, no leakage) | DONE | `src/models/train_mouse.py` |
| Per-modality threshold (95th percentile) | DONE | Both training scripts |
| Fixed-time-bucket decision-level fusion | DONE | `src/fusion/fuse.py` |
| Random Forest meta-classifier | DONE | `src/fusion/fuse.py` |
| Evaluation engine (FAR/FRR/EER/AUC/ROC) | DONE | `src/evaluation/evaluate.py` |
| Synthetic data generator | DONE | `data/synthetic/synthetic_data_gen.py` |
| Centralized YAML config | DONE | `configs/config.yaml` |
| Python logging throughout | DONE | All modules |
| Checkpoint save/load | DONE | `src/models/lstm_ae.py` |
| Pipeline orchestrator | DONE | `main.py` |
| Real data collection infrastructure | DONE | `scripts/start_real_session.py` |
| Data leakage fix (fusion imposter split) | DONE | `main.py` + `evaluate.py` |
| Trained models saved | DONE | `models_checkpoints/` |
| Evaluation results generated | DONE | `results/` |

### Phase 2 — NOT YET STARTED (End-Semester)

| Component | Status |
|-----------|--------|
| Real human data collection (10-14 days) | PENDING |
| CTGAN / TimeGAN modality alignment | PENDING |
| Full 8-model stacking comparison | PENDING |
| Adaptive Learning Module | PENDING |
| Imitation-attack evaluation | PENDING |
| Drift-resistance testing | PENDING |
| IEEE/Scopus paper draft | PENDING |
| Patent claim draft | PENDING |

---

## 16. Phase 2 — Planned Work (End-Sem)

### 16.1 Modality Alignment via CTGAN / TimeGAN

**Problem:** Current fixed-time-bucket approach simply averages windows that fall in the same 10-second bucket — lossy and imprecise.

**Planned Solution:** Use generative adversarial networks to learn each modality's temporal distribution and produce aligned synthetic samples:

```
Generator G:      Produces synthetic aligned behavioral samples
Discriminator D:  Distinguishes real vs. generated samples

min_G max_D  E[log D(x_real)] + E[log(1 - D(G(z)))]
```

### 16.2 Full Meta-Model Comparison (8 Algorithms)

| # | Algorithm | Type |
|---|-----------|------|
| 1 | Naive Bayes | Probabilistic |
| 2 | Logistic Regression | Linear |
| 3 | K-Nearest Neighbors | Instance-based |
| 4 | Decision Tree | Tree |
| 5 | Random Forest | Ensemble (bagging) |
| 6 | SVM | Kernel |
| 7 | Gradient Boosting (XGBoost) | Ensemble (boosting) |
| 8 | LightGBM | Ensemble (boosting) |

### 16.3 Adaptive Learning Module (Novel Contribution — Patent Claim)

```
Confidence-Gated Update Rule:

Accept sample for adaptation ONLY IF:
  reconstruction_error(sample) < tau_accept    (strict, tighter than auth threshold)
  AND sample passed authentication with high margin

Exponential Moving Update:
  theta_t = alpha * theta_{t-1} + (1 - alpha) * theta_new

  Where alpha in [0.95, 0.99] controls drift speed (anti-poisoning safeguard)

Dual-Confirmation Safeguard:
  Require N consecutive high-confidence sessions before committing update
  Log + rollback capability if later sessions contradict recent adaptation
```

### 16.4 Imitation-Attack Evaluation

Protocol:
1. Show a third party recordings/descriptions of the owner's typing/mouse style
2. Ask them to deliberately mimic that style for a test session
3. Compare FAR under passive imposters vs. active (deliberate mimicry) imposters

---

## 17. Novel Contributions & Patent Claim

| Contribution | Description | Status |
|-------------|-------------|--------|
| **Adaptive Learning Module** (Primary, patentable) | Confidence-gated incremental profile update with bounded drift rate and dual-confirmation safeguard to prevent profile poisoning | Phase 2 |
| **Imitation-Attack Robustness Testing** (Secondary, publishable) | Evaluating FAR under deliberate mimicry, which the reference paper never tested | Phase 2 |
| **Detection Delay Metric** (New metric) | `t(imposter flagged) - t(imposter took control)` — practical metric not in reference paper | Phase 2 |

**What is NOT novel (adapted from prior art):**
- LSTM-Autoencoder architecture (from MBBFAuth / Li et al.)
- Feature engineering pipeline (from MBBFAuth)
- Decision-level fusion via stacking (from MBBFAuth)

---

## 18. Dataset Summary

### Synthetic Dataset (Current — for smoke testing)

| Data Type | Genuine Sessions | Imposter Sessions | Events/Session | Total Events |
|-----------|-----------------|-------------------|----------------|--------------|
| Mouse | 30 | 15 | ~2000 | ~90,000 |
| Keystroke | 30 | 15 | ~1000 | ~45,000 |
| **Total** | **60** | **30** | — | **~135,000** |

### Processed Feature Arrays

| Modality | Shape per Session | Approx. Windows/Session |
|----------|------------------|------------------------|
| Mouse | (N, 50, 20) | ~78 windows |
| Keystroke | (N, 30, 6) | ~32 windows |

### Real Dataset (Planned)

- **Genuine sessions:** Multiple sessions/day over 10-14 days
- **Imposter sessions:** 2-3 different people, 1 session each (~15-30 min)
- **Target volume:** ~2000 mouse events + ~1000 keystroke events per session

---

## 19. Saved Model Checkpoints

| File | Size | Description |
|------|------|-------------|
| `mouse_lstm_ae.pt` | 133 KB | Mouse LSTM-AE weights + threshold + training losses |
| `keystroke_lstm_ae.pt` | 509 KB | Keystroke LSTM-AE weights + threshold + training losses |
| `mouse_scaler.pkl` | <1 KB | StandardScaler fitted on mouse training data |
| `fusion_rf.pkl` | 128 KB | Random Forest classifier (100 trees, max_depth=10) |

---

## 20. How to Run the Project

### Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py                 # All 5 stages sequentially
```

### Run Individual Stages
```bash
python main.py --stage generate    # Generate synthetic data
python main.py --stage features    # Extract features
python main.py --stage train       # Train LSTM-AEs
python main.py --stage fuse        # Train fusion Random Forest
python main.py --stage evaluate    # Evaluate + ROC plots
```

### Collect Real Data
```bash
# Start both loggers simultaneously
python scripts/start_real_session.py --session_id s01 --label genuine

# Or start individually
python -m src.logging.mouse_logger --session_id s01 --label genuine
python -m src.logging.keystroke_logger --session_id s01 --label genuine

# Press Ctrl+C to stop
```

### Custom Config
```bash
python main.py --config path/to/custom_config.yaml
```

---

## Summary

**Behavioral Guard** is a fully functional Phase 1 prototype of a continuous authentication system using behavioral biometrics. It captures mouse and keystroke patterns, engineers 26 features (20 mouse + 6 keystroke), trains one-class LSTM-Autoencoder anomaly detectors per modality, fuses their outputs via a Random Forest meta-classifier, and evaluates with standard biometric security metrics (FAR, FRR, EER, AUC). The pipeline achieves near-perfect AUC (0.9999) on synthetic test data with the fused model. The primary novel contribution planned for Phase 2 is a confidence-gated adaptive learning module with anti-poisoning safeguards — the core patent claim.
