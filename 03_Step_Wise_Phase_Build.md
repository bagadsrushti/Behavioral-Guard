# Behavioral Guard — Step-Wise Phase Build (Every Step + Individual Prompt)

How to use this file: every phase from the Master Build Prompt is broken down here into small, individually buildable steps. Each step has its own copy-paste-ready prompt for Antigravity. Build steps **in order** within a phase — later steps assume earlier ones exist. Phase 1 steps are marked `[COMPLETED]` — they document what was already built and are included for rebuild/onboarding/patent-trail purposes, not to be re-run. Start active work at Phase 0.

---

# PHASE 0 — Remediation (3 steps)

## Step 0.1 — Fix mouse feature normalization

**Goal:** stop the mouse LSTM-AE loss from being enormous/flat by scaling features properly.

```
In src/features/mouse_features.py and src/models/train_mouse.py, the raw mouse kinematic features (x, y, delta_t, delta_x, delta_y, d, v, v_x, v_y, acc, acc_x, acc_y, jerk, jerk_x, jerk_y, theta, direction, k, r) are fed into the LSTM-Autoencoder with no scaling, causing training loss in the tens of millions that does not converge.

Fix this:
1. Add a fit_scaler(train_windows) function (use sklearn StandardScaler, reshaped appropriately for 3D window data: fit per-feature across all timesteps and all training windows).
2. Fit the scaler ONLY on the training split of mouse windows in train_mouse.py, before model training begins.
3. Transform train, validation, and test windows using this fitted scaler before they are passed to the model anywhere (training, threshold-setting, evaluation).
4. Save the fitted scaler object (pickle) alongside the model checkpoint in models_checkpoints/ (e.g. mouse_scaler.pkl), and load it wherever the mouse model checkpoint is loaded (train_mouse.py, fuse.py, evaluate.py) so inference always uses the same scaling.
5. Do NOT fit a new scaler on validation or test data at any point.

Re-run mouse training only (python main.py --stage train) and report the new loss curve. Confirm loss is now in a small, reasonable range and decreases meaningfully across epochs rather than staying flat.
```

## Step 0.2 — Fix fusion data leakage

**Goal:** stop the Random Forest fusion model from being trained and tested on the same imposter sessions.

```
In main.py's stage_fuse() function and src/evaluation/evaluate.py's fused-model evaluation block, ALL imposter sessions are currently used both to train the fusion Random Forest and to test it afterward — the same sessions appear on both sides, inflating the fused model's apparent FAR/FRR/EER.

Fix this:
1. Add a new config.yaml field under `fusion`: `imposter_train_ratio` (e.g. 0.6).
2. In stage_fuse() (main.py), split imposter sessions (by session_id, not by window, to avoid within-session leakage) into a fusion-training subset (imposter_train_ratio) and a fusion-holdout subset. Only the fusion-training subset of imposter sessions may be used when calling build_meta_features() for training the Random Forest.
3. In evaluate.py's fused-model evaluation block, use ONLY the fusion-holdout imposter subset (the sessions NOT used in step 2) plus the existing held-out genuine test split, when calling build_meta_features() for evaluation.
4. Make sure the exact same split logic/session IDs are used consistently between stage_fuse() and evaluate.py (e.g. by saving the split assignment to a small JSON/CSV file after stage_fuse() runs, and having evaluate.py read it back, rather than recomputing the split independently in two places where it could drift out of sync).

Re-run the full pipeline (python main.py) and report the new Fused (RF) FAR/FRR/EER/AUC. Flag clearly if the fused numbers are still suspiciously perfect (0.0000 / 1.0000) — that would now point to the synthetic data being too easily separable rather than a leakage bug.
```

## Step 0.3 — Scaffold real data collection

**Goal:** get real genuine/imposter data collection started in parallel with further development.

```
Set up the groundwork for real (non-synthetic) data collection, without modifying the core pipeline logic:

1. Create data/real/ with raw/ and processed/ subfolders, matching the existing data/raw and data/processed structure/conventions used for synthetic data.
2. Create docs/data_collection_protocol.md documenting:
   - How to invoke the existing loggers for real sessions: `python -m src.logging.mouse_logger --session_id <id> --label genuine --output data/real/raw/mouse_<id>.csv` and the keystroke equivalent.
   - Recommended collection schedule: multiple genuine sessions per day, spread across at least 10-14 days, aiming for a similar total-event volume per session as the synthetic sessions (~2000 mouse events, ~1000 keystroke events) so results are comparable.
   - How to label and collect a small number of imposter sessions (2-3 different people, one session each, ~15-30 min, clearly labeled --label imposter), plus a note that consent must be obtained from anyone whose behavior is logged.
   - A reminder that synthetic data is for pipeline smoke-testing only and must never be reported as an evaluation result.
3. Add a small CLI helper script (scripts/start_real_session.py or similar) that starts BOTH the mouse and keystroke loggers together for a given session_id/label with one command, since real sessions need both modalities logged simultaneously.

Do not touch feature engineering, training, or evaluation code in this step — this step only sets up the data collection scaffolding and documentation.
```

---

# PHASE 1 — Mid-Sem Build [COMPLETED — reference only, do not re-run]

## Step 1.1 — [COMPLETED] Project scaffolding
```
[REFERENCE] Set up the Behavioral Guard project structure: src/{logging,features,models,fusion,evaluation,utils} packages with __init__.py files, configs/config.yaml, data/{raw,processed,synthetic} directories, models_checkpoints/, results/, main.py entry point, requirements.txt pinning torch/pandas/numpy/scikit-learn/pyyaml/matplotlib/pynput, and a README.md documenting setup and usage.
```

## Step 1.2 — [COMPLETED] Mouse event logger
```
[REFERENCE] Build src/logging/mouse_logger.py: a pynput-based MouseLogger class capturing move events rate-limited to ~10 Hz and click events (left/right down/up) immediately, writing timestamp/event_type/x/y/session_id/label rows to an append-only CSV, with a CLI (--session_id, --label genuine|imposter, --output, --config) and Ctrl+C-safe start/stop.
```

## Step 1.3 — [COMPLETED] Keystroke event logger
```
[REFERENCE] Build src/logging/keystroke_logger.py: a pynput-based KeystrokeLogger class capturing key_down/key_up events with normalized key names, writing timestamp/key/event_type/session_id/label rows to an append-only CSV, with the same CLI pattern as the mouse logger.
```

## Step 1.4 — [COMPLETED] Mouse feature engineering
```
[REFERENCE] Build src/features/mouse_features.py: compute per-event kinematic features (event_type_enc, x, y, delta_t, delta_x, delta_y, d, v, v_x, v_y, acc, acc_x, acc_y, jerk, jerk_x, jerk_y, theta, direction, k, r — 20 total) from a raw mouse CSV, segment into fixed-length sliding windows (configurable window_size/stride), handle first-event edge cases and inf/NaN cleanup, and provide process_all_mouse_sessions() to batch-process a raw directory into .npy windows + a manifest CSV.
```

## Step 1.5 — [COMPLETED] Keystroke feature engineering
```
[REFERENCE] Build src/features/keystroke_features.py: match key_down/key_up pairs to compute hold_time, flight_time, digraph_latency, trigraph_latency, rolling typing_speed, and rolling backspace_freq (6 features total) from a raw keystroke CSV, segment into sliding windows, and provide process_all_keystroke_sessions() mirroring the mouse batch-processing function.
```

## Step 1.6 — [COMPLETED] Shared LSTM-Autoencoder architecture
```
[REFERENCE] Build src/models/lstm_ae.py: an LSTMAutoencoder nn.Module with a 2-layer LSTM encoder + FC bottleneck (encode()), and an FC + 2-layer LSTM + FC decoder that repeats the bottleneck vector across seq_length timesteps (decode()), a forward() returning (reconstruction, embedding), a reconstruction_mse() helper supporting per-sample/mean/raw reduction, and save_checkpoint()/load_checkpoint() functions persisting weights plus architecture config and optional extra metadata (e.g. threshold).
```

## Step 1.7 — [COMPLETED] Mouse LSTM-AE training script
```
[REFERENCE] Build src/models/train_mouse.py: load the mouse manifest, take only genuine sessions, time-ordered split into train/validation per config ratios, build a DataLoader from concatenated training windows, train an LSTMAutoencoder with Adam + MSE loss for the configured number of epochs (logging every 10 epochs), compute per-sample reconstruction error on the validation set, set the anomaly threshold at the configured percentile, and save the checkpoint with the threshold and loss history attached.
```

## Step 1.8 — [COMPLETED] Keystroke LSTM-AE training script
```
[REFERENCE] Build src/models/train_keystroke.py: identical structure to train_mouse.py, but using the keystroke manifest, keystroke config section (model_keystroke), and keystroke checkpoint filename.
```

## Step 1.9 — [COMPLETED] Basic fusion (time-bucket + Random Forest)
```
[REFERENCE] Build src/fusion/fuse.py: assign each modality's windows to fixed-length time buckets estimated from each session's raw CSV start/end timestamps, compute reconstruction error per window for both modalities using their trained LSTM-AEs, aggregate mean error per (session, bucket), join mouse and keystroke buckets on shared keys to build a 4-column meta-feature matrix ([mouse_error, mouse_pred, keystroke_error, keystroke_pred]) with ground-truth labels, and train/save/load a RandomForestClassifier on these meta-features.
```

## Step 1.10 — [COMPLETED] Evaluation module
```
[REFERENCE] Build src/evaluation/metrics.py (compute_far_frr, compute_eer via ROC-based FPR/FNR crossing, compute_roc, format_results_table) and src/evaluation/evaluate.py (load test-split genuine + imposter windows per modality, evaluate mouse-only and keystroke-only models against their thresholds, rebuild fused meta-features on the test split and evaluate the saved Random Forest, save ROC plots per model, print and save a consolidated results table).
```

## Step 1.11 — [COMPLETED] Synthetic data generator
```
[REFERENCE] Build data/synthetic/synthetic_data_gen.py: generate configurable numbers of genuine/imposter mouse sessions (random-walk cursor paths with click events, distinct speed/click-probability parameters per label) and keystroke sessions (key_down/key_up pairs with distinct hold/flight-time/backspace-rate parameters per label), writing them as raw CSVs matching the logger's schema, for end-to-end pipeline smoke testing.
```

## Step 1.12 — [COMPLETED] Orchestration, config, logging, README
```
[REFERENCE] Build main.py (argparse-driven stage runner: generate/features/train/fuse/evaluate/all), src/utils/config_loader.py (YAML loading with a repo-relative default path and a get_nested() helper), src/utils/logger_setup.py (centralized Python logging configuration from config), configs/config.yaml (all hyperparameters/paths/window sizes/thresholds, no hardcoded values in code), and README.md documenting project structure, setup, how to run each stage, configuration reference, evaluation output description, and an explicit "Implemented in this phase" vs "Planned for Phase 2" checklist.
```

---

# PHASE 2 — End-Sem Build (18 steps — the remaining real work)

## Step 2.1 — Modality alignment: design note
**Goal:** think through the alignment approach before writing code.
```
Before writing any alignment code, produce a short design note (docs/alignment_design.md) covering: (a) why fixed time-bucket aggregation (current Phase 1 approach) loses information — specifically, within-bucket averaging discards intra-bucket temporal structure and can wash out short anomalous bursts; (b) a concrete plan for a CTGAN/TimeGAN-based alternative — what exactly will be modeled (per-modality per-window feature distributions), what the generator/discriminator will be trained to do, and how the trained generator will be used at alignment time (e.g. to interpolate one modality's feature trajectory onto the other modality's timestamps, or to generate synthetic aligned samples for missing timestamps); (c) what will be logged/compared to show this is actually an improvement over Phase 1's fixed-bucket approach (e.g. reconstruction fidelity of held-out real windows, or downstream fusion AUC with old vs new alignment). Do not write model code yet — just the design note.
```

## Step 2.2 — Modality alignment: CTGAN/TimeGAN implementation
```
Based on docs/alignment_design.md, implement src/alignment/modality_align.py:
1. A generator/discriminator pair (start with a simpler TimeGAN-style recurrent generator/discriminator if a full CTGAN adaptation is too heavy — either is acceptable as long as it's time-series-aware, not purely tabular CTGAN treating each row independently).
2. A fit(mouse_windows, keystroke_windows) method that trains the alignment model on training-split data only.
3. An align(mouse_windows, keystroke_windows) method that returns both modalities' features resampled/interpolated onto a shared set of time steps, replacing the current fixed-time-bucket join in src/fusion/fuse.py.
4. Save/load functions for the trained alignment model, consistent with the existing checkpoint pattern in src/models/lstm_ae.py.
Keep this as a new, separate module — do not delete the existing _assign_time_buckets()/_bucket_agg() functions in fuse.py yet.
```

## Step 2.3 — Modality alignment: wire in as a config-controlled fallback
```
Add a config.yaml field `fusion.alignment_method: "bucket" | "generative"`. Update src/fusion/fuse.py's build_meta_features() to branch on this setting: "bucket" uses the existing Phase 1 fixed-time-bucket logic unchanged; "generative" uses the new src/alignment/modality_align.py module from Step 2.2. Update main.py's stage_fuse() to train the alignment model (if alignment_method == "generative") before building meta-features. Run the full pipeline once with each setting and save both results tables (results/evaluation_bucket.txt and results/evaluation_generative.txt) so the two alignment approaches can be directly compared in the paper.
```

## Step 2.4 — Fusion: extend to 8 candidate meta-models
```
Extend src/fusion/fuse.py's train_fusion_model() (or add a new train_fusion_models_all() function) to train all 8 candidate meta-model algorithms on the same meta-feature matrix: Naive Bayes (sklearn GaussianNB), Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest (existing), Support Vector Classifier, Gradient Boosting Machine (sklearn GradientBoostingClassifier), and LightGBM (lightgbm package — add to requirements.txt). Use sensible default hyperparameters for each; a full grid search per model is optional if time allows but not required for this step.
```

## Step 2.5 — Fusion: comparison table + model selection
```
Add a new evaluation path (e.g. src/fusion/compare_meta_models.py) that evaluates all 8 trained meta-models from Step 2.4 on the same held-out fusion test set (respecting the Step 0.2 leakage fix), computing FAR/FRR/EER/AUC for each, and saves a comparison table to results/meta_model_comparison.txt plus an overlaid ROC plot (results/roc_meta_model_comparison.png) with all 8 curves. Select the best-performing model by lowest EER as the new default fusion model going forward, but keep all 8 results available for the paper's comparison table.
```

## Step 2.6 — Adaptive module: confidence-gating logic
```
Create src/adaptive/adaptive_update.py. First implement just the confidence-gating check: a function is_high_confidence_genuine(reconstruction_error, auth_threshold, accept_margin) that returns True only if reconstruction_error < auth_threshold * (1 - accept_margin), where accept_margin is a configurable fraction (e.g. 0.3, meaning the sample must be comfortably below the normal authentication threshold, not just barely under it). Add config.yaml fields under a new `adaptive` section for accept_margin. Write a small standalone test script demonstrating this function correctly accepts clearly-genuine samples and rejects borderline/anomalous samples using a few hand-constructed example reconstruction errors.
```

## Step 2.7 — Adaptive module: exponential threshold update
```
In src/adaptive/adaptive_update.py, add an update_threshold(old_threshold, recent_error_percentile, alpha) function implementing new_threshold = alpha * old_threshold + (1 - alpha) * recent_error_percentile. Add `alpha` to config.yaml's `adaptive` section (default ~0.97). Add a function recompute_recent_percentile(recent_high_confidence_errors, percentile) that recomputes the reference percentile (matching the original threshold-setting percentile, e.g. 95th) from only the recent samples that passed the Step 2.6 confidence gate. Write a test demonstrating that repeated small genuine drifts shift the threshold gradually, while a single outlier session does not move it much (since alpha keeps updates bounded per step).
```

## Step 2.8 — Adaptive module: N-consecutive-session confirmation
```
In src/adaptive/adaptive_update.py, add a stateful AdaptiveProfileManager class that: tracks a rolling count of consecutive high-confidence genuine sessions (using Step 2.6's gate) per modality; only calls update_threshold() (Step 2.7) once N consecutive qualifying sessions have been observed (N configurable in config.yaml's `adaptive` section, default 3); resets the consecutive counter to zero if any session in between fails the confidence gate. Write a test simulating a sequence of sessions with a mix of pass/fail confidence outcomes and confirm updates only fire at the correct points.
```

## Step 2.9 — Adaptive module: audit log + rollback
```
Extend AdaptiveProfileManager (src/adaptive/adaptive_update.py) to: log every committed threshold update to a JSON Lines audit file (results/adaptive_audit_log.jsonl) with fields {timestamp, modality, old_threshold, new_threshold, triggering_session_ids}; provide a rollback_to(timestamp_or_index) method that restores a previous threshold value from the audit log; provide a simple anomaly check helper flag_possible_bad_adaptation(current_frr, baseline_frr, spike_ratio_threshold) that returns True if FRR spikes significantly right after a recent adaptation (suggesting the adaptation should be investigated/rolled back). Wire this manager into main.py as an optional new stage (`--stage adapt`) that can be run against a sequence of real sessions.
```

## Step 2.10 — Adaptive module: drift + poisoning simulation tests
```
Create tests/test_adaptive_module.py with two simulation scenarios using synthetic data:
SCENARIO A (legitimate drift): generate a sequence of genuine sessions with a slow, gradual parameter drift (e.g. mouse speed_mu slowly increasing over 20 sessions) and confirm that with the adaptive module active, FRR stays roughly stable over the sequence, whereas with the adaptive module OFF (static Phase 1 threshold), FRR visibly rises.
SCENARIO B (slow poisoning attempt): generate a sequence where an "imposter" session's parameters are slowly nudged closer to the genuine profile over many sessions, interspersed with genuine sessions, and confirm the adaptive module does NOT fully adapt to accept the imposter within a reasonable number of sessions (i.e. the N-consecutive-genuine-session gate and confidence margin successfully block gradual poisoning).
Report both scenarios' FAR/FRR numbers in results/adaptive_module_test_results.txt.
```

## Step 2.11 — Imitation-attack: data collection protocol
```
Create docs/imitation_attack_protocol.md describing how to collect mimicry data: recruit 2-3 volunteers, show them a short recording or written description of the genuine user's typing rhythm and mouse movement style (e.g. "moves the mouse in smooth, moderate-speed arcs; types at a steady pace with occasional backspacing"), have each volunteer complete one ~15-30 minute session attempting to mimic that style, and log these sessions with a NEW label value "mimicry" (distinct from ordinary "imposter") using the existing loggers (--label mimicry — update the logger CLI's choices list to include "mimicry" alongside "genuine"/"imposter"). Update src/logging/mouse_logger.py and keystroke_logger.py's --label argparse choices accordingly.
```

## Step 2.12 — Imitation-attack: evaluation breakdown
```
Create src/evaluation/imitation_attack.py: load sessions labeled "mimicry" separately from ordinary "imposter" sessions, and add a function evaluate_by_imposter_type(model, threshold, genuine_test_windows, imposter_windows, mimicry_windows) that computes FAR separately for imposter-only and mimicry-only (each combined with the same genuine test windows), for a given unimodal or fused model. Extend src/evaluation/evaluate.py to call this for mouse-only, keystroke-only, and the fused model, and save the breakdown to results/imitation_attack_results.txt as a table with columns [Model, FAR (passive imposter), FAR (active mimicry)].
```

## Step 2.13 — Drift-resistance evaluation over real data
```
Create src/evaluation/drift_eval.py: given a real, timestamped multi-week genuine dataset (from Phase 0's data/real/ collection), split it into weekly chunks in chronological order. Run two parallel simulations across the weeks: (a) STATIC — threshold fixed from week 1 only, never updated; (b) ADAPTIVE — using the Step 2.6-2.9 AdaptiveProfileManager, updating as qualifying sessions arrive. For each week, compute FRR against that week's genuine data under both conditions, and plot FRR-over-time for both as line series on the same chart, saved to results/drift_comparison.png. Save the underlying weekly numbers to results/drift_comparison.csv.
```

## Step 2.14 — Full evaluation suite consolidation
```
Update src/evaluation/evaluate.py to produce one master evaluation run covering: mouse-only, keystroke-only, basic-fusion (Phase 1 RF with bucket alignment), tuned-fusion (best of Step 2.5's 8 meta-models with generative alignment from Step 2.3), and adaptive-fusion (tuned-fusion + Step 2.9's adaptive module active). Report FAR/FRR/EER/ACC/Precision/Recall/F1/AUC for all five variants in one table (results/final_evaluation_table.txt) plus one combined ROC plot with all five curves overlaid (results/final_roc_comparison.png).
```

## Step 2.15 — Detection delay metric
```
Add a detection-delay simulation to src/evaluation/evaluate.py or a new src/evaluation/detection_delay.py: construct a synthetic "session timeline" that starts with genuine windows and then switches to imposter windows at a known timestamp (simulating a takeover event), run the trained fused model over this timeline in sequence, and measure the time elapsed between the takeover timestamp and the first window correctly flagged as imposter. Repeat over multiple randomized takeover points and report mean/median/std detection delay in seconds, added to results/final_evaluation_table.txt as an additional column or a separate small table.
```

## Step 2.16 — Real data pipeline validation
```
Once at least 1-2 weeks of real genuine data (Phase 0, Step 0.3) plus a few real imposter/mimicry sessions are available, re-run the ENTIRE pipeline (all Phase 0 fixes + all Phase 2 features) against data/real/ instead of synthetic data, end to end. Compare the resulting evaluation numbers against the synthetic-data numbers and explicitly document, in results/real_vs_synthetic_notes.md, how much the real-data FAR/FRR/EER differ from the (expected to be unrealistically perfect) synthetic-data numbers — this comparison itself is a useful thing to include in the paper's discussion/limitations section.
```

## Step 2.17 — Results consolidation + LaTeX tables
```
Using all results/*.txt and results/*.csv files produced by Steps 2.5, 2.12, 2.13, 2.14, 2.15, and 2.16, generate results/final_results_summary.md consolidating every table with a short (2-3 sentence) plain-language interpretation under each, written in a factual tone suitable for a paper's Results section. Also generate a LaTeX-formatted version of each key table (booktabs-style, bold the best value per column) saved as results/tables_latex.tex, ready to paste into an IEEE/Scopus paper template.
```

## Step 2.18 — Patent description draft
```
Using the finished src/adaptive/adaptive_update.py implementation (Steps 2.6-2.9), draft docs/patent_technical_description.md: a precise, numbered, step-by-step description of the adaptive profile update mechanism — the confidence-gating condition and its formula, the N-consecutive-session confirmation rule, the exponential threshold update formula, and the audit-log/rollback mechanism — written so that a person skilled in the art could implement it from the description alone, with no marketing language, no comparative claims, and no reference to specific code variable names (describe it as an algorithm/method, not as "this function does X"). Include one simple flow diagram (as a Mermaid flowchart or numbered ASCII diagram) showing the decision sequence: new session -> confidence check -> consecutive-count check -> threshold update -> audit log -> (optional) rollback trigger.
```
