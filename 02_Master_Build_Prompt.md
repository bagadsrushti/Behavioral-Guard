# Behavioral Guard — Master Build Prompt (Phase-Wise)

How to use this file: each phase below is a **self-contained prompt** in a code block. Paste one phase's prompt into Antigravity at a time, in order. Phase 0 must be done first (it fixes known bugs in the existing repo). Phase 1 is already built — its prompt is included only for context/rebuild/onboarding purposes, not to be re-run as-is. Phase 2 is the real remaining work. Phase 3 covers evaluation + writing support.

---

## PHASE 0 — Bug Fixes & Real Data Kickoff (do this first, before any new feature work)

```
I have an existing repo called "Behavioral Guard" (continuous authentication via mouse + keystroke behavioral biometrics, LSTM-Autoencoder anomaly detection, Random Forest decision-level fusion). The repo structure is:

src/{logging, features, models, fusion, evaluation, utils}
configs/config.yaml
data/{raw, processed, synthetic}
models_checkpoints/
results/
main.py, requirements.txt, README.md

Two confirmed bugs need fixing before any new work:

BUG 1 — Mouse feature normalization missing:
- src/features/mouse_features.py computes raw kinematic features (event_type_enc, x, y, delta_t, delta_x, delta_y, d, v, v_x, v_y, acc, acc_x, acc_y, jerk, jerk_x, jerk_y, theta, direction, k, r) with no scaling.
- src/models/train_mouse.py feeds these raw features directly into the LSTM-Autoencoder.
- Result: training loss is enormous (tens of millions) and does not converge — the model isn't learning meaningful reconstruction.
- FIX: add a StandardScaler (or equivalent) step. Fit the scaler ONLY on the training split's mouse features, save the fitted scaler (e.g. pickle it alongside the model checkpoint), apply it to train/val/test features before feeding to the LSTM-AE, and apply the SAME saved scaler at inference/evaluation time. Do not fit a new scaler on validation or test data.
- Verify the fix by re-running training and confirming: (a) loss values are in a reasonable range (not millions), and (b) loss decreases meaningfully across epochs, not staying flat.

BUG 2 — Data leakage in the fusion stage:
- In main.py's stage_fuse(), the Random Forest meta-model is trained using ALL imposter sessions.
- In src/evaluation/evaluate.py's fused-model evaluation, the SAME imposter sessions are used again to test the Random Forest.
- This means the fusion model is evaluated on data it was trained on, inflating its apparent performance.
- FIX: split imposter sessions into a "fusion-train" subset and a "fusion-test" subset (e.g. 60/40 split, or hold out at least 1-2 full imposter sessions exclusively for testing), applied consistently in both stage_fuse() (main.py) and the fused-model evaluation block (evaluate.py). Never let the same imposter session appear in both the fusion training set and the fusion test set.
- Update config.yaml to add an explicit fusion imposter train/test ratio if useful, rather than hardcoding the split.

After both fixes:
- Re-run the full pipeline (python main.py) on synthetic data as a regression check.
- Report the new evaluation table (FAR/FRR/EER/AUC for mouse-only, keystroke-only, fused) and flag if results still look suspiciously perfect (which would indicate the synthetic data itself is too easily separable — that's expected and fine for a smoke test, just don't treat it as a real result).

Additionally, scaffold (but do not fully build yet) a `data/real/` directory structure and a short `docs/data_collection_protocol.md` describing: how many days of genuine-user logging is needed, how to label imposter sessions, and how to invoke the existing mouse_logger.py / keystroke_logger.py CLIs for real collection — so real data collection can start immediately in parallel with Phase 2 development.
```

---

## PHASE 1 — Mid-Sem Prototype (ALREADY BUILT — reference only, for rebuild/onboarding)

```
[REFERENCE ONLY — this phase is already implemented in the current repo. Use this prompt only if rebuilding the project from scratch on a new machine, or onboarding a new teammate who needs to understand what exists.]

Build a mid-semester prototype for continuous authentication using behavioral biometrics (mouse + keystroke). Components: mouse/keystroke event loggers (pynput-based background capture), feature engineering (20 mouse kinematic features, 6 keystroke timing features) with sliding-window segmentation, a shared LSTM-Autoencoder architecture (2-layer LSTM encoder + FC bottleneck, FC + 2-layer LSTM decoder) trained one-class per modality with MSE reconstruction loss, per-modality 95th-percentile reconstruction-error thresholding, basic fixed-time-window decision-level fusion via a single Random Forest meta-model, and an evaluation module computing FAR/FRR/EER/ROC/AUC with saved plots. Include a synthetic data generator for smoke testing, a centralized YAML config, Python logging throughout, checkpointing, a README documenting what's implemented vs. planned for Phase 2, and a requirements.txt. Reference: Li et al., "MBBFAuth," IEEE TIFS 2024 — adapt their feature engineering, LSTM-AE anomaly detection, and decision-level fusion concepts to this two-modality, single-user setting.
```

---

## PHASE 2 — End-Sem Build: Alignment, Fusion Tuning, Adaptive Module, Adversarial Testing

```
Extend the existing Behavioral Guard repo (Phase 0 and Phase 1 already complete — mouse + keystroke LSTM-AEs, basic RF fusion, fixed bugs) with the following Phase 2 components. Build on the existing src/ structure; do not restructure what already works.

1. MODALITY ALIGNMENT (replace fixed time-bucket fusion)
   - Implement a proper temporal alignment layer between mouse and keystroke streams, replacing src/fusion/fuse.py's current fixed-time-bucket aggregation.
   - Use a CTGAN-based or TimeGAN-based approach: model each modality's per-window feature distribution, and use the generative model to align/interpolate both streams onto a shared, consistent time scale before meta-feature extraction, rather than simply averaging within fixed buckets.
   - Keep the old fixed-time-bucket method available as a fallback/comparison baseline (flag-controlled in config.yaml) so you can report "old alignment vs. new alignment" as a result.
   - Place this in a new module: src/alignment/modality_align.py

2. FULL META-MODEL COMPARISON (extend fusion)
   - Extend src/fusion/fuse.py to train and evaluate all 8 candidate meta-model algorithms: Naive Bayes, Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting Machine, LightGBM.
   - Use grid search or reasonable default hyperparameters per algorithm; use the SAME meta-feature set and SAME train/test split (with the Phase 0 leakage fix in place) for a fair comparison.
   - Output a comparison table (FAR/FRR/EER/AUC per meta-model) to results/, and select the best-performing model as the default for downstream use, but keep all 8 results for the paper.

3. ADAPTIVE LEARNING MODULE (core novel contribution — build this carefully)
   - New module: src/adaptive/adaptive_update.py
   - Implement confidence-gated incremental profile updates:
     a. Define a strict acceptance threshold (tighter than the authentication threshold) — a sample only qualifies for adaptation if its reconstruction error is well below the normal genuine-acceptance threshold (a configurable margin).
     b. Require N consecutive high-confidence genuine sessions (configurable, e.g. N=3) before committing any profile update — a single session should never immediately redefine "genuine."
     c. Implement the update as an exponential moving update of the per-modality reconstruction-error threshold: new_threshold = alpha * old_threshold + (1-alpha) * recent_threshold_estimate, with alpha configurable (default ~0.95-0.99) to bound how fast the profile can drift.
     d. Optionally (if time allows) also support periodic fine-tuning of the LSTM-AE weights themselves on a rolling buffer of high-confidence recent genuine windows, with the same N-session confirmation gate.
     e. Implement a rollback mechanism: log every adaptation event (timestamp, old threshold, new threshold, triggering sessions) to a JSON/CSV audit log, and provide a function to revert to a previous threshold/profile state if a later session strongly contradicts a recent adaptation (e.g. a sudden spike in FRR immediately after an update).
   - Write unit tests simulating: (a) genuine drift over time — profile should adapt and FRR should NOT rise, (b) a simulated slow-poisoning attack — an imposter's behavior gradually shifted toward the profile over many sessions — the module should NOT fully adapt to accept the imposter within a reasonable number of sessions. Report both test outcomes numerically.

4. IMITATION-ATTACK EVALUATION
   - New module: src/evaluation/imitation_attack.py
   - Support loading a "mimicry" labeled dataset (sessions from a third party deliberately trying to copy the genuine user's typing/mouse style) as a separate category from ordinary "imposter" (unrelated user) sessions.
   - Extend the evaluation module to report FAR separately for: passive imposters (ordinary different users) vs. active imposters (deliberate mimicry), for each of mouse-only, keystroke-only, and fused models.
   - Add this breakdown to the results table and to a dedicated results/imitation_attack_results.txt

5. DRIFT-RESISTANCE EVALUATION
   - New module: src/evaluation/drift_eval.py
   - Given a real, timestamped multi-week genuine-user dataset, simulate running the system WITHOUT adaptation (static profile from Phase 1) vs. WITH the adaptive module (Phase 2, item 3), and plot FRR over time (week by week) for both, to show whether adaptation prevents FRR from rising as behavior drifts.
   - Save the comparison plot to results/drift_comparison.png

6. FULL EVALUATION SUITE
   - Update src/evaluation/evaluate.py to run and report: unimodal (mouse, keystroke) vs. basic-fusion (Phase 1 fixed RF) vs. tuned-fusion (best of the 8 meta-models) vs. adaptive-fusion (with the adaptive module active), across FAR/FRR/EER/ACC/P/R/F1/AUC, plus detection delay (time from a simulated imposter takeover event to the first correct imposter flag).
   - Produce one consolidated results table and one consolidated ROC comparison plot with all model variants overlaid.

Config: extend configs/config.yaml with new sections for `alignment`, `adaptive`, and `imitation_attack` — no hardcoded parameters in code. Keep all existing Phase 0/1 functionality working; this is additive, not a rewrite. Ask before making any breaking changes to existing module interfaces that Phase 1 code depends on.
```

---

## PHASE 3 — Evaluation Consolidation, Paper Support, Patent Support

```
Using the results produced by Phase 2 (evaluation tables, ROC plots, drift-resistance plots, imitation-attack breakdown), do the following:

1. Generate a single consolidated results document (results/final_results_summary.md) that presents, in order: (a) unimodal baseline results, (b) basic decision-level fusion results, (c) tuned meta-model comparison table (all 8 algorithms), (d) adaptive vs. static drift-resistance comparison, (e) passive vs. active imposter (imitation-attack) breakdown, (f) detection delay statistics. Each section should have a 2-3 sentence plain-language interpretation of what the numbers show, written for inclusion in a paper's Results section (factual tone, no marketing language).

2. Generate a LaTeX-ready table (or CSV that can be converted) for each results table above, formatted in a style similar to typical IEEE TIFS/Scopus paper tables (model name, FAR, FRR, EER, AUC columns; bold the best value per column).

3. Draft a plain technical description (not marketing copy) of the Adaptive Learning Module's algorithm (from Phase 2, item 3) suitable for inclusion in a patent application's "detailed description" section: describe the confidence-gating condition, the N-consecutive-session confirmation rule, the exponential threshold update formula, and the rollback/audit-log mechanism, as a numbered step-by-step process a person skilled in the art could implement from the description alone. Do not include marketing language or comparative claims — just a precise, implementable description of the mechanism.

4. Flag, in a short list, any results that look suspicious (e.g. FAR or FRR of exactly 0.0000, AUC of exactly 1.0000) so they can be sanity-checked against real (not synthetic) data before being reported in the paper or patent filing.
```
