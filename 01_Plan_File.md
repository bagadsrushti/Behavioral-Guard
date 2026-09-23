# Behavioral Guard — Complete Project Plan
### Adaptive Multimodal Behavioral Biometrics for Continuous Authentication (Single-User Laptops/PCs)

Reference baseline: Li et al., *"MBBFAuth: Multimodal Behavioral Biometrics Fusion for Continuous Authentication on Non-Portable Devices,"* IEEE TIFS, 2024.
Goals: (1) working prototype, (2) IEEE/Scopus publication, (3) patent filing on the adaptive mechanism.

---

## PART A — CONCEPTUAL FOUNDATIONS (with formulas)

Every concept the pipeline depends on is explained here once, in full, so it can be referenced from the build prompts and the paper draft without re-deriving anything.

### A.1 Behavioral Biometrics & Continuous Authentication (CA)

**Why CA exists:** login-time authentication (password/PIN/biometric) verifies identity only once. After that, the session is trusted indefinitely. CA re-verifies identity *continuously* using passive signals — no extra action from the user — by comparing live behavior against a learned profile of the genuine user.

**Core idea used here:** treat authentication as **one-class anomaly detection**. We only ever have reliable data for the genuine user (the owner). We never have real imposter data at training time (an imposter is, by definition, unknown in advance). So the model learns "what normal looks like" and flags deviation — it does not learn "what an imposter looks like."

---

### A.2 Modalities Used

| Modality | Signal | Why it's useful |
|---|---|---|
| **MB — Mouse Behavior** | Cursor movement, velocity, acceleration, click patterns | Continuous, high-frequency, hard to consciously control/fake |
| **KD — Keystroke Dynamics** | Key hold time, timing between keys | Deeply ingrained motor pattern, highly individual |

(IIB — information interaction behavior — is the reference paper's third modality; optional future extension, see Part C.)

---

### A.3 Mouse Behavior Feature Formulas

Given a sequence of mouse events with timestamp `t`, position `(x, y)`:

```
Δt_i   = t_i − t_{i-1}                         (time delta)
Δx_i   = x_i − x_{i-1}
Δy_i   = y_i − y_{i-1}

d_i    = √(Δx_i² + Δy_i²)                      (Euclidean distance moved)

v_i    = d_i / Δt_i                            (speed)
v_x,i  = Δx_i / Δt_i                           (horizontal speed)
v_y,i  = Δy_i / Δt_i                           (vertical speed)

acc_i    = (v_i − v_{i-1}) / Δt_i              (acceleration)
acc_x,i  = (v_x,i − v_x,i-1) / Δt_i
acc_y,i  = (v_y,i − v_y,i-1) / Δt_i

jerk_i   = (acc_i − acc_{i-1}) / Δt_i          (rate of accel. change)
jerk_x,i = (acc_x,i − acc_x,i-1) / Δt_i
jerk_y,i = (acc_y,i − acc_y,i-1) / Δt_i

θ_i        = atan2(Δy_i, Δx_i)                 (movement bearing, range [−π, π])
direction_i = θ_i − θ_{i-1}                    (turning: >0 CCW, <0 CW, 0 straight)

k_i    = |θ_i − θ_{i-1}| / d_i                 (curvature)
r_i    = 1 / k_i                               (radius of curvature; capped to avoid ÷0 → ∞)
```

**Why this matters:** velocity/acceleration/jerk profiles and curvature capture the *physical dynamics* of how someone moves a mouse — smoothness, overshoot, correction habits — which differ meaningfully between individuals and are difficult to consciously imitate over a full session.

---

### A.4 Keystroke Dynamics Feature Formulas

Given key-press (`down`) and key-release (`up`) events:

```
hold_time_i (dwell)     = t_up(key_i) − t_down(key_i)
flight_time_i           = t_down(key_i) − t_down(key_{i-1})     (a.k.a. digraph latency)
trigraph_latency_i      = t_down(key_i) − t_down(key_{i-2})     (spans 3 keys)

typing_speed_i  = (N−1) / (t_down(key_i) − t_down(key_{i-N+1})) (keys/sec, rolling window of N)

backspace_freq_i = mean( is_backspace ) over a rolling window   (correction rate)
```

**Why this matters:** dwell/flight times encode finger dexterity and muscle memory that is extremely stable for a given person over short-to-medium timeframes, and is one of the most well-studied behavioral biometrics in the literature.

---

### A.5 Sliding Window Segmentation

Raw events are turned into fixed-length sequences before being fed to the model:

```
window[j] = features[ j·stride : j·stride + window_size ]
```
- `window_size` = number of consecutive events per sequence (e.g. 50 for mouse, 30 for keystroke)
- `stride` = step between window starts (overlap = window_size − stride)

This converts an irregular event stream into fixed-shape tensors `(window_size, num_features)` suitable for sequence models.

---

### A.6 LSTM Cell (core recurrent unit)

For input `B_t` at time step `t`, previous hidden state `h_{t-1}`, previous cell state `C_{t-1}`:

```
f_t = σ(W_f · [h_{t-1}, B_t] + b_f)        # forget gate
i_t = σ(W_i · [h_{t-1}, B_t] + b_i)        # input gate
C̃_t = tanh(W_C · [h_{t-1}, B_t] + b_C)     # candidate cell state
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t            # new cell state
o_t = σ(W_o · [h_{t-1}, B_t] + b_o)        # output gate
h_t = o_t ⊙ tanh(C_t)                      # new hidden state
```
`σ` = sigmoid, `tanh` = hyperbolic tangent, `⊙` = elementwise product. LSTMs are used because behavioral sequences have **long-term temporal dependencies** (a typing rhythm or mouse-movement style persists across many timesteps) that plain feed-forward networks can't capture.

---

### A.7 LSTM-Autoencoder (LSTM-AE) — the per-modality anomaly detector

**Encoder:** 2 stacked LSTM layers → 1 fully-connected (FC) layer → low-dimensional embedding `z`
**Decoder:** FC layer → 2 stacked LSTM layers → FC layer → reconstructed sequence `B''`

```
z          = Encoder(B)              # compress sequence to a bottleneck vector
B'' (recon) = Decoder(z)             # attempt to reconstruct the original sequence
```

**Why an autoencoder:** it is trained **unsupervised**, on genuine-user data only (one-class). It never needs imposter examples. If it has only ever learned to reconstruct the owner's behavior well, it will reconstruct an imposter's (different) behavior *poorly* — that reconstruction gap is the anomaly signal.

---

### A.8 Reconstruction Error & Threshold (anomaly decision rule)

```
MSE(B, B'') = (1/N) · Σ_{t=1}^{N} (B_t − B''_t)²
```

**Threshold setting (static, Phase 1):**
```
threshold = 95th percentile of MSE over genuine VALIDATION data
```

**Decision rule:**
```
if MSE(new_sequence) > threshold:  →  classify as IMPOSTER
else:                              →  classify as GENUINE
```

This threshold is deliberately set only from genuine data (no imposter data used) — consistent with the one-class assumption.

---

### A.9 Modality Alignment (Phase 2 — CTGAN / TimeGAN)

**Problem:** mouse events (~10 Hz continuous) and keystroke events (bursty, only while typing) have different, uncorrelated sampling patterns. Before fusing them, both streams must be represented on a **common time scale**.

**Phase 1 (built already):** simple fixed time-bucket aggregation — group both modalities' windows into the same N-second buckets and average.

**Phase 2 (planned):** a generative approach (CTGAN-style, adapted for time series / TimeGAN) learns to model each modality's distribution and produce/interpolate aligned synthetic samples so that both streams can be resampled onto identical time steps without simply discarding data via bucket-averaging. Conceptually, a GAN alternates:

```
Generator G: tries to produce synthetic samples indistinguishable from real data
Discriminator D: tries to distinguish real vs. generated samples
min_G max_D  E[log D(x_real)] + E[log(1 − D(G(z)))]
```
G and D are trained adversarially until G's outputs are statistically indistinguishable from real behavioral data, at which point G can be used to align/interpolate the modality streams onto a shared timeline while preserving each modality's statistical structure.

---

### A.10 Decision-Level Fusion — Meta-Features & Meta-Model

Per modality, per time-aligned window, compute:

```
DET (Deviation from Error Threshold) = reconstruction_error − threshold
SP  (Sum of Predictions)             = Σ (binary genuine/imposter label across modalities)
```

These meta-features (`[mouse_error, mouse_pred, keystroke_error, keystroke_pred]` in Phase 1; extended with `DET`/`SP` in Phase 2) are fed into a **meta-model** (stacking) trained to output the final decision:

```
final_decision = MetaModel( [DET_mouse, SP_mouse, DET_keystroke, SP_keystroke, ...] )
```

**Phase 1:** single meta-model — Random Forest.
**Phase 2:** exhaustive comparison across 8 candidate algorithms — Naive Bayes, Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, LightGBM — selecting the best by validation performance (reference paper found Random Forest best; this is a hypothesis to confirm or refute for your own data).

**Why decision-level (not feature-level) fusion:** each modality is evaluated independently first, so the system remains functional even if one modality's data is temporarily unavailable (e.g., user is only using the mouse, not typing) — a robustness property that feature-level fusion doesn't offer.

---

### A.11 Evaluation Metrics

```
FAR (False Acceptance Rate) = FN / (FN + TP)     # imposters wrongly let in
FRR (False Rejection Rate)  = FP / (FP + TN)     # genuine user wrongly locked out
EER (Equal Error Rate)      = point where FAR = FRR (lower is better; single-number security/usability tradeoff summary)

ACC (Accuracy)   = (TP + TN) / (TP + TN + FP + FN)
P   (Precision)  = TP / (TP + FP)
R   (Recall)     = TP / (TP + FN)
F1               = 2 · P · R / (P + R)

ROC curve: TPR vs FPR across all thresholds; AUC = area under this curve (1.0 = perfect separability)
```
Convention used in this project: **TP = imposter correctly flagged**, **TN = genuine correctly accepted** (matches the reference paper's convention — double-check this convention is applied consistently in code, since flipping it silently inverts FAR/FRR).

**New metric introduced in this project (not in the reference paper):**
```
Detection Delay = t(imposter flagged) − t(imposter took control of device)
```
This measures *how quickly* a takeover is caught — a practical usability/security metric the reference paper never measured (it only measured processing latency, not detection speed).

---

### A.12 Adaptive Learning Module — the core novel contribution

**Problem being solved:** a static profile (fixed after enrollment) degrades as the genuine user's behavior naturally drifts (fatigue, new habits, minor injury, different mouse/keyboard, etc.), causing rising FRR over time. But naively retraining on "whatever comes in" is dangerous — an imposter who slowly nudges the profile toward their own behavior could eventually get accepted (**profile poisoning**).

**Design principle — confidence-gated incremental update:**
```
Accept a new sample for adaptation ONLY IF:
    reconstruction_error(sample) < τ_accept       # strict confidence bound, tighter than the auth threshold
    AND
    sample passed authentication with high margin  # not just barely under threshold
```

**Update rule (exponential moving update of the threshold / profile statistics):**
```
θ_t = α · θ_{t-1} + (1 − α) · θ_new
```
where `θ` can represent the reconstruction-error threshold, or (in a fuller version) periodic fine-tuning of the LSTM-AE weights on a rolling buffer of high-confidence recent genuine samples. `α` (close to 1, e.g. 0.95–0.99) controls how slowly the profile is allowed to move — bounding the *maximum drift per update* is the anti-poisoning safeguard.

**Additional safeguard (dual confirmation):** don't let a single session silently redefine "genuine" — require **N consecutive high-confidence sessions** agreeing before committing an update, and log/rollback capability if a later session strongly contradicts a recent adaptation.

This mechanism — precisely specified, with bounded update rate and dual confirmation — is the primary **patentable claim** of this project. The rest of the pipeline (LSTM-AE, stacking fusion) is adapted from prior art (MBBFAuth) and is not independently patentable.

---

### A.13 Imitation-Attack Evaluation (adversarial robustness — not tested in the reference paper)

**Protocol:**
1. Show a third party (not the owner) recordings/descriptions of the owner's typing/mouse style.
2. Ask them to deliberately mimic that style for a test session.
3. Run the mimicked session through the trained pipeline.
4. Compare FAR under **passive imposters** (a random different user) vs. **active imposters** (deliberate mimicry).

**Why this matters:** a system that only reports low FAR against passive/random users but fails against deliberate mimicry is not a solid publication or patent claim — reviewers and patent examiners will ask this question if you don't answer it yourself first.

---

## PART B — FULL PROJECT PLAN (phase mapping)

### Phase 0 — Remediation (do this before anything else — see analysis notes)
1. Fix mouse feature normalization (StandardScaler, fit on training data only)
2. Fix fusion data-leakage (hold out imposter sessions exclusively for fusion testing — never reuse fusion-training imposter sessions at test time)
3. Begin **real** genuine + imposter data collection immediately (synthetic data is for smoke-testing only, never for reported results)

### Phase 1 — Mid-Sem (50%) — ALREADY BUILT, pending Phase 0 fixes
- Mouse + keystroke event loggers
- Feature engineering (20 mouse features, 6 keystroke features)
- Two independent LSTM-Autoencoders (one-class, per modality)
- Static (95th-percentile) threshold anomaly detection
- Basic fixed-time-window fusion + single Random Forest meta-model
- FAR/FRR/EER/ROC evaluation

### Phase 2 — End-Sem (remaining 50%) — the novelty phase
1. Proper CTGAN/TimeGAN-based modality alignment (replacing fixed time-bucket)
2. Full 8-algorithm stacking comparison, select best meta-model
3. **Adaptive Learning Module** (Part A.12) — design, implement, safeguard
4. Imitation-attack evaluation (Part A.13)
5. Drift-resistance testing: static profile vs. adaptive profile over multi-week real usage
6. Full evaluation suite + unimodal vs. basic-fusion vs. adaptive-fusion comparison
7. Paper draft (IEEE/Scopus format) + patent claim draft (centered on A.12)

---

## PART C — Optional Extensions (only if time allows)
- Add IIB (information interaction behavior — scroll depth, dwell time, navigation) as a third modality, making the modality set a strict superset alignment with the reference paper (CB→KD swap, MB kept, IIB kept)
- Federated/on-device-only learning framing for the privacy angle (the reference paper explicitly flags privacy as unresolved)

---

## PART D — Paper & Patent Mapping (quick reference)

| Section of paper | Backed by |
|---|---|
| Related work / gap analysis | Reference paper's own stated limitations (drift, adversarial testing) — Part A.12, A.13 |
| Methodology | Parts A.3–A.11 |
| Core contribution | Part A.12 (adaptive mechanism) |
| Robustness evaluation | Part A.13 |
| Results | Part A.11 metrics, drift-resistance comparison |
| Patent claim | Part A.12's specific update rule + dual-confirmation safeguard — describe as a concrete algorithm, not just a general idea |
