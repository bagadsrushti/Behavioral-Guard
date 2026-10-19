# Behavioral Guard: Real Data Collection Protocol

## 1. Using the Loggers Directly
To manually start a session, use the existing background loggers. You must start both for each session.

**Mouse Logger:**
```bash
python -m src.logging.mouse_logger --session_id <id> --label genuine --output data/real/raw/mouse_<id>.csv
```

**Keystroke Logger:**
```bash
python -m src.logging.keystroke_logger --session_id <id> --label genuine --output data/real/raw/keystroke_<id>.csv
```

## 2. Recommended Collection Schedule
*   **Genuine Sessions:** Collect multiple sessions per day, spread across at least **10-14 days**.
*   **Target Volume:** Aim for a similar total-event volume per session as the synthetic sessions (approx. **~2000 mouse events, ~1000 keystroke events** per session) so the sliding window comparisons remain consistent.

## 3. Imposter Data Collection
*   Recruit 2-3 different people to use your device.
*   Ask each person to complete one ~15-30 minute session.
*   Ensure their sessions are clearly labeled using `--label imposter`.
*   **IMPORTANT:** Explicit consent must be obtained from anyone whose behavior is logged on your device.

## 4. Keystroke Privacy Guarantee
*   The keystroke logger does **NOT** log or store actual typed characters, letters, numbers, or passwords.
*   Keys are mapped in real time to coarse functional categories: `"letter"`, `"digit"`, `"space"`, `"backspace"`, `"enter"`, `"tab"`, `"modifier"`, `"punctuation"`, and `"other"`.
*   Only the event timestamp, the key category, and the event type (`key_down` / `key_up`) are persisted.
*   This ensures that even when recording real typing sessions on private devices, no sensitive text, passwords, or personal messages are ever captured.

## 5. Note on Synthetic Data
The synthetic data generator (`main.py --stage generate`) exists **only for pipeline smoke-testing and development**. Its data is easily separable by design. Synthetic results must **never** be reported as actual evaluation results in the final publication or patent application. By default, existing raw data will never be overwritten unless `--force` is explicitly provided.

