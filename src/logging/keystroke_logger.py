"""
keystroke_logger.py — Background keystroke-event capture with simple start/stop CLI.

Captures: timestamp, key_identity, event_type (key_down / key_up).
Output is an append-only CSV tagged with session_id and label.

Usage:
    python -m src.logging.keystroke_logger --session_id s01 --label genuine \
           --output data/raw/keystroke_s01.csv
    (Press Ctrl+C to stop)
"""

import argparse
import csv
import logging
import os
import time

from pynput import keyboard

from src.utils.config_loader import load_config
from src.utils.logger_setup import setup_logging

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "key", "event_type", "session_id", "label"]


def _key_name(key) -> str:
    """Normalise a pynput key object to a readable string."""
    if hasattr(key, "char") and key.char is not None:
        return key.char
    return str(key).replace("Key.", "")


class KeystrokeLogger:
    """Records key-down and key-up events to CSV."""

    def __init__(self, output_path: str, session_id: str, label: str):
        self.output_path = output_path
        self.session_id = session_id
        self.label = label
        self._listener: keyboard.Listener | None = None
        self._file = None
        self._writer = None

    # -- internal -----------------------------------------------------------

    def _write_row(self, key_name: str, event_type: str) -> None:
        ts = time.time()
        self._writer.writerow(
            [ts, key_name, event_type, self.session_id, self.label]
        )
        self._file.flush()

    # -- pynput callbacks ---------------------------------------------------

    def _on_press(self, key) -> None:
        self._write_row(_key_name(key), "key_down")

    def _on_release(self, key) -> None:
        self._write_row(_key_name(key), "key_up")

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        file_exists = os.path.isfile(self.output_path)
        self._file = open(self.output_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        if not file_exists or os.path.getsize(self.output_path) == 0:
            self._writer.writerow(COLUMNS)
            self._file.flush()

        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()
        log.info(
            "Keystroke logger started — session=%s, label=%s, output=%s",
            self.session_id,
            self.label,
            self.output_path,
        )

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._file is not None:
            self._file.close()
            self._file = None
        log.info("Keystroke logger stopped — session=%s", self.session_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record keystroke events to an append-only CSV."
    )
    parser.add_argument("--session_id", required=True)
    parser.add_argument(
        "--label", required=True, choices=["genuine", "imposter"]
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    raw_dir = cfg["paths"]["raw_data_dir"]
    output = args.output or os.path.join(
        raw_dir, f"keystroke_{args.session_id}.csv"
    )

    logger = KeystrokeLogger(output, args.session_id, args.label)
    logger.start()

    try:
        print(
            f"Recording keystrokes (session={args.session_id}).  "
            "Press Ctrl+C to stop."
        )
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
