"""
mouse_logger.py — Background mouse-event capture with simple start/stop CLI.

Captures: timestamp, event_type (move/left_down/left_up/right_down/right_up),
x, y.  Move events are rate-limited to ~10 Hz; click events are immediate.
Output is an append-only CSV tagged with session_id and label.

Usage:
    python -m src.logging.mouse_logger --session_id s01 --label genuine \
           --output data/raw/mouse_s01.csv
    (Press Ctrl+C to stop)
"""

import argparse
import csv
import logging
import os
import time

from pynput import mouse

from src.utils.config_loader import load_config
from src.utils.logger_setup import setup_logging

log = logging.getLogger(__name__)

# ---- Column schema for the CSV output ----
COLUMNS = ["timestamp", "event_type", "x", "y", "session_id", "label"]


class MouseLogger:
    """Rate-limited mouse-event recorder."""

    def __init__(
        self,
        output_path: str,
        session_id: str,
        label: str,
        sampling_rate_hz: float = 10.0,
    ):
        self.output_path = output_path
        self.session_id = session_id
        self.label = label
        self.min_interval = 1.0 / sampling_rate_hz  # seconds between move events
        self._last_move_time = 0.0
        self._listener: mouse.Listener | None = None
        self._file = None
        self._writer = None

    # -- internal helpers --------------------------------------------------

    def _write_row(self, event_type: str, x: int, y: int) -> None:
        ts = time.time()
        self._writer.writerow(
            [ts, event_type, x, y, self.session_id, self.label]
        )
        self._file.flush()

    # -- pynput callbacks ---------------------------------------------------

    def _on_move(self, x: int, y: int) -> None:
        now = time.time()
        if now - self._last_move_time >= self.min_interval:
            self._write_row("move", x, y)
            self._last_move_time = now

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        if button == mouse.Button.left:
            etype = "left_down" if pressed else "left_up"
        elif button == mouse.Button.right:
            etype = "right_down" if pressed else "right_up"
        else:
            etype = f"{button.name}_{'down' if pressed else 'up'}"
        self._write_row(etype, x, y)

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Open the output file and begin listening for mouse events."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        file_exists = os.path.isfile(self.output_path)
        self._file = open(self.output_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        # Write header only if the file is new / empty
        if not file_exists or os.path.getsize(self.output_path) == 0:
            self._writer.writerow(COLUMNS)
            self._file.flush()

        self._listener = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click
        )
        self._listener.start()
        log.info(
            "Mouse logger started — session=%s, label=%s, output=%s",
            self.session_id,
            self.label,
            self.output_path,
        )

    def stop(self) -> None:
        """Stop the listener and close the file."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._file is not None:
            self._file.close()
            self._file = None
        log.info("Mouse logger stopped — session=%s", self.session_id)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record mouse events to an append-only CSV."
    )
    parser.add_argument("--session_id", required=True, help="Unique session identifier")
    parser.add_argument(
        "--label",
        required=True,
        choices=["genuine", "imposter"],
        help="Session label",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/raw/mouse_<session_id>.csv)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    raw_dir = cfg["paths"]["raw_data_dir"]
    output = args.output or os.path.join(raw_dir, f"mouse_{args.session_id}.csv")
    rate = cfg.get("collection", {}).get("mouse", {}).get("sampling_rate_hz", 10)

    logger = MouseLogger(output, args.session_id, args.label, rate)
    logger.start()

    try:
        print(f"Recording mouse events (session={args.session_id}).  Press Ctrl+C to stop.")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
