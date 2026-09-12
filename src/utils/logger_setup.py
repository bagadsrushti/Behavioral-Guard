"""
logger_setup.py — Configures Python's logging module for the whole pipeline.

Call `setup_logging()` once at pipeline entry; every module then uses
`logging.getLogger(__name__)` as usual.
"""

import logging
import os

from src.utils.config_loader import load_config


def setup_logging(cfg: dict | None = None) -> None:
    """Initialise root logger from config settings.

    Args:
        cfg: Pre-loaded config dict.  Loaded from disk if *None*.
    """
    if cfg is None:
        cfg = load_config()

    log_cfg = cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    log_file = log_cfg.get("log_file", "results/pipeline.log")
    fmt = log_cfg.get(
        "format",
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    )

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, level_str, logging.INFO)

    # Configure root logger
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)

    logging.getLogger(__name__).debug(
        "Logging initialised — level=%s, file=%s", level_str, log_file
    )
