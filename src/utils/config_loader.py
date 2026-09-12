"""
config_loader.py — Loads and validates the central YAML config.

Provides a single `load_config()` entry point used by every module.
"""

import os
import yaml


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "config.yaml"
)


def load_config(config_path: str | None = None) -> dict:
    """Load the YAML configuration file and return it as a dict.

    Args:
        config_path: Explicit path to config.yaml.  Falls back to the
                     repository-root default if not provided.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the resolved config path does not exist.
        yaml.YAMLError: If the file is malformed YAML.
    """
    path = config_path or os.path.abspath(_DEFAULT_CONFIG_PATH)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    return cfg


def get_nested(cfg: dict, *keys, default=None):
    """Safely drill into nested config keys.

    Example:
        get_nested(cfg, "model_mouse", "learning_rate", default=1e-3)
    """
    node = cfg
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k, default)
        else:
            return default
    return node
