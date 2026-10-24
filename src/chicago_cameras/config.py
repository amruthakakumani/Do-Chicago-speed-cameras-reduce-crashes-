"""Configuration loading and path helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/analysis.yaml")
REQUIRED_SECTIONS = ("data", "spatial", "design", "inference", "paths")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the YAML analysis configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Run commands from the repository root or pass --config explicitly."
        )
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"{config_path} is missing required sections: {missing}")

    radius = float(config["spatial"]["radius_m"])
    buffer_m = float(config["spatial"]["buffer_m"])
    if buffer_m <= 2 * radius:
        raise ValueError(
            "spatial.buffer_m must exceed 2 * spatial.radius_m so that treated and "
            f"comparison catchments cannot overlap (got {buffer_m} <= {2 * radius})."
        )

    config["_config_path"] = str(config_path)
    return config


def project_root(config: dict[str, Any] | None = None) -> Path:
    """Repository root: the parent of the config directory."""
    if config is not None and "_config_path" in config:
        return Path(config["_config_path"]).resolve().parent.parent
    return Path.cwd()


def path_for(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from ``config['paths']`` relative to the repository root."""
    root = project_root(config)
    target = root / config["paths"][key]
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
