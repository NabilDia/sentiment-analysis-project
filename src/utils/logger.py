"""
Centralised logging configuration.

Usage
-----
from src.utils.logger import setup_logging
setup_logging()          # reads logging_config.yaml automatically
logger = logging.getLogger(__name__)
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: Optional[Path] = None,
    default_level: int = logging.INFO,
) -> None:
    """Configure the root logger from a YAML file or with sensible defaults.

    Parameters
    ----------
    config_path:
        Path to a ``logging_config.yaml`` file.  When *None* the function
        looks for ``logging_config.yaml`` in the project root (two levels up
        from this file).
    default_level:
        Fallback log level used when no config file is found.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "logging_config.yaml"

    if config_path.exists():
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        # Ensure the log directory exists
        for handler in cfg.get("handlers", {}).values():
            filename = handler.get("filename")
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.getLogger(__name__).warning(
            "Logging config not found at %s – using basicConfig.", config_path
        )
