"""
Configuration management for the sentiment analysis project.
Loads settings from config.yaml and provides a central Config object.
"""

import os
import yaml
from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


class Config:
    """Central configuration object built from config.yaml."""

    def __init__(self, config_path: Path = CONFIG_FILE):
        raw = load_config(config_path)

        # Paths
        paths = raw.get("paths", {})
        self.data_raw_dir: Path = PROJECT_ROOT / paths.get("data_raw", "data/raw")
        self.data_processed_dir: Path = PROJECT_ROOT / paths.get(
            "data_processed", "data/processed"
        )
        self.models_dir: Path = PROJECT_ROOT / paths.get("models", "models")
        self.logs_dir: Path = PROJECT_ROOT / paths.get("logs", "logs")

        # Preprocessing
        preprocessing = raw.get("preprocessing", {})
        self.language: str = preprocessing.get("language", "english")
        self.remove_stopwords: bool = preprocessing.get("remove_stopwords", True)
        self.lemmatize: bool = preprocessing.get("lemmatize", True)

        # Features
        features = raw.get("features", {})
        self.tfidf_max_features: int = features.get("tfidf_max_features", 5000)
        self.tfidf_ngram_range: tuple = tuple(
            features.get("tfidf_ngram_range", [1, 2])
        )

        # Model
        model = raw.get("model", {})
        self.model_type: str = model.get("type", "logistic_regression")
        self.test_size: float = model.get("test_size", 0.2)
        self.random_state: int = model.get("random_state", 42)

        # Logging
        self.logging_config: Path = PROJECT_ROOT / raw.get(
            "logging_config", "logging_config.yaml"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Config(model_type={self.model_type!r}, "
            f"tfidf_max_features={self.tfidf_max_features})"
        )
