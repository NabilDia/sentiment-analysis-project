"""
DataLoader – load raw review data from CSV or JSON files.

Supported dataset flavours
--------------------------
* Amazon  – expects columns ``review_body`` and ``star_rating``
* Yelp    – expects columns ``text`` and ``stars``
* Twitter – expects columns ``text`` and ``label``
* Generic – any CSV/JSON with explicit ``text_col`` / ``label_col``
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and perform basic validation of review datasets."""

    # Mapping of known dataset flavours to their text/label columns
    _KNOWN_SOURCES = {
        "amazon": ("review_body", "star_rating"),
        "yelp": ("text", "stars"),
        "twitter": ("text", "label"),
    }

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/raw")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filename: str,
        text_col: str = "text",
        label_col: str = "label",
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """Load a CSV file and return a DataFrame with *text* and *label* columns."""
        filepath = self.data_dir / filename
        logger.info("Loading CSV from %s", filepath)
        df = pd.read_csv(filepath, encoding=encoding)
        return self._normalise(df, text_col, label_col)

    def load_json(
        self,
        filename: str,
        text_col: str = "text",
        label_col: str = "label",
    ) -> pd.DataFrame:
        """Load a JSON-lines or JSON array file."""
        filepath = self.data_dir / filename
        logger.info("Loading JSON from %s", filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        df = pd.DataFrame(raw) if isinstance(raw, list) else pd.json_normalize(raw)
        return self._normalise(df, text_col, label_col)

    def load_amazon(self, filename: str) -> pd.DataFrame:
        """Convenience loader for Amazon review datasets."""
        text_col, label_col = self._KNOWN_SOURCES["amazon"]
        return self.load_csv(filename, text_col=text_col, label_col=label_col)

    def load_yelp(self, filename: str) -> pd.DataFrame:
        """Convenience loader for Yelp review datasets."""
        text_col, label_col = self._KNOWN_SOURCES["yelp"]
        return self.load_csv(filename, text_col=text_col, label_col=label_col)

    def load_twitter(self, filename: str) -> pd.DataFrame:
        """Convenience loader for Twitter sentiment datasets."""
        text_col, label_col = self._KNOWN_SOURCES["twitter"]
        return self.load_csv(filename, text_col=text_col, label_col=label_col)

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train, test) DataFrames."""
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["label"]
        )
        logger.info("Split → train=%d  test=%d", len(train), len(test))
        return train.reset_index(drop=True), test.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
        """Rename columns to canonical *text* / *label* and drop nulls."""
        missing = [c for c in (text_col, label_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in dataset: {missing}")
        df = df.rename(columns={text_col: "text", label_col: "label"})
        before = len(df)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with null text or label", dropped)
        return df[["text", "label"]]
