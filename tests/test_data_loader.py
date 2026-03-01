"""Unit tests for src.data.data_loader."""

import json
import pytest
import pandas as pd

from src.data.data_loader import DataLoader


@pytest.fixture
def tmp_csv(tmp_path):
    df = pd.DataFrame({"text": ["good", "bad", "ok"], "label": [1, 0, 1]})
    p = tmp_path / "reviews.csv"
    df.to_csv(p, index=False)
    return tmp_path, "reviews.csv"


@pytest.fixture
def tmp_json(tmp_path):
    data = [{"text": "good", "label": 1}, {"text": "bad", "label": 0}]
    p = tmp_path / "reviews.json"
    p.write_text(json.dumps(data))
    return tmp_path, "reviews.json"


class TestDataLoader:
    def test_load_csv(self, tmp_csv):
        data_dir, filename = tmp_csv
        loader = DataLoader(data_dir=data_dir)
        df = loader.load_csv(filename)
        assert list(df.columns) == ["text", "label"]
        assert len(df) == 3

    def test_load_json(self, tmp_json):
        data_dir, filename = tmp_json
        loader = DataLoader(data_dir=data_dir)
        df = loader.load_json(filename)
        assert list(df.columns) == ["text", "label"]
        assert len(df) == 2

    def test_missing_column_raises(self, tmp_csv):
        data_dir, filename = tmp_csv
        loader = DataLoader(data_dir=data_dir)
        with pytest.raises(ValueError):
            loader.load_csv(filename, text_col="nonexistent")

    def test_null_rows_dropped(self, tmp_path):
        df = pd.DataFrame({"text": ["good", None, "ok"], "label": [1, 0, None]})
        p = tmp_path / "nulls.csv"
        df.to_csv(p, index=False)
        loader = DataLoader(data_dir=tmp_path)
        result = loader.load_csv("nulls.csv")
        assert len(result) == 1  # only "good"/1 survives

    def test_split(self, tmp_csv):
        data_dir, filename = tmp_csv
        loader = DataLoader(data_dir=data_dir)
        df = loader.load_csv(filename)
        # Need enough samples for stratified split – add more rows
        big = pd.concat([df] * 10, ignore_index=True)
        train, test = loader.split(big, test_size=0.2)
        assert len(train) + len(test) == len(big)
