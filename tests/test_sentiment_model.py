"""Unit tests for src.models.sentiment_model."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.sentiment_model import SentimentModel


def _make_data():
    texts = [
        "great product love it",
        "amazing quality highly recommend",
        "excellent fantastic wonderful",
        "terrible experience never again",
        "awful horrible waste of money",
        "bad product disappointed",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    return X, np.array(labels)


class TestSentimentModel:
    def test_train_predict(self):
        X, y = _make_data()
        model = SentimentModel(model_type="logistic_regression")
        model.train(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_evaluate_keys(self):
        X, y = _make_data()
        model = SentimentModel(model_type="logistic_regression")
        model.train(X, y)
        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError):
            SentimentModel(model_type="unknown_model")

    def test_save_load(self, tmp_path):
        X, y = _make_data()
        model = SentimentModel(model_type="logistic_regression")
        model.train(X, y)
        save_path = tmp_path / "model.pkl"
        model.save(save_path)
        loaded = SentimentModel.load(save_path)
        preds = loaded.predict(X)
        assert len(preds) == len(y)

    @pytest.mark.parametrize("model_type", ["logistic_regression", "svm", "random_forest"])
    def test_all_model_types(self, model_type):
        X, y = _make_data()
        model = SentimentModel(model_type=model_type)
        model.train(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
