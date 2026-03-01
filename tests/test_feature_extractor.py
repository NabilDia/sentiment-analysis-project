"""Unit tests for src.features.feature_extractor."""

import numpy as np
import pytest
from src.features.feature_extractor import FeatureExtractor


TEXTS = [
    "great product love it",
    "terrible experience never again",
    "okay product nothing special",
    "amazing quality highly recommend",
]


class TestFeatureExtractorTFIDF:
    def test_fit_transform_shape(self):
        fe = FeatureExtractor(method="tfidf", max_features=50)
        X = fe.fit_transform(TEXTS)
        assert X.shape[0] == len(TEXTS)

    def test_transform_before_fit_raises(self):
        fe = FeatureExtractor(method="tfidf")
        with pytest.raises(RuntimeError):
            fe.transform(TEXTS)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            FeatureExtractor(method="invalid_method")

    def test_save_load(self, tmp_path):
        fe = FeatureExtractor(method="tfidf", max_features=50)
        fe.fit(TEXTS)
        save_path = tmp_path / "extractor.pkl"
        fe.save(save_path)
        loaded = FeatureExtractor.load(save_path)
        X1 = fe.transform(TEXTS)
        X2 = loaded.transform(TEXTS)
        assert np.allclose(X1.toarray(), X2.toarray())
