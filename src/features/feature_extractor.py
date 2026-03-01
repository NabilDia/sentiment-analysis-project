"""
FeatureExtractor – convert text into numerical feature matrices.

Supported methods
-----------------
* TF-IDF  (default) via scikit-learn's TfidfVectorizer
* Average Word2Vec  via gensim (optional)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Transform cleaned text into feature vectors for ML models."""

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        embedding_dim: int = 100,
    ):
        if method not in ("tfidf", "word2vec"):
            raise ValueError(f"Unknown method '{method}'. Choose 'tfidf' or 'word2vec'.")
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.embedding_dim = embedding_dim

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._word2vec_model = None  # lazy-loaded gensim model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit the feature extractor on *texts*."""
        if self.method == "tfidf":
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            self._vectorizer.fit(texts)
            logger.info(
                "TF-IDF fitted: vocab_size=%d", len(self._vectorizer.vocabulary_)
            )
        else:
            self._fit_word2vec(texts)
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform *texts* into a feature matrix."""
        if self.method == "tfidf":
            if self._vectorizer is None:
                raise RuntimeError("Call fit() before transform().")
            return self._vectorizer.transform(texts)
        return self._transform_word2vec(texts)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

    def save(self, path: Path) -> None:
        """Persist the fitted extractor to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("FeatureExtractor saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureExtractor":
        """Load a previously saved extractor from *path*."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        logger.info("FeatureExtractor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Word2Vec helpers
    # ------------------------------------------------------------------

    def _fit_word2vec(self, texts: List[str]) -> None:
        try:
            from gensim.models import Word2Vec
        except ImportError as exc:
            raise ImportError(
                "gensim is required for word2vec embeddings. "
                "Install it with: pip install gensim"
            ) from exc
        tokenized = [t.split() for t in texts]
        self._word2vec_model = Word2Vec(
            sentences=tokenized,
            vector_size=self.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
        )
        logger.info("Word2Vec fitted: vocab_size=%d", len(self._word2vec_model.wv))

    def _transform_word2vec(self, texts: List[str]) -> np.ndarray:
        if self._word2vec_model is None:
            raise RuntimeError("Call fit() before transform().")
        wv = self._word2vec_model.wv
        vectors = []
        for text in texts:
            tokens = [t for t in text.split() if t in wv]
            if tokens:
                vectors.append(np.mean([wv[t] for t in tokens], axis=0))
            else:
                vectors.append(np.zeros(self.embedding_dim))
        return np.vstack(vectors)
