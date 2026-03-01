"""
SentimentModel – train and evaluate a scikit-learn based classifier.

Supported classifiers
---------------------
* logistic_regression  (default)
* naive_bayes          (MultinomialNB or BernoulliNB)
* svm                  (LinearSVC)
* random_forest
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

_CLASSIFIERS: Dict[str, Any] = {
    "logistic_regression": lambda rs: LogisticRegression(
        max_iter=1000, random_state=rs
    ),
    "naive_bayes": lambda _: MultinomialNB(),
    "bernoulli_nb": lambda _: BernoulliNB(),
    "svm": lambda rs: LinearSVC(max_iter=2000, random_state=rs),
    "random_forest": lambda rs: RandomForestClassifier(
        n_estimators=100, random_state=rs
    ),
}


class SentimentModel:
    """Wrapper around a scikit-learn classifier for sentiment analysis."""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        random_state: int = 42,
    ):
        if model_type not in _CLASSIFIERS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(_CLASSIFIERS)}"
            )
        self.model_type = model_type
        self.random_state = random_state
        self._clf = _CLASSIFIERS[model_type](random_state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, y_train) -> "SentimentModel":
        """Fit the classifier on (X_train, y_train)."""
        logger.info("Training %s …", self.model_type)
        self._clf.fit(X_train, y_train)
        logger.info("Training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Return class predictions for *X*."""
        return self._clf.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Return class probabilities if the underlying model supports it."""
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X)
        logger.warning("%s does not support predict_proba.", self.model_type)
        return None

    def predict_text(self, texts: List[str], feature_extractor) -> np.ndarray:
        """Convenience helper: transform *texts* then predict."""
        X = feature_extractor.transform(texts)
        return self.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """Return a dictionary with accuracy, F1 and a full report."""
        y_pred = self.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        logger.info(
            "Accuracy: %.4f  |  F1 (weighted): %.4f",
            metrics["accuracy"],
            metrics["f1_weighted"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the trained model to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("SentimentModel saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SentimentModel":
        """Load a previously saved model from *path*."""
        with open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        logger.info("SentimentModel loaded from %s", path)
        return obj
