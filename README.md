# Sentiment Analysis Project

> Predict whether a customer review (Amazon, Yelp or Twitter) is **positive** or **negative** using a complete NLP pipeline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Running Tests](#running-tests)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This project provides a modular, production-ready NLP pipeline for binary sentiment classification.

Key features:
- **Data loading** – CSV and JSON support for Amazon, Yelp and Twitter datasets.
- **Text cleaning** – lower-casing, URL/HTML removal, stop-word filtering, lemmatisation.
- **Feature extraction** – TF-IDF (default) and average Word2Vec embeddings.
- **Models** – Logistic Regression, Naïve Bayes, SVM, Random Forest (all via scikit-learn).
- **Centralised configuration** via `config.yaml`.
- **Structured logging** via `logging_config.yaml`.

---

## Project Structure

```
sentiment-analysis-project/
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration loader
│   ├── data/
│   │   └── data_loader.py      # DataLoader class
│   ├── preprocessing/
│   │   └── text_cleaner.py     # TextCleaner class
│   ├── features/
│   │   └── feature_extractor.py# FeatureExtractor class
│   ├── models/
│   │   └── sentiment_model.py  # SentimentModel class
│   └── utils/
│       └── logger.py           # Logging setup
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks for exploration
├── data/
│   ├── raw/                    # Raw datasets (not committed)
│   └── processed/              # Processed datasets (not committed)
├── models/                     # Saved models (not committed)
├── logs/                       # Log files (not committed)
├── config.yaml                 # Main configuration
├── logging_config.yaml         # Logging configuration
├── requirements.txt
├── setup.py
├── CONTRIBUTING.md
└── LICENSE
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/NabilDia/sentiment-analysis-project.git
cd sentiment-analysis-project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (required for stop-words and lemmatisation)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

---

## Quick Start

```python
from src.utils.logger import setup_logging
from src.data.data_loader import DataLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.features.feature_extractor import FeatureExtractor
from src.models.sentiment_model import SentimentModel

setup_logging()

# 1. Load data
loader = DataLoader(data_dir="data/raw")
df = loader.load_csv("reviews.csv", text_col="review_body", label_col="star_rating")
train_df, test_df = loader.split(df)

# 2. Clean text
cleaner = TextCleaner()
train_df["text"] = cleaner.clean_series(train_df["text"])
test_df["text"]  = cleaner.clean_series(test_df["text"])

# 3. Extract features
fe = FeatureExtractor(method="tfidf", max_features=5000)
X_train = fe.fit_transform(train_df["text"].tolist())
X_test  = fe.transform(test_df["text"].tolist())

# 4. Train model
model = SentimentModel(model_type="logistic_regression")
model.train(X_train, train_df["label"])

# 5. Evaluate
metrics = model.evaluate(X_test, test_df["label"])
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"F1 Score : {metrics['f1_weighted']:.4f}")
print(metrics["classification_report"])

# 6. Save artefacts
fe.save("models/feature_extractor.pkl")
model.save("models/sentiment_model.pkl")
```

---

## Configuration

All settings live in `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `preprocessing.language` | `english` | Language for stop-word removal |
| `preprocessing.remove_stopwords` | `true` | Remove stop words |
| `preprocessing.lemmatize` | `true` | Apply lemmatisation |
| `features.method` | `tfidf` | `tfidf` or `word2vec` |
| `features.tfidf_max_features` | `5000` | Maximum vocabulary size |
| `model.type` | `logistic_regression` | Classifier to use |
| `model.test_size` | `0.2` | Fraction of data for testing |
| `model.random_state` | `42` | Random seed |

---

## Running Tests

```bash
pytest tests/ -v
# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.
