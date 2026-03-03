# Projet d'Analyse de Sentiments

> Prédire si un avis client (Amazon, Yelp ou Twitter) est **positif** ou **négatif** grâce à un pipeline NLP complet.

[![Licence: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## Table des matières
1. [Vue d'ensemble](#vue-densemble)
2. [Structure du projet](#structure-du-projet)
3. [Description des modules](#description-des-modules)
4. [Installation](#installation)
5. [Démarrage rapide](#démarrage-rapide)
6. [Configuration](#configuration)
7. [Lancer les tests](#lancer-les-tests)
8. [Contribuer](#contribuer)
9. [Licence](#licence)

---

## Vue d'ensemble

Ce projet fournit un pipeline NLP modulaire et organisé pour la **classification binaire de sentiments** (positif / négatif).

Fonctionnalités principales :
- **Chargement des données** – lecture de fichiers CSV et JSON (Amazon, Yelp, Twitter).
- **Nettoyage du texte** – mise en minuscules, suppression des URLs, balises HTML, ponctuation, mots vides (*stop words*), lemmatisation.
- **Extraction de features** – TF-IDF (par défaut) ou embeddings Word2Vec moyennés.
- **Modèles ML** – Régression Logistique, Naïve Bayes, SVM, Forêt Aléatoire (tous via scikit-learn).
- **Configuration centralisée** dans `config.yaml` (un seul endroit pour changer les paramètres).
- **Journalisation structurée** via `logging_config.yaml`.

---

## Structure du projet

```
sentiment-analysis-project/
│
├── src/                          ← Code source principal
│   ├── __init__.py               ← Déclare src comme un package Python
│   ├── config.py                 ← Lit config.yaml et expose un objet Config
│   ├── data/
│   │   └── data_loader.py        ← Charge et valide les datasets (CSV/JSON)
│   ├── preprocessing/
│   │   └── text_cleaner.py       ← Nettoie et normalise le texte brut
│   ├── features/
│   │   └── feature_extractor.py  ← Convertit le texte en vecteurs numériques
│   ├── models/
│   │   └── sentiment_model.py    ← Entraîne, évalue et sauvegarde un classifieur
│   └── utils/
│       └── logger.py             ← Configure le système de logs
│
├── tests/                        ← Tests unitaires (pytest)
│   ├── test_data_loader.py
│   ├── test_text_cleaner.py
│   ├── test_feature_extractor.py
│   └── test_sentiment_model.py
│
├── notebooks/                    ← Notebooks Jupyter pour l'exploration
├── data/
│   ├── raw/                      ← Données brutes (non commitées)
│   └── processed/                ← Données nettoyées (non commitées)
├── models/                       ← Modèles sauvegardés (non commités)
├── logs/                         ← Fichiers de log (non commités)
│
├── config.yaml                   ← Paramètres du projet (à modifier ici)
├── logging_config.yaml           ← Configuration des logs
├── requirements.txt              ← Dépendances Python
├── setup.py                      ← Installation du package
├── CONTRIBUTING.md               ← Guide de contribution
└── LICENSE                       ← Licence MIT
```

---

## Description des modules

### `src/config.py` – Configuration centralisée
Lit `config.yaml` et expose un objet `Config` avec tous les paramètres du projet (chemins, prétraitement, features, modèle). Toutes les autres classes peuvent l'utiliser pour éviter les valeurs en dur dans le code.

### `src/data/data_loader.py` – Chargement des données
La classe `DataLoader` charge un fichier CSV ou JSON et retourne un `DataFrame` pandas avec deux colonnes standardisées : `text` (le texte de l'avis) et `label` (la note/sentiment). Des raccourcis existent pour Amazon (`load_amazon`), Yelp (`load_yelp`) et Twitter (`load_twitter`).

### `src/preprocessing/text_cleaner.py` – Nettoyage du texte
La classe `TextCleaner` applique les étapes suivantes (toutes optionnelles) :
1. Mise en minuscules
2. Suppression des URLs
3. Suppression des balises HTML
4. Suppression de la ponctuation
5. Suppression des chiffres
6. Suppression des *stop words* (mots courants sans sens : "le", "the", "is"…)
7. **Lemmatisation** : ramène chaque mot à sa forme de base ("running" → "run")

### `src/features/feature_extractor.py` – Extraction de features
La classe `FeatureExtractor` transforme le texte nettoyé en vecteurs numériques compréhensibles par un modèle ML :
- **TF-IDF** *(par défaut)* : représente chaque document par l'importance relative de ses mots dans le corpus.
- **Word2Vec** *(optionnel, nécessite gensim)* : représente chaque document par la moyenne des vecteurs de ses mots.

### `src/models/sentiment_model.py` – Modèle de classification
La classe `SentimentModel` encapsule un classifieur scikit-learn. Classifieurs disponibles :
| `model_type` | Description |
|---|---|
| `logistic_regression` | Régression logistique (recommandé pour débuter) |
| `naive_bayes` | Naïve Bayes multinomial |
| `bernoulli_nb` | Naïve Bayes de Bernoulli |
| `svm` | Machine à vecteurs de support linéaire |
| `random_forest` | Forêt aléatoire |

### `src/utils/logger.py` – Journalisation
La fonction `setup_logging()` configure les logs à partir de `logging_config.yaml`. Les logs s'affichent dans le terminal **et** sont écrits dans `logs/app.log`.

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/NabilDia/sentiment-analysis-project.git
cd sentiment-analysis-project

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données NLTK (stop words, lemmatisation)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

---

## Démarrage rapide

Voici un exemple complet du pipeline, de la donnée brute à l'évaluation :

```python
from src.utils.logger import setup_logging
from src.data.data_loader import DataLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.features.feature_extractor import FeatureExtractor
from src.models.sentiment_model import SentimentModel

# Activer les logs
setup_logging()

# 1. Charger les données
loader = DataLoader(data_dir="data/raw")
df = loader.load_csv("avis.csv", text_col="texte_avis", label_col="note")
train_df, test_df = loader.split(df)  # 80 % entraînement, 20 % test

# 2. Nettoyer le texte
cleaner = TextCleaner()
train_df["text"] = cleaner.clean_series(train_df["text"])
test_df["text"]  = cleaner.clean_series(test_df["text"])

# 3. Extraire les features (TF-IDF)
fe = FeatureExtractor(method="tfidf", max_features=5000)
X_train = fe.fit_transform(train_df["text"].tolist())  # apprend le vocabulaire
X_test  = fe.transform(test_df["text"].tolist())       # applique le vocabulaire appris

# 4. Entraîner le modèle
model = SentimentModel(model_type="logistic_regression")
model.train(X_train, train_df["label"])

# 5. Évaluer
metrics = model.evaluate(X_test, test_df["label"])
print(f"Précision : {metrics['accuracy']:.4f}")
print(f"Score F1  : {metrics['f1_weighted']:.4f}")
print(metrics["classification_report"])

# 6. Sauvegarder pour réutilisation
fe.save("models/feature_extractor.pkl")
model.save("models/sentiment_model.pkl")
```

---

## Configuration

Tous les paramètres se trouvent dans `config.yaml`. Il suffit de modifier ce fichier pour changer le comportement du pipeline, **sans toucher au code**.

| Clé | Valeur par défaut | Description |
|-----|-------------------|-------------|
| `preprocessing.language` | `english` | Langue pour la suppression des stop words |
| `preprocessing.remove_stopwords` | `true` | Activer la suppression des stop words |
| `preprocessing.lemmatize` | `true` | Activer la lemmatisation |
| `features.method` | `tfidf` | Méthode de vectorisation : `tfidf` ou `word2vec` |
| `features.tfidf_max_features` | `5000` | Taille maximale du vocabulaire TF-IDF |
| `model.type` | `logistic_regression` | Classifieur à utiliser |
| `model.test_size` | `0.2` | Part des données réservée au test (20 %) |
| `model.random_state` | `42` | Graine aléatoire pour la reproductibilité |

---

## Lancer les tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Avec rapport de couverture de code
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les consignes de contribution.

---

## Licence

Ce projet est distribué sous licence MIT – voir [LICENSE](LICENSE) pour plus de détails.
