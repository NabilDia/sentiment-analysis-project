# Changelog

## [0.1.0] – Structure initiale du projet

Ce PR a créé l'**architecture complète** du projet à partir d'un dépôt presque vide (qui ne contenait que `README.md` et `requirements.txt`).

### Ce qui a été ajouté et pourquoi

#### 📁 Dossiers créés
| Dossier | Rôle |
|---|---|
| `src/` | Contient tout le code Python du projet |
| `src/data/` | Module de chargement des données |
| `src/preprocessing/` | Module de nettoyage du texte |
| `src/features/` | Module d'extraction de features |
| `src/models/` | Module d'entraînement et d'évaluation |
| `src/utils/` | Utilitaires (logs, etc.) |
| `tests/` | Tests unitaires automatisés |
| `notebooks/` | Notebooks Jupyter pour l'exploration |
| `data/raw/` | Données brutes (non commitées dans Git) |
| `data/processed/` | Données nettoyées (non commitées) |
| `models/` | Modèles sauvegardés (non commités) |
| `logs/` | Fichiers de log (non commités) |
| `config/` | Répertoire pour les configs supplémentaires |

#### 🐍 Fichiers Python créés

**`src/config.py`**
Lit le fichier `config.yaml` et expose tous les paramètres du projet dans un objet `Config`. Cela évite d'avoir des valeurs en dur dans le code.

**`src/data/data_loader.py`**
Classe `DataLoader` : charge un fichier CSV ou JSON et retourne un `DataFrame` pandas avec les colonnes `text` et `label`. Gère les cas particuliers d'Amazon, Yelp et Twitter automatiquement.

**`src/preprocessing/text_cleaner.py`**
Classe `TextCleaner` : nettoie le texte brut (minuscules, suppression URLs/HTML/ponctuation, stop words, lemmatisation). Chaque étape peut être activée ou désactivée.

**`src/features/feature_extractor.py`**
Classe `FeatureExtractor` : transforme le texte nettoyé en vecteurs numériques via TF-IDF ou Word2Vec, pour que le modèle ML puisse l'utiliser.

**`src/models/sentiment_model.py`**
Classe `SentimentModel` : entraîne, évalue et sauvegarde un classifieur scikit-learn (Régression Logistique, SVM, etc.). Retourne précision, F1 et un rapport détaillé.

**`src/utils/logger.py`**
Fonction `setup_logging()` : configure les logs selon `logging_config.yaml` (affichage terminal + fichier `logs/app.log`).

**`tests/test_*.py`**
25 tests unitaires couvrant tous les modules. Se lancent avec `pytest tests/ -v`.

#### ⚙️ Fichiers de configuration

**`config.yaml`** – Paramètres centraux du projet (langue, méthode de vectorisation, type de modèle, etc.). C'est ici que l'on change les options sans modifier le code.

**`logging_config.yaml`** – Définit le format des logs, les niveaux (DEBUG, INFO…) et la rotation des fichiers de log.

#### 📄 Autres fichiers

**`.gitignore`** – Dit à Git d'ignorer les fichiers inutiles : environnements virtuels, caches Python, données brutes, modèles binaires, logs.

**`setup.py`** – Permet d'installer le projet comme un package Python (`pip install -e .`).

**`requirements.txt`** – Liste des bibliothèques nécessaires (pandas, scikit-learn, nltk, etc.).

**`LICENSE`** – Licence MIT (libre d'utilisation).

**`CONTRIBUTING.md`** – Guide pour contribuer au projet.
