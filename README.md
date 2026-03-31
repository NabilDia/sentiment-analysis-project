# 📊 Projet : Analyse de Sentiments (NLP & Machine Learning)

## 🎯 Objectif du Projet
Ce projet est un pipeline complet d'**Intelligence Artificielle** (Traitement du Langage Naturel - NLP) dont l'objectif est d'analyser des avis clients pour prédire automatiquement si une critique est **positive** ou **négative**. 

Le modèle a été entraîné sur le jeu de données public **IMDB** (5000 critiques de films en anglais).

## 🚀 Fonctionnalités & Pipeline Technique

Le projet suit rigoureusement les étapes standards d'un projet de Data Science :

1. **Acquisition et Exploration (EDA)**
   - Téléchargement des données via l'API Hugging Face `datasets`.
   - Échantillonnage de 5000 lignes avec un équilibre parfait (50.3% négatif / 49.7% positif).
   - Outils : `pandas`, `matplotlib`, `seaborn`.

2. **Prétraitement du Texte (NLP)**
   - Passage en minuscules et suppression des balises HTML/ponctuation.
   - Suppression des **Stop words** (mots vides anglais).
   - **Lemmatisation** (réduction des mots à leur racine) via la librairie `nltk`.

3. **Vectorisation (Feature Engineering)**
   - Transformation du texte nettoyé en matrices numériques avec **TF-IDF** (Term Frequency-Inverse Document Frequency).
   - Conservation des 5000 mots les plus discriminants.

4. **Modélisation (Machine Learning)**
   - Entraînement d'un modèle de classification **Régression Logistique**.
   - Évaluation sur 20% des données (Test set).

5. **Application Interactive**
   - Script permettant à l'utilisateur de tester ses propres phrases en direct.

## 📈 Résultats et Performances

- **Précision globale (Accuracy) :** **85.60 %**
- Le modèle identifie très bien les textes possédant des marqueurs d'opinion forts (ex: *masterpiece, terrible, disaster, wonderful*).

### ⚠️ Limites rencontrées (Biais du TF-IDF)
Lors des tests finaux, il a été constaté que le modèle peine sur les phrases très courtes ne contenant que du vocabulaire trop commun (ex: *"I like this film"*). C'est une limite classique de la méthode TF-IDF qui calcule ses poids par rapport à la longueur globale des textes de la base d'entraînement (moyenne de 500 à 1000 caractères par critique). Pour un projet futur, une approche par Deep Learning (Transformers/BERT) permettrait de capter le contexte des phrases courtes.

## 📂 Structure du Projet

```text
sentiment-analysis-project/
│
├── data/
│   ├── processed/          # Matrices TF-IDF et modèle vectorizer (.npz, .pkl)
│   ├── visualizations/     # Graphiques générés (Camemberts, Matrice de confusion)
│   ├── reviews_sample.csv  # Données brutes de départ
│   └── reviews_cleaned.csv # Données après nettoyage NLP
│
├── models/
│   └── sentiment_model.pkl # Modèle de Machine Learning entraîné
│
├── notebooks/              # Scripts sources du projet
│   ├── data_acquisition.py # Téléchargement des données
│   ├── analyse.py          # Analyse exploratoire (EDA)
│   ├── preprocessing.py    # Nettoyage NLP NLTK
│   ├── vectorisation.py    # Transformation TF-IDF
│   ├── model.py            # Entraînement du modèle
│   └── prediction.py       # Script interactif pour tester l'IA
│
└── requirements.txt        # Fichier de dépendances Python
```
