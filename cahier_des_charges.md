# 📋 Cahier des Charges - Projet d'Analyse de Sentiments

Ce document détaille, étape par étape, le processus complet du projet d'analyse de sentiments. Il sert de feuille de route pour le développement.

---

## ✅ ÉTAPE 1 : Acquisition et Exploration (Terminée)
* **Objectif :** Obtenir les données et comprendre leur structure.
* **Actions réalisées :** 
  * Téléchargement de 5000 critiques (dataset IMDB) via `data_acquisition.py`.
  * Sauvegarde en formats CSV et Excel dans le dossier `data/`.
  * Vérification de l'équilibre des classes (Positif/Négatif) via `analyse_sentiments.py`.
  * Analyse de la distribution de la longueur des textes via `analyse.py`.
* **Conclusion :** Données prêtes et équilibrées. Le texte est brut et nécessite un nettoyage.

---

## 🚧 ÉTAPE 2 : Prétraitement du Texte / NLP (En cours)
* **Objectif :** Nettoyer le texte pour que la machine puisse se concentrer uniquement sur les mots porteurs de sens.
* **Actions à réaliser :**
  1. Passage en minuscules (minuscule textuelle).
  2. Suppression de la ponctuation, des balises HTML (ex: `<br />`) et des caractères spéciaux.
  3. Suppression des **Stop words** (mots vides comme "the", "and", "is" qui n'apportent aucun sentiment).
  4. Application de la **Lemmatisation** ou du **Stemming** (réduire les mots à leur racine, ex: "running" -> "run").
* **Livrables :**
  * Script de nettoyage (`notebooks/preprocessing.py` ou `src/preprocessing.py`).
  * Un nouveau fichier de données nettoyées (ex: `reviews_cleaned.csv`).

---

## ⏳ ÉTAPE 3 : Vectorisation (Transformation en Nombres)
* **Objectif :** Transformer le texte nettoyé en vecteurs numériques compréhensibles par un algorithme d'apprentissage automatique.
* **Actions à réaliser :**
  * Séparer les données en ensembles d'entraînement (Train) et de test (Test).
  * Appliquer la méthode **TF-IDF** (Term Frequency-Inverse Document Frequency) ou un **Word Embedding** (Word2Vec) pour pondérer l'importance des mots.
* **Livrable :** Script/Notebook de vectorisation (`notebooks/vectorisation.py`).

---

## ⏳ ÉTAPE 4 : Modélisation et Apprentissage Automatique
* **Objectif :** Entraîner une Intelligence Artificielle à prédire le sentiment d'un texte.
* **Actions à réaliser :**
  1. Entraîner un modèle de Machine Learning basique (Régression Logistique, Naive Bayes).
  2. Entraîner un modèle plus complexe (Random Forest ou SVM) pour comparer.
  3. Faire des prédictions sur l'ensemble de test.
* **Livrable :** Script d'entraînement (`notebooks/model.py` ou `src/model.py`).

---

## ⏳ ÉTAPE 5 : Évaluation et Amélioration
* **Objectif :** Mesurer les performances du modèle et tirer des conclusions.
* **Actions à réaliser :**
  * Calculer l'**Accuracy** (précision globale) et le **F1-Score**.
  * Générer une **Matrice de confusion** pour voir si le modèle se trompe plus sur les critiques positives ou négatives.
  * Tester le modèle avec vos propres phrases inventées.
* **Livrable :** Rapport de performances et graphiques finaux.