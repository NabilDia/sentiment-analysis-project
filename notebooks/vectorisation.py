import pandas as pd
import os
import joblib
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print(" Démarrage de la phase de vectorisation...")

# --- 1. Définition des chemins ---
input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_cleaned.csv')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Créer le dossier pour les données transformées s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# --- 2. Chargement des données nettoyées ---
print(" Chargement du dataset nettoyé...")
df = pd.read_csv(input_path)

# Retirer les lignes qui seraient devenues vides après le nettoyage
df = df.dropna(subset=['cleaned_text', 'label'])

# --- 3. Séparation des données (Train / Test) ---
# 80% pour l'entraînement (apprendre), 20% pour le test (évaluer)
print(" Séparation des données en Train (80%) et Test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42 # Pour avoir les mêmes résultats à chaque exécution
)

print(f" Quantité d'entraînement : {len(X_train)} critiques")
print(f" Quantité de test : {len(X_test)} critiques")

# --- 4. Vectorisation TF-IDF ---
print(" Transformation du texte en nombres (TF-IDF)...")
# On limite aux 5000 mots les plus importants pour ne pas surcharger la mémoire
vectorizer = TfidfVectorizer(max_features=5000)

# Apprendre le vocabulaire sur le train ET transformer
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transformer uniquement le test (ne jamais "apprendre" sur le test !)
X_test_tfidf = vectorizer.transform(X_test)

# --- 5. Sauvegarde pour la prochaine étape ---
print(" Sauvegarde des matrices et du modèle de vectorisation...")

# Sauvegarde des cibles (labels)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

# Sauvegarde des matrices de nombres (format compressé npz)
scipy.sparse.save_npz(os.path.join(output_dir, 'X_train_tfidf.npz'), X_train_tfidf)
scipy.sparse.save_npz(os.path.join(output_dir, 'X_test_tfidf.npz'), X_test_tfidf)

# Sauvegarde du "traducteur" pour pouvoir l'utiliser plus tard sur vos propres phrases
joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))

print("\n Terminé ! Les données sont prêtes pour l'Intelligence Artificielle.")
print(f"La matrice d'entraînement ressemble à un tableau de {X_train_tfidf.shape[0]} lignes et {X_train_tfidf.shape[1]} colonnes (mots).")