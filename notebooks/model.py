import os
import pandas as pd
import scipy.sparse
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(" Démarrage de l'entraînement du modèle (Machine Learning)...")

# --- 1. Définition des chemins ---
processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# --- 2. Chargement des données vectorisées ---
print(" Chargement des données d'entraînement et de test...")
X_train = scipy.sparse.load_npz(os.path.join(processed_dir, 'X_train_tfidf.npz'))
X_test = scipy.sparse.load_npz(os.path.join(processed_dir, 'X_test_tfidf.npz'))

y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()

# --- 3. Entraînement du modèle ---
print(" Entraînement du modèle (Régression Logistique)...")
# On choisit la Régression Logistique car elle est très efficace pour l'analyse de texte
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 4. Évaluation du modèle ---
print(" Évaluation du modèle sur les données de test...")
y_pred = model.predict(X_test)

# Calcul de la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Précision globale (Accuracy) : {accuracy * 100:.2f} %")

# Rapport détaillé
print("\n Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Négatif (0)', 'Positif (1)']))

# Matrice de confusion (pour voir où le modèle se trompe)
print("\n Génération de la matrice de confusion...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Négatif', 'Positif'], 
            yticklabels=['Négatif', 'Positif'])
plt.ylabel('Vrai Sentiment')
plt.xlabel('Sentiment Prédit par l\'IA')
plt.title('Matrice de Confusion')

# Sauvegarde de l'image
viz_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
plt.savefig(os.path.join(viz_dir, 'matrice_confusion.png'))
plt.close()

# --- 5. Sauvegarde du modèle ---
print(" Sauvegarde du modèle entraîné...")
model_path = os.path.join(models_dir, 'sentiment_model.pkl')
joblib.dump(model, model_path)
print(f" Modèle sauvegardé ici : {model_path}")
print(" L'entraînement est terminé !")