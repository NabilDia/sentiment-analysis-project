import os
import joblib
import sys

# Importation de notre fonction de nettoyage (étape 2)
try:
    from preprocessing import clean_text
except ImportError:
    print(" Erreur : Impossible de trouver le fichier preprocessing.py")
    sys.exit(1)

print(" Chargement du modèle d'Intelligence Artificielle...")

# --- 1. Définition des chemins ---
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '..', 'models', 'sentiment_model.pkl')
vectorizer_path = os.path.join(base_dir, '..', 'data', 'processed', 'tfidf_vectorizer.pkl')

# --- 2. Chargement des composants ---
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(" Modèle chargé avec succès !\n")
except FileNotFoundError as e:
    print(f" Erreur de chargement : {e}")
    sys.exit(1)

print("=" * 50)
print(" TESTEUR DE SENTIMENTS IMDB")
print("Tapez une phrase en anglais pour tester l'IA.")
print("Tapez 'quit' pour arrêter le programme.")
print("=" * 50)

# --- 3. Boucle de prédiction interactive ---
while True:
    # Demander une phrase à l'utilisateur
    user_input = input("\n Entrez une critique : ")
    
    # Condition d'arrêt
    if user_input.lower().strip() == 'quit':
        print(" Au revoir !")
        break
        
    # Ignorer si l'entrée est vide
    if not user_input.strip():
        continue

    # A. Nettoyage (exactement le même que pour l'entraînement)
    cleaned_input = clean_text(user_input)
    
    # B. Vectorisation (transformation des mots en nombres)
    vectorized_input = vectorizer.transform([cleaned_input])
    
    # C. Prédiction
    prediction = model.predict(vectorized_input)[0]       # Le résultat (0 ou 1)
    probabilities = model.predict_proba(vectorized_input)[0] # Le pourcentage de certitude
    
    # D. Affichage du résultat
    if prediction == 1:
        confidence = probabilities[1] * 100
        print(f" Résultat : POSITIF (Certitude : {confidence:.1f}%)")
    else:
        confidence = probabilities[0] * 100
        print(f" Résultat : NÉGATIF (Certitude : {confidence:.1f}%)")