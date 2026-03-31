import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. Téléchargement des ressources NLTK nécessaires ---
print(" Téléchargement des ressources linguistiques NLTK...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- 2. Fonction de nettoyage de texte ---
def clean_text(text):
    # Sécurité si le texte est vide
    if not isinstance(text, str):
        return ""
        
    # 1. Mettre en minuscules
    text = text.lower()
    
    # 2. Supprimer les balises HTML (très commun dans IMDB, ex: <br />)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Supprimer la ponctuation et les chiffres (garder uniquement les lettres de a à z)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenisation (séparer en mots)
    words = text.split()
    
    # 5. Supprimer les mots vides (Stop words en anglais)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    # 6. Lemmatisation (ramener les mots à leur racine)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    
    # Reconstruire la phrase
    return ' '.join(words)

# --- 3. Script principal ---
if __name__ == "__main__":
    # Remplacer .xlsx par .csv ici
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_sample.csv')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_cleaned.csv')
    
    print("\n Chargement des données...")
    df = pd.read_csv(input_path) # <-- Changer read_excel en read_csv
    
    if 'text' in df.columns:
        print(" Nettoyage des textes en cours (cela peut prendre quelques secondes)...")
        # Appliquer la fonction de nettoyage à toute la colonne 'text'
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Afficher un avant/après pour la première critique
        print("\n Exemple de nettoyage :")
        print(f"AVANT : {df['text'].iloc[0][:150]}...")
        print(f"APRÈS : {df['cleaned_text'].iloc[0][:150]}...")
        
        # Sauvegarde du nouveau dataset nettoyé
        df.to_csv(output_path, index=False)
        print(f"\n Nouveau dataset nettoyé sauvegardé : {output_path}")
    else:
        print(" Erreur : La colonne 'text' est introuvable.")