import inspect
if not hasattr(inspect, 'formatargspec'):
    # On crée une fonction factice pour éviter que la librairie crashe
    inspect.formatargspec = lambda *args, **kwargs: ""

import streamlit as st
import joblib
import os

# 2. On importe la fonction en précisant le dossier 'notebooks'
from notebooks.text_preprocessing import clean_text

# --- 1. Configuration de la page ---
st.set_page_config(page_title="Sentiment AI", page_icon="🎬", layout="centered")

# --- 2. Chargement du modèle (avec cache pour la rapidité) ---
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'models', 'sentiment_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'data', 'processed', 'tfidf_vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

try:
    model, vectorizer = load_models()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# --- 3. Interface Graphique ---
st.title(" Analyseur de Sentiments (IA)")
st.write("Cette Intelligence Artificielle lit une critique de film et devine si elle est **Positive** ou **Négative**.")

# Zone de texte pour l'utilisateur
user_input = st.text_area(
    " Entrez votre critique (en anglais) :", 
    height=150, 
    placeholder="Exemple: This movie was absolutely fantastic and I loved the characters!"
)

# Bouton de validation
if st.button(" Analyser le texte", use_container_width=True):
    if user_input.strip() == "":
        st.warning(" Veuillez entrer du texte avant d'analyser.")
    else:
        with st.spinner('L\'IA réfléchit...'):
            # A. Nettoyage
            cleaned_input = clean_text(user_input)
            
            # B. Vectorisation
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # C. Prédiction
            prediction = model.predict(vectorized_input)[0]
            probabilities = model.predict_proba(vectorized_input)[0]
            
            # D. Affichage du résultat avec un visuel sympa
            st.markdown("---")
            if prediction == 1:
                confidence = probabilities[1] * 100
                st.success(f"### Résultat : POSITIF")
                st.info(f"L'IA est sûre à **{confidence:.1f}%** de son choix.")
                st.balloons() # Petite animation Streamlit
            else:
                confidence = probabilities[0] * 100
                st.error(f"### Résultat : NÉGATIF")
                st.warning(f"L'IA est sûre à **{confidence:.1f}%** de son choix.")