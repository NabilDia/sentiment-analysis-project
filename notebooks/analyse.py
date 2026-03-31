import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Style pour les graphiques
sns.set_style("whitegrid")

# Créer un dossier pour les visualisations
base_dir = os.path.dirname(__file__)
output_viz_dir = os.path.join(base_dir, '..', 'data', 'visualizations')
os.makedirs(output_viz_dir, exist_ok=True)

# --- Chargement des données via CSV (Plus stable que l'Excel) ---
csv_path = os.path.join(base_dir, '..', 'data', 'reviews_sample.csv')

print(f" Lecture du fichier CSV : {csv_path}")
df = pd.read_csv(csv_path)

print(f" Colonnes trouvées : {df.columns.tolist()}")

# --- 1. Graphique des sentiments ---
if 'label' in df.columns:
    print("\n Génération du graphique de répartition des sentiments...")
    
    sentiment_counts = df['label'].value_counts()
    print(f"Détail des comptes :\n{sentiment_counts}") # Pour vérifier dans le terminal
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts.values, 
        labels=['Négatif (0)', 'Positif (1)'],
        autopct='%1.1f%%',
        colors=['#FF6B6B', '#4ECDC4'],
        startangle=90,
        explode=(0.05, 0)
    )
    plt.title('Répartition des critiques (Positives vs Négatives)')
    
    chemin_graphique = os.path.join(output_viz_dir, 'repartition_sentiments.png')
    plt.savefig(chemin_graphique)
    print(f" Graphique sauvegardé ici : {chemin_graphique}")
    plt.show()
else:
    print(" La colonne 'label' n'existe pas dans le DataFrame.")
    
print("\n 2. Création du graphique des longueurs de texte...")
if 'text' in df.columns:
    df['text_length'] = df['text'].astype(str).str.len()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['text_length'], bins=50, color='#95E1D3')
    plt.title('Distribution de la Longueur des Critiques')
    plt.xlabel('Nombre de caractères')
    plt.ylabel('Fréquence')
    
    # Sauvegarde et fermeture
    plt.savefig(os.path.join(output_viz_dir, 'text_length_distribution.png'))
    plt.close()
    print(" Graphique 'text_length_distribution.png' sauvegardé ! ")

print("\n Terminé ! Allez vérifier le dossier data/visualizations/")