import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_viz_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'visualizations')
os.makedirs(output_viz_dir, exist_ok=True)

# --- Chargement des données ---
excel_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_formate.xlsx')
df = pd.read_excel(excel_path)

print(" Analyse de la répartition des sentiments...")

if 'label' in df.columns:
    # Calcul des effectifs et pourcentages
    counts = df['label'].value_counts()
    percentages = df['label'].value_counts(normalize=True) * 100
    
    # Mapping des labels pour la lisibilité (1 = Positif, 0 = Négatif)
    labels_names = ['Positif' if x == 1 else 'Négatif' for x in counts.index]
    
    print("\n Nombre de critiques par catégorie :")
    for label, count, pct in zip(labels_names, counts, percentages):
        print(f" - {label} : {count} ({pct:.1f}%)")

    # Création dynamique selon le nombre de catégories trouvées
    dynamic_explode = [0.05 if i == 0 else 0 for i in range(len(counts))]
    dynamic_colors = ['#4ECDC4', '#FF6B6B', '#FFD166', '#06D6A0'][:len(counts)]

    # Visualisation : Diagramme circulaire (Pie Chart)
    plt.figure(figsize=(8, 6))
    plt.pie(counts, 
            labels=labels_names, 
            autopct='%1.1f%%', 
            colors=dynamic_colors, 
            startangle=90,
            explode=dynamic_explode) # S'adapte automatiquement
            
    plt.title('Répartition des critiques (Positives vs Négatives)')
    
    # Sauvegarde et fermeture
    save_path = os.path.join(output_viz_dir, 'repartition_sentiments_pie.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"\n Graphique circulaire sauvegardé : {save_path}")
else:
    print(" Erreur : La colonne 'label' est introuvable dans le fichier Excel.")