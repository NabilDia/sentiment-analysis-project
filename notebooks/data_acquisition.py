import pandas as pd
import os
from datasets import load_dataset

# Création du dossier data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

print(" Chargement des données en cours...")

try:
    # Chargement du dataset IMDB (fonctionne bien)
    dataset = load_dataset("imdb", split="train")
    
    # Conversion en DataFrame pandas
    df = pd.DataFrame(dataset)
    
    # Échantillonnage de 5000 lignes
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    
    # Affichage des infos
    print(f" Dataset chargé avec succès !")
    print(f" Colonnes : {df_sample.columns.tolist()}")
    print(f" Nombre de lignes : {len(df_sample)}")
    
    # Sauvegarde en CSV
    output_path = os.path.join(data_dir, "reviews_sample.csv")
    df_sample.to_csv(output_path, index=False)
    
    print(f" Données sauvegardées dans : {output_path}")
    
except Exception as e:
    print(f" Erreur : {e}")




# df.to_excel('reviews_formaté.xlsx', index=False)