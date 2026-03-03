# Contribuer au projet d'Analyse de Sentiments

Merci de vouloir contribuer à ce projet ! 🎉

## Comment contribuer

### Signaler un bug
1. Vérifiez d'abord les [issues existantes](../../issues).
2. Ouvrez une nouvelle issue avec un titre clair et une description précise.
3. Fournissez un exemple minimal pour reproduire le problème si possible.

### Proposer une amélioration
1. Ouvrez une issue avec le label **enhancement**.
2. Décrivez le cas d'usage et le comportement attendu.

### Soumettre une Pull Request
1. Faites un fork du dépôt et créez votre branche depuis `main` :
   ```bash
   git checkout -b feature/ma-nouvelle-fonctionnalite
   ```
2. Installez les dépendances de développement :
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```
3. Apportez vos modifications et écrivez des tests.
4. Lancez les tests :
   ```bash
   pytest tests/ -v
   ```
5. Vérifiez que tous les tests passent avant de soumettre.
6. Poussez sur votre fork et ouvrez une Pull Request.

## Style de code
* Respectez [PEP 8](https://pep8.org/).
* Utilisez des noms de variables explicites.
* Ajoutez des docstrings à toutes les classes et méthodes publiques.
* Gardez les fonctions courtes et ciblées.

## Messages de commit
Utilisez le style de commit conventionnel :
```
feat: ajout du support Word2Vec
fix: gestion des chaînes vides dans TextCleaner
docs: mise à jour du README avec les étapes d'installation
test: ajout de tests unitaires pour DataLoader
```

## Structure du projet
```
sentiment-analysis-project/
├── src/
│   ├── config.py           # Configuration centrale
│   ├── data/               # Chargement des données
│   ├── preprocessing/      # Nettoyage du texte
│   ├── features/           # Extraction de features
│   ├── models/             # Modèles ML
│   └── utils/              # Utilitaires (logs, …)
├── tests/                  # Tests unitaires
├── notebooks/              # Notebooks d'exploration
├── data/                   # Données brutes et traitées
├── models/                 # Modèles sauvegardés
├── config.yaml             # Configuration principale
└── logging_config.yaml     # Configuration des logs
```

## Questions ?
Ouvrez une issue ou démarrez une discussion. Nous sommes là pour aider !
