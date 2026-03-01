# Contributing to Sentiment Analysis Project

Thank you for considering contributing to this project! 🎉

## How to Contribute

### Reporting Bugs
1. Check existing [issues](../../issues) first.
2. Open a new issue with a clear title and description.
3. Include a minimal reproducible example when possible.

### Suggesting Enhancements
1. Open an issue with the label **enhancement**.
2. Describe the use case and expected behaviour.

### Submitting Pull Requests
1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```
3. Make your changes and write tests for them.
4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
5. Ensure all tests pass before submitting.
6. Push to your fork and open a Pull Request.

## Code Style
* Follow [PEP 8](https://pep8.org/).
* Use descriptive variable names.
* Add docstrings to all public classes and methods.
* Keep functions small and focused.

## Commit Messages
Use conventional commit style:
```
feat: add Word2Vec embedding support
fix: handle empty string input in TextCleaner
docs: update README with installation steps
test: add unit tests for DataLoader
```

## Project Structure
```
sentiment-analysis-project/
├── src/
│   ├── config.py           # Central configuration
│   ├── data/               # Data loading
│   ├── preprocessing/      # Text cleaning
│   ├── features/           # Feature extraction
│   ├── models/             # ML models
│   └── utils/              # Utilities (logging, …)
├── tests/                  # Unit tests
├── notebooks/              # Exploration notebooks
├── data/                   # Raw & processed data
├── models/                 # Saved models
├── config.yaml             # Main config
└── logging_config.yaml     # Logging config
```

## Questions?
Open an issue or start a discussion. We are happy to help!
