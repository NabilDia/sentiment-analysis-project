"""
TextCleaner – clean and normalise raw review text.

Steps performed (all toggleable via constructor arguments):
1. Lower-case
2. Remove URLs
3. Remove HTML tags
4. Remove punctuation and special characters
5. Remove digits
6. Tokenise
7. Remove stop-words
8. Lemmatise (spaCy) *or* stem (NLTK PorterStemmer)
"""

import re
import logging
import string
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for optional NLP back-ends
# ---------------------------------------------------------------------------

def _get_nltk_stopwords(language: str = "english"):
    import nltk
    try:
        from nltk.corpus import stopwords as sw
        return set(sw.words(language))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import stopwords as sw
        return set(sw.words(language))


def _get_nltk_stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def _get_nltk_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer()
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer()


class TextCleaner:
    """Pre-process raw text for downstream NLP tasks."""

    def __init__(
        self,
        language: str = "english",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        remove_digits: bool = True,
    ):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        self.remove_digits = remove_digits

        self._stopwords: Optional[set] = None
        self._stemmer = None
        self._lemmatizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Return cleaned text as a single string."""
        tokens = self.tokenize(text)
        return " ".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Return a list of cleaned tokens."""
        text = self._to_lower(text)
        text = self._remove_urls(text)
        text = self._remove_html(text)
        text = self._remove_punctuation(text)
        if self.remove_digits:
            text = self._remove_digits_fn(text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = self._filter_stopwords(tokens)
        if self.lemmatize:
            tokens = self._lemmatize(tokens)
        elif self.stem:
            tokens = self._stem(tokens)
        return [t for t in tokens if t]

    def clean_series(self, series) -> "pd.Series":  # noqa: F821
        """Vectorised cleaning of a pandas Series."""
        return series.astype(str).map(self.clean)

    # ------------------------------------------------------------------
    # Private text-transformation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        return text.translate(translator)

    @staticmethod
    def _remove_digits_fn(text: str) -> str:
        return re.sub(r"\d+", " ", text)

    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        if self._stopwords is None:
            self._stopwords = _get_nltk_stopwords(self.language)
        return [t for t in tokens if t not in self._stopwords]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        if self._lemmatizer is None:
            self._lemmatizer = _get_nltk_lemmatizer()
        return [self._lemmatizer.lemmatize(t) for t in tokens]

    def _stem(self, tokens: List[str]) -> List[str]:
        if self._stemmer is None:
            self._stemmer = _get_nltk_stemmer()
        return [self._stemmer.stem(t) for t in tokens]
