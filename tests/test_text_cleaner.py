"""Unit tests for src.preprocessing.text_cleaner."""

import pytest
from src.preprocessing.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner(remove_stopwords=False, lemmatize=False, stem=False)


class TestTextCleaner:
    def test_lower_case(self, cleaner):
        assert cleaner.clean("Hello World") == "hello world"

    def test_remove_url(self, cleaner):
        result = cleaner.clean("Visit https://example.com for more.")
        assert "https" not in result
        assert "example" not in result

    def test_remove_html(self, cleaner):
        result = cleaner.clean("<b>Great</b> product")
        assert "<b>" not in result
        assert "great" in result

    def test_remove_punctuation(self, cleaner):
        result = cleaner.clean("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_remove_digits(self):
        cleaner = TextCleaner(remove_stopwords=False, lemmatize=False, remove_digits=True)
        result = cleaner.clean("I have 2 cats and 3 dogs")
        assert "2" not in result
        assert "3" not in result

    def test_keep_digits(self):
        cleaner = TextCleaner(remove_stopwords=False, lemmatize=False, remove_digits=False)
        result = cleaner.clean("Model X500")
        assert "500" in result

    def test_stopword_removal(self):
        cleaner = TextCleaner(remove_stopwords=True, lemmatize=False)
        tokens = cleaner.tokenize("this is a very good product")
        # common stop words like "this", "is", "a" should be removed
        assert "this" not in tokens
        assert "is" not in tokens

    def test_empty_string(self, cleaner):
        assert cleaner.clean("") == ""

    def test_clean_series(self, cleaner):
        import pandas as pd
        s = pd.Series(["Hello World", "Foo Bar"])
        result = cleaner.clean_series(s)
        assert result[0] == "hello world"
        assert result[1] == "foo bar"
