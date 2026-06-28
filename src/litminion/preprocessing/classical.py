"""
classical.py

Classical text preprocessing for biomedical NLP.

This module implements a preprocessing pipeline suitable for
traditional NLP methods such as TF-IDF, Bag-of-Words,
topic modeling, and keyword extraction.
"""

from __future__ import annotations
import re

from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from litminion.config import DEFAULT_SPACY_MODEL
from litminion.preprocessing.base import BasePreprocessor


class ClassicalPreprocessor(BasePreprocessor):
    """
    Classical text preprocessing pipeline based on spaCy.
    """

    def __init__(
        self,
        model: str = DEFAULT_SPACY_MODEL,
        lowercase: bool = False,
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        lemmatize: bool = True,
    ) -> None:
        """
        Initialize the preprocessor.

        Parameters
        ----------
        model : str
            Name of the spaCy language model.

        lowercase : bool
            Convert text to lowercase.

        remove_stopwords : bool
            Remove stopwords.

        remove_punctuation : bool
            Remove punctuation.

        lemmatize : bool
            Replace words with their lemmas.
        """

        self.model = model

        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize

        self._nlp: Optional[Language] = None

    def transform(self, text: str) -> str:
        """
        Transform a single document.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Preprocessed text.
        """

        self._load_model()

        text = self._clean_text(text)

        if self.lowercase:
            text = text.lower()

        # At this point _nlp cannot be None because _load_model()
        # has initialized it.
        assert self._nlp is not None

        doc = self._nlp(text)

        tokens = self._process_document(doc)

        return " ".join(tokens)

    def _clean_text(self, text: str) -> str:
        """
        Perform basic text cleaning.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text.
        """

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _load_model(self) -> None:
        """
        Load the spaCy language model on demand.

        The model is loaded only the first time it is needed
        and then cached for future use.
        """

        if self._nlp is not None:
            return

        try:
            self._nlp = spacy.load(self.model)

        except OSError as exc:
            raise OSError(
                f"spaCy model '{self.model}' is not installed.\n\n"
                f"Install it with:\n"
                f"python -m spacy download {self.model}"
            ) from exc

    def _process_document(self, doc: Doc) -> list[str]:
        """
        Process a spaCy document into normalized tokens.

        Parameters
        ----------
        doc : Doc
            spaCy document.

        Returns
        -------
        list[str]
            Processed tokens.
        """

        # TODO:
        # Improve handling of biomedical entities (e.g. SARS-CoV-2,
        # TNF-alpha, gene symbols) using scispaCy or custom token rules.

        tokens: list[str] = []

        for token in doc:

            if token.is_space:
                continue

            if self.remove_punctuation and token.is_punct:
                continue

            if self.remove_stopwords and token.is_stop:
                continue

            if self.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        return tokens
