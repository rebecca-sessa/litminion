"""

preprocessing.py

Utilities for preprocessing biomedical text before downstream NLP tasks.

"""

from __future__ import annotations

import re
from typing import Iterable

import spacy


class ClassicalPreprocessor:
    """

    Preprocess biomedical text using spaCy.

    Parameters

    ----------

    model : str, default="en_core_web_sm"

        Name of the spaCy language model.

    lowercase : bool, default=True

        Convert text to lowercase.

    remove_stopwords : bool, default=True

        Remove stop words.

    lemmatize : bool, default=True

        Replace tokens with their lemmas.

    remove_punctuation : bool, default=True

        Remove punctuation tokens.

    """

    def __init__(
            self,
            model: str = "en_core_web_sm",
            lowercase: bool = True,
            remove_stopwords: bool = True,
            lemmatize: bool = True,
            remove_punctuation: bool = True
    ) -> None:

        self.nlp = spacy.load(model)

        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_punctuation = remove_punctuation

    def clean_text(self, text: str) -> str:
        """

    Perform basic text cleaning.

    Parameters

    ----------

    text : str

        Raw text.

    Returns

    -------

    str

        Cleaned text.

    """
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def transform(self, text: str) -> str:
        """

        Preprocess a single document.

        Parameters

        ----------

        text : str

            Raw text.

        Returns

        -------

        str

            Preprocessed text.

        """

        text = self.clean_text(text)

        if self.lowercase:
            text = text.lower()

        doc = self.nlp(text)

        tokens = []

        for token in doc:
            if self.remove_stopwords and token.is_stop:
                continue

            if self.remove_punctuation and token.is_punct:
                continue

            if token.is_space:
                continue

            if self.lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        return " ".join(tokens)

    def transform_corpus(

        self,

        texts: Iterable[str],

    ) -> list[str]:
        """

        Preprocess multiple documents.

        Parameters

        ----------

        texts : iterable of str

            Collection of documents.

        Returns

        -------

        list[str]

            Preprocessed documents.

        """

        return [self.transform(text) for text in texts]
