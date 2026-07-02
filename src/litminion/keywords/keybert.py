"""
keybert.py

Semantic keyword extraction using KeyBERT.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from keybert import KeyBERT

from litminion.config import (
    DEFAULT_KEYBERT_TOP_N,
    DEFAULT_KEYBERT_NGRAM_RANGE,
    DEFAULT_KEYBERT_STOP_WORDS,
)

from litminion.embeddings import (
    SentenceTransformerExtractor,
)

from litminion.keywords.base import (
    BaseKeywordExtractor,
)


class KeyBERTKeywordExtractor(BaseKeywordExtractor):
    """
    Extract semantic keywords from biomedical documents using KeyBERT.

    This class wraps the KeyBERT library while exposing a consistent API
    with the rest of litminion. Keyword extraction results are computed
    once during ``fit()`` and cached for subsequent retrieval.
    """

    def __init__(
        self,
        top_n: int = DEFAULT_KEYBERT_TOP_N,
        ngram_range: tuple[int, int] = DEFAULT_KEYBERT_NGRAM_RANGE,
        stop_words: str | list[str] | None = DEFAULT_KEYBERT_STOP_WORDS,
        embedding_extractor: SentenceTransformerExtractor | None = None,
    ) -> None:
        """
        Initialize the KeyBERT keyword extractor.

        Parameters
        ----------
        top_n : int
            Default number of keywords extracted per document.

        ngram_range : tuple[int, int]
            Lower and upper boundary of candidate n-grams.

        stop_words : str | list[str] | None
            Stop-word strategy passed to KeyBERT.

        embedding_extractor : SentenceTransformerExtractor | None
            Embedding extractor used internally. If None, a new
            SentenceTransformerExtractor is created.
        """

        self.top_n = top_n

        self.ngram_range = ngram_range

        self.stop_words = stop_words

        self.embedding_extractor = (
            embedding_extractor
            if embedding_extractor is not None
            else SentenceTransformerExtractor()
        )

        self._model: KeyBERT | None = None

        self.documents: list[str] = []

        self.keywords: list[
            list[tuple[str, float]]
        ] | None = None

    def _load_model(
        self,
    ) -> None:
        """
        Lazily initialize the KeyBERT model.
        """

        if self._model is None:

            self.embedding_extractor._load_model()

            self._model = KeyBERT(
                model=self.embedding_extractor.model
            )

    def _validate_document(
        self,
        document: int,
    ) -> None:
        """
        Validate a document index.

        Parameters
        ----------
        document : int
            Document index.

        Raises
        ------
        ValueError
            If the extractor has not been fitted.

        IndexError
            If the document index is out of range.
        """

        if self.keywords is None:

            raise ValueError(
                "The keyword extractor has not been fitted. "
                "Call 'fit()' or 'fit_transform()' first."
            )

        if (
            document < 0
            or document >= len(self.documents)
        ):

            raise IndexError(
                f"Document index {document} is out of range. "
                f"The corpus contains {len(self.documents)} documents."
            )

    def _validate_n(
        self,
        n: int,
    ) -> None:
        """
        Validate the requested number of keywords.

        Parameters
        ----------
        n : int
            Number of keywords.

        Raises
        ------
        ValueError
            If ``n`` is not positive.
        """

        if n <= 0:

            raise ValueError(
                "'n' must be greater than zero."
            )

    def fit(
        self,
        documents: Sequence[str],
    ) -> "KeyBERTKeywordExtractor":
        """
        Extract keywords from a corpus of documents.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        KeyBERTKeywordExtractor
            The fitted keyword extractor.
        """

        self._load_model()

        assert self._model is not None

        self.documents = list(documents)

        self.keywords = []

        if not self.documents:

            return self

        self.embedding_extractor.fit(
            self.documents,
        )

        for document in self.documents:

            keywords = self._model.extract_keywords(
                document,
                keyphrase_ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                top_n=self.top_n,
            )

            self.keywords.append(
                [
                    (
                        str(keyword),
                        float(score),
                    )
                    for keyword, score in keywords
                ]
            )

        return self

    def transform(
        self,
    ) -> list[list[tuple[str, float]]]:
        """
        Return the extracted keywords.

        Parameters
        ----------
        documents : Sequence[str] | None, optional
            Ignored. Present only for API consistency with
            scikit-learn estimators.

        Returns
        -------
        list[list[tuple[str, float]]]
            Cached keywords for every document.

        Raises
        ------
        ValueError
            If the extractor has not been fitted.
        """

        if self.keywords is None:

            raise ValueError(
                "The keyword extractor has not been fitted. "
                "Call 'fit()' or 'fit_transform()' first."
            )

        return self.keywords

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> list[list[tuple[str, float]]]:
        """
        Fit the extractor and return the extracted keywords.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        list[list[tuple[str, float]]]
            Extracted keywords for every document.
        """

        self.fit(documents)

        assert self.keywords is not None

        return self.keywords

    def get_keywords(
        self,
        document: int = 0,
        n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Return the highest-scoring keywords for a document.

        Parameters
        ----------
        document : int, default=0
            Document index.

        n : int, default=20
            Number of keywords to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (keyword, score) pairs sorted by decreasing score.
        """

        self._validate_document(document)

        self._validate_n(n)

        assert self.keywords is not None

        return self.keywords[document][:n]

    def to_dataframe(
        self,
        document: int = 0,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Convert extracted keywords into a pandas DataFrame.

        Parameters
        ----------
        document : int, default=0
            Document index.

        n : int | None, default=None
            Number of keywords to include. If None, all extracted
            keywords are returned.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing keywords and their scores.
        """

        self._validate_document(document)

        if n is None:

            assert self.keywords is not None

            n = len(
                self.keywords[document]
            )

        keywords = self.get_keywords(
            document=document,
            n=n,
        )

        return pd.DataFrame(
            keywords,
            columns=[
                "Keyword",
                "Score",
            ],
        )

    def plot(
        self,
        document: int = 0,
        n: int = 20,
        ascending: bool = True,
    ) -> Axes:
        """
        Plot the highest-scoring keywords.

        Parameters
        ----------
        document : int, default=0
            Document index.

        n : int, default=20
            Number of keywords to display.

        ascending : bool, default=True
            If True, display the lowest scores at the bottom
            of the chart and the highest at the top.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.
        """

        df = self.to_dataframe(
            document=document,
            n=n,
        )

        if ascending:

            df = df.iloc[::-1]

        plt.figure(
            figsize=(8, 6),
        )

        plt.barh(
            df["Keyword"],
            df["Score"],
        )

        plt.xlabel("KeyBERT Score")

        plt.ylabel("Keyword")

        plt.title(
            f"Top {n} Keywords (Document {document})"
        )

        plt.tight_layout()

        ax = plt.gca()

        plt.show()

        return ax

    def __len__(
        self,
    ) -> int:
        """
        Return the number of processed documents.

        Returns
        -------
        int
            Number of documents.
        """

        return len(self.documents)

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation of the keyword extractor.

        Returns
        -------
        str
            Summary of the extractor.
        """

        fitted = self.keywords is not None

        return (
            f"KeyBERTKeywordExtractor("
            f"documents={len(self)}, "
            f"fitted={fitted}"
            f")"
        )
