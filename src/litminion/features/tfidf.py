"""
tfidf.py

TF-IDF feature extraction for biomedical corpora.
"""

from __future__ import annotations
from sklearn.utils.validation import check_is_fitted
from collections.abc import Sequence

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfExtractor:
    """
    Extract TF-IDF features from a corpus of documents.

    This class wraps scikit-learn's TfidfVectorizer while providing
    a simplified API tailored to biomedical literature mining.
    """

    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] = (1, 1),
    ) -> None:
        """
        Initialize the TF-IDF extractor.

        Parameters
        ----------
        max_features : int | None, default=None
            Maximum number of terms to include.

        ngram_range : tuple[int, int], default=(1, 1)
            Lower and upper boundary of the n-gram range.
        """

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        self.matrix: csr_matrix | None = None

    def fit(
        self,
        documents: Sequence[str],
    ) -> "TfidfExtractor":
        """
        Learn the TF-IDF vocabulary from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.
        """

        self.matrix = self.vectorizer.fit_transform(documents)

        return self

    def transform(
        self,
        documents: Sequence[str],
    ) -> csr_matrix:
        """
        Transform a collection of preprocessed documents into TF-IDF vectors.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        csr_matrix
            TF-IDF representation of the documents.

        Raises
        ------
        ValueError
            If the vectorizer has not been fitted.
        """

        try:
            check_is_fitted(self.vectorizer)

        except Exception as exc:
            raise ValueError(
                "The TF-IDF extractor has not been fitted. "
                "Call 'fit()' or 'fit_transform()' first."
            ) from exc

        return self.vectorizer.transform(documents)

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> csr_matrix:
        """
        Fit the extractor and transform the documents.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        csr_matrix
            TF-IDF matrix.
        """

        self.matrix = self.vectorizer.fit_transform(documents)

        return self.matrix

    def get_feature_names(self) -> Sequence[str]:
        """
        Return the learned feature names.

        Returns

        -------

        Sequence[str]

            Vocabulary learned by the TF-IDF vectorizer.
        """

        check_is_fitted(self.vectorizer)

        return self.vectorizer.get_feature_names_out()

    def get_top_terms(
        self,
        document: int,
        n: int = 20,

    ) -> list[tuple[str, float]]:
        """
        Return the top TF-IDF terms for a document.

        Parameters
        ----------
        document : int
            Index of the document.

        n : int, default=20
            Number of terms to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (term, score) pairs sorted by decreasing TF-IDF score.
        """

        check_is_fitted(self.vectorizer)

        if self.matrix is None:

            raise ValueError(

                "No TF-IDF matrix available. "
                "Call 'fit()' or 'fit_transform()' first."

            )

        if document < 0 or document >= self.matrix.shape[0]:
            raise IndexError(
                f"Document index {document} is out of range. "
                f"The corpus contains {self.matrix.shape[0]} documents."
            )

        if n <= 0:
            raise ValueError(
                "'n' must be greater than zero."
            )

        feature_names = self.get_feature_names()
        row = self.matrix[document].toarray().ravel()
        scores = zip(feature_names, row)

        scores = [

            (term, float(score))
            for term, score in scores
            if score > 0

        ]

        scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return scores[:n]

    def get_vocabulary_size(self) -> int:
        """
        Return the size of the learned vocabulary.

        Returns
        -------
        int
            Number of unique terms in the vocabulary.
        """

        check_is_fitted(self.vectorizer)

        return len(self.vectorizer.vocabulary_)

    def to_dataframe(
        self,
        document: int,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Convert the TF-IDF representation of a document into a pandas DataFrame.

        Parameters
        ----------
        document : int
            Index of the document.

        n : int | None, default=None
            Number of top terms to include. If None, all non-zero
            TF-IDF terms are returned.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the selected terms and their TF-IDF
            scores sorted in descending order.

        Notes
        -----
        This method is a convenience wrapper around
        ``get_top_terms()`` and is intended for exploratory
        analysis, reporting, and visualization.
        """

        if n is None:
            n = self.get_vocabulary_size()

        terms = self.get_top_terms(
            document=document,
            n=n,
        )

        return pd.DataFrame(
            terms,
            columns=["Term", "Score"],
        )

    def plot(
        self,
        document: int = 0,
        n: int = 20,
        ascending: bool = True,
    ) -> Axes:
        """
        Plot the highest-scoring TF-IDF terms for a document.

        Parameters
        ----------
        document : int
            Index of the document.

        n : int, default=20
            Number of top terms to display.

        ascending : bool, default=True
            If True, display the smallest scores at the bottom of the
            chart and the largest at the top.

        Returns
        -------
        None
            Displays a horizontal bar chart.
        """

        df = self.to_dataframe(
            document=document,
            n=n,
        )

        if ascending:
            df = df.iloc[::-1]

        plt.figure(figsize=(8, 6))

        plt.barh(
            df["Term"],
            df["Score"],
        )

        plt.xlabel("TF-IDF Score")
        plt.ylabel("Term")
        plt.title(f"Top {n} TF-IDF Terms (Document {document})")

        plt.tight_layout()

        ax = plt.gca()

        plt.show()

        return ax
