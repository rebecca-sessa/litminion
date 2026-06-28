"""
ngrams.py

N-gram frequency extraction for biomedical corpora.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.axes import Axes
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted


class NGramExtractor:
    """
    Extract n-gram frequencies from a corpus.

    This class wraps scikit-learn's CountVectorizer while providing
    a simplified API tailored to biomedical literature mining.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 2),
        max_features: int | None = None,
    ) -> None:
        """
        Initialize the n-gram extractor.

        Parameters
        ----------
        ngram_range : tuple[int, int], default=(2, 2)
            Lower and upper boundary of the n-gram range.

        max_features : int | None, default=None
            Maximum number of n-grams to include.
        """

        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
        )

        self.matrix: csr_matrix | None = None

    def fit(
        self,
        documents: Sequence[str],
    ) -> "NgramExtractor":
        """
        Learn the n-gram vocabulary from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        NgramExtractor
            The fitted extractor.
        """

        self.matrix = self.vectorizer.fit_transform(documents)

        return self

    def transform(
        self,
        documents: Sequence[str],
    ) -> csr_matrix:
        """
        Transform documents into an n-gram count matrix.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        csr_matrix
            Sparse matrix of n-gram counts.
        """

        try:
            check_is_fitted(self.vectorizer)

        except Exception as exc:
            raise ValueError(
                "The n-gram extractor has not been fitted. "
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
            Sparse n-gram count matrix.
        """

        self.matrix = self.vectorizer.fit_transform(documents)

        return self.matrix

    def get_feature_names(self) -> Sequence[str]:
        """
        Return the learned n-gram vocabulary.

        Returns
        -------
        Sequence[str]
            Learned n-grams.
        """

        check_is_fitted(self.vectorizer)

        return self.vectorizer.get_feature_names_out()

    def get_vocabulary_size(self) -> int:
        """
        Return the size of the learned vocabulary.

        Returns
        -------
        int
            Number of unique n-grams.
        """

        check_is_fitted(self.vectorizer)

        return len(self.vectorizer.vocabulary_)

    def get_ngram_counts(
        self,
    ) -> dict[str, int]:
        """
        Return the frequency of each n-gram across the corpus.

        Returns
        -------
        dict[str, int]
            Dictionary mapping n-grams to their total counts
            across all documents.
        """

        try:

            check_is_fitted(self.vectorizer)

        except Exception as exc:

            raise ValueError(

                "The n-gram extractor has not been fitted. "

                "Call 'fit()' or 'fit_transform()' first."

            ) from exc

        if self.matrix is None:
            raise ValueError(
                "No n-gram matrix available. "
                "Call 'fit()' or 'fit_transform()' first."
            )

        feature_names = self.get_feature_names()

        counts = self.matrix.sum(axis=0).A1

        return {
            ngram: int(count)
            for ngram, count in zip(feature_names, counts)
        }

    def get_top_ngrams(
        self,
        n: int = 20,
    ) -> list[tuple[str, int]]:
        """
        Return the most frequent n-grams in the corpus.

        Parameters
        ----------
        n : int, default=20
            Number of n-grams to return.

        Returns
        -------
        list[tuple[str, int]]
            List of (n-gram, count) pairs sorted by decreasing
            frequency.
        """

        if n <= 0:
            raise ValueError(
                "'n' must be greater than zero."
            )

        counts = self.get_ngram_counts()

        return sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    def to_dataframe(
        self,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Convert n-gram frequencies into a pandas DataFrame.

        Parameters
        ----------
        n : int | None, default=None
            Number of top n-grams to include.
            If None, all n-grams are returned.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing n-grams and their frequencies.
        """

        if n is None:
            data = sorted(
                self.get_ngram_counts().items(),
                key=lambda x: x[1],
                reverse=True,
            )

        else:
            data = self.get_top_ngrams(n)

        return pd.DataFrame(
            data,
            columns=[
                "N-gram",
                "Count",
            ],
        )

    def plot(
        self,
        n: int = 20,
        ascending: bool = True,
    ) -> Axes:
        """
        Plot the most frequent n-grams.

        Parameters
        ----------
        n : int, default=20
            Number of n-grams to display.

        ascending : bool, default=True
            Whether to display the smallest counts at the bottom.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot.
        """

        df = self.to_dataframe(n=n)

        if ascending:
            df = df.iloc[::-1]

        plt.figure(figsize=(8, 6))

        plt.barh(
            df["N-gram"],
            df["Count"],
        )

        plt.xlabel("Count")
        plt.ylabel("N-gram")
        plt.title(f"Top {n} N-grams")

        plt.tight_layout()

        ax = plt.gca()

        plt.show()

        return ax
