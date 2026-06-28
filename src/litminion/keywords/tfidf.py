"""
tfidf.py

Keyword extraction based on corpus-level TF-IDF scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.utils.validation import check_is_fitted

from litminion.features.tfidf import TfidfExtractor
from litminion.keywords.base import BaseKeywordExtractor


class TfidfKeywordExtractor(BaseKeywordExtractor):
    """
    Extract corpus-level keywords using TF-IDF.

    The importance of each term is computed as the mean TF-IDF
    score across all documents in the corpus.
    """

    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] = (1, 1),
    ) -> None:
        """
        Initialize the keyword extractor.

        Parameters
        ----------
        max_features : int | None, default=None
            Maximum vocabulary size.

        ngram_range : tuple[int, int], default=(1, 1)
            Lower and upper boundary of the n-gram range.
        """

        self.extractor = TfidfExtractor(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        self.keyword_scores: pd.Series | None = None

    def fit(
        self,
        documents: Sequence[str],
    ) -> "TfidfKeywordExtractor":
        """
        Learn keyword scores from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        TfidfKeywordExtractor
            Fitted extractor.
        """

        matrix = self.extractor.fit_transform(documents)

        feature_names = self.extractor.get_feature_names()

        mean_scores = matrix.mean(axis=0).A1

        self.keyword_scores = pd.Series(
            mean_scores,
            index=feature_names,
        ).sort_values(
            ascending=False
        )

        return self

    def get_keywords(
        self,
        n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Return the highest-scoring keywords.

        Parameters
        ----------
        n : int, default=20
            Number of keywords.

        Returns
        -------
        list[tuple[str, float]]
            (keyword, score) pairs.
        """

        if self.keyword_scores is None:
            raise ValueError(
                "The keyword extractor has not been fitted. "
                "Call 'fit()' or 'fit_extract()' first."
            )

        if n <= 0:
            raise ValueError(
                "'n' must be greater than zero."
            )

        return [
            (term, float(score))
            for term, score in self.keyword_scores.head(n).items()
        ]

    def to_dataframe(
        self,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Return keywords as a pandas DataFrame.

        Parameters
        ----------
        n : int | None, default=None
            Number of keywords to include.

        Returns
        -------
        pandas.DataFrame
            Keyword table.
        """

        if self.keyword_scores is None:
            raise ValueError(
                "The keyword extractor has not been fitted."
            )

        if n is None:
            df = self.keyword_scores

        else:
            df = self.keyword_scores.head(n)

        return df.rename(
            "Score"
        ).rename_axis(
            "Keyword"
        ).reset_index()

    def plot(
        self,
        n: int = 20,
        ascending: bool = True,
    ) -> Axes:
        """
        Plot the highest-scoring keywords.

        Parameters
        ----------
        n : int, default=20
            Number of keywords.

        ascending : bool, default=True
            Display bars from smallest to largest.

        Returns
        -------
        matplotlib.axes.Axes
            Plot axes.
        """

        df = self.to_dataframe(n)

        if ascending:
            df = df.iloc[::-1]

        plt.figure(figsize=(8, 6))

        plt.barh(
            df["Keyword"],
            df["Score"],
        )

        plt.xlabel("Mean TF-IDF Score")
        plt.ylabel("Keyword")
        plt.title(f"Top {n} Keywords")

        plt.tight_layout()

        ax = plt.gca()

        plt.show()

        return ax
