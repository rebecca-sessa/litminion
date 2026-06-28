"""
base.py

Abstract base classes for keyword extraction.

All keyword extractors in litminion should inherit from
BaseKeywordExtractor and implement the fit() and
get_keywords() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseKeywordExtractor(ABC):
    """
    Abstract base class for keyword extraction algorithms.
    """

    @abstractmethod
    def fit(
        self,
        documents: Sequence[str],
    ) -> "BaseKeywordExtractor":
        """
        Learn keyword statistics from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        BaseKeywordExtractor
            The fitted extractor.
        """
        ...

    @abstractmethod
    def get_keywords(
        self,
        n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Return the highest-scoring keywords.

        Parameters
        ----------
        n : int, default=20
            Number of keywords to return.

        Returns
        -------
        list[tuple[str, float]]
            List of (keyword, score) pairs sorted by
            decreasing importance.
        """
        ...

    def fit_extract(
        self,
        documents: Sequence[str],
        n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Fit the extractor and immediately return keywords.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        n : int, default=20
            Number of keywords to return.

        Returns
        -------
        list[tuple[str, float]]
            Top keywords extracted from the corpus.
        """

        self.fit(documents)

        return self.get_keywords(n)
