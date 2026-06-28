"""
word_frequency.py

Tools for computing word frequencies in a corpus.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Iterable


class WordFrequencyAnalyzer:
    """
    Compute word frequencies across a corpus.
    """

    def __init__(self) -> None:
        self.frequencies: Counter[str] = Counter()

    def fit(
        self,
        documents: Sequence[str],
    ) -> None:
        """
        Compute word frequencies from a collection of documents.

        Parameters
        ----------
        documents : list[str]
            Preprocessed documents.
        """

        self.frequencies.clear()

        for document in documents:

            tokens = document.split()

            self.frequencies.update(tokens)

    def transform(self) -> Counter[str]:
        """
        Return the computed word frequencies.

        Returns
        -------
        Counter[str]
            Word frequency counter.
        """

        return self.frequencies

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> Counter[str]:
        """
        Fit the analyzer and return the frequencies.

        Parameters
        ----------
        documents : list[str]
            Preprocessed documents.

        Returns
        -------
        Counter[str]
            Word frequency counter.
        """

        self.fit(documents)

        return self.transform()

    def top_n(
        self,
        n: int = 20,
    ) -> list[tuple[str, int]]:
        """
        Return the n most frequent words.

        Parameters
        ----------
        n : int, default=20
            Number of words to return.

        Returns
        -------
        list[tuple[str, int]]
            Word-frequency pairs sorted by frequency.
        """

        return self.frequencies.most_common(n)
