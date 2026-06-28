"""
word_frequency.py

Word frequency extraction for biomedical corpora.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


class WordFrequencyExtractor:
    """
    Compute word frequencies across a corpus.
    """

    def __init__(self) -> None:
        """
        Initialize the word frequency extractor.
        """

        self.frequencies: Counter[str] | None = None

    def fit(
        self,
        documents: Sequence[str],
    ) -> "WordFrequencyExtractor":
        """
        Compute word frequencies from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        WordFrequencyExtractor
            The fitted extractor.
        """

        self.frequencies = Counter()

        for document in documents:
            self.frequencies.update(document.split())

        return self

    def transform(self) -> Counter[str]:
        """
        Return the computed word frequencies.

        Returns
        -------
        Counter[str]
            Word frequency counter.

        Raises
        ------
        ValueError
            If the extractor has not been fitted.
        """

        if self.frequencies is None:
            raise ValueError(
                "The word frequency extractor has not been fitted. "
                "Call 'fit()' or 'fit_transform()' first."
            )

        return self.frequencies

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> Counter[str]:
        """
        Fit the extractor and return the frequencies.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        Counter[str]
            Word frequencies.
        """

        self.fit(documents)

        return self.transform()

    def get_word_counts(self) -> dict[str, int]:
        """
        Return all word counts.

        Returns
        -------
        dict[str, int]
            Mapping between words and their frequencies.
        """

        return dict(self.transform())

    def get_top_words(
        self,
        n: int = 20,
    ) -> list[tuple[str, int]]:
        """
        Return the most frequent words.

        Parameters
        ----------
        n : int, default=20
            Number of words to return.

        Returns
        -------
        list[tuple[str, int]]
            Word-frequency pairs sorted by decreasing frequency.
        """

        if n <= 0:
            raise ValueError(
                "'n' must be greater than zero."
            )

        return self.transform().most_common(n)

    def get_vocabulary_size(self) -> int:
        """
        Return the vocabulary size.

        Returns
        -------
        int
            Number of unique words.
        """

        return len(self.transform())

    def to_dataframe(
        self,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Convert the word frequencies into a pandas DataFrame.

        Parameters
        ----------
        n : int | None, default=None
            Number of top words to include.
            If None, all words are returned.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing words and counts.
        """

        if n is None:
            words = self.transform().most_common()

        else:
            words = self.get_top_words(n)

        return pd.DataFrame(
            words,
            columns=[
                "Word",
                "Count",
            ],
        )

    def plot(
        self,
        n: int = 20,
        ascending: bool = True,
    ) -> Axes:
        """
        Plot the most frequent words.

        Parameters
        ----------
        n : int, default=20
            Number of words to display.

        ascending : bool, default=True
            If True, display the smallest counts at the bottom
            of the chart.

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
            df["Word"],
            df["Count"],
        )

        plt.xlabel("Count")
        plt.ylabel("Word")
        plt.title(f"Top {n} Words")

        plt.tight_layout()

        ax = plt.gca()

        plt.show()

        return ax
