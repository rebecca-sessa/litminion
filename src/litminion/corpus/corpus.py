"""
corpus.py

Corpus abstraction for biomedical literature collections.

The Corpus class provides a unified interface for storing,
preprocessing, and managing collections of scientific publications.
"""

from __future__ import annotations

from typing import Self

import pandas as pd

from litminion.data.downloader import download_pubmed
from litminion.preprocessing.base import BasePreprocessor


class Corpus:
    """
    Represent a collection of biomedical publications.
    """

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        """
        Initialize a corpus.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing biomedical publications.
        """

        self.data = data.copy()

        self.abstracts: list[str] = (
            self.data["Abstract"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.processed_abstracts: list[str] | None = None

        self.preprocessor: BasePreprocessor | None = None

    @classmethod
    def from_pubmed(
        cls,
        query: str,
        max_results: int = 100,
        abstract_only: bool = True,
    ) -> Self:
        """
        Build a corpus directly from PubMed.

        Parameters
        ----------
        query : str
            PubMed search query.

        max_results : int, default=100
            Maximum number of articles to retrieve.

        abstract_only : bool, default=True
            Whether to discard records without abstracts.

        Returns
        -------
        Corpus
            Corpus containing the downloaded articles.
        """

        data = download_pubmed(
            query=query,
            max_results=max_results,
            abstract_only=abstract_only,
        )

        return cls(data)

    def preprocess(
        self,
        preprocessor: BasePreprocessor,
    ) -> Self:
        """
        Preprocess all abstracts.

        Parameters
        ----------
        preprocessor : BasePreprocessor
            Preprocessor used to transform the corpus.

        Returns
        -------
        Corpus
            The current corpus.
        """

        self.preprocessor = preprocessor

        self.processed_abstracts = (
            preprocessor.transform_corpus(
                self.abstracts
            )
        )

        return self

    def head(
        self,
        n: int = 5,
    ) -> pd.DataFrame:
        """
        Return the first rows of the corpus.

        Parameters
        ----------
        n : int, default=5
            Number of rows.

        Returns
        -------
        pandas.DataFrame
            First rows of the corpus.
        """

        return self.data.head(n)

    def __len__(
        self,
    ) -> int:
        """
        Return the number of publications.

        Returns
        -------
        int
            Number of publications.
        """

        return len(self.data)

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation of the corpus.

        Returns
        -------
        str
            Summary of the corpus.
        """

        processed = self.processed_abstracts is not None

        return (

            "Corpus("

            f"documents={len(self)}, "

            f"processed={self.processed_abstracts is not None}, "

            f"columns={len(self.columns)}"

            ")"

        )

    def get_documents(
        self,
    ) -> list[str]:
        """
        Return the documents available for analysis.

        If the corpus has been preprocessed, the processed abstracts
        are returned. Otherwise, the raw abstracts are returned.

        Returns
        -------
        list[str]
            Documents available for downstream analysis.
        """

        if self.processed_abstracts is not None:
            return self.processed_abstracts

        return self.abstracts

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        """
        Return the shape of the corpus.

        Returns
        -------
        tuple[int, int]
            Number of rows and columns.
        """

        return self.data.shape

    @property
    def columns(
        self,
    ) -> pd.Index:
        """
        Return the column names.

        Returns
        -------
        pandas.Index
            DataFrame column names.
        """

        return self.data.columns

    def copy(
        self,
    ) -> Self:
        """
        Return a deep copy of the corpus.

        Returns
        -------
        Corpus
            Independent copy of the corpus.
        """

        new = Corpus(self.data.copy())

        new.abstracts = self.abstracts.copy()

        if self.processed_abstracts is not None:
            new.processed_abstracts = self.processed_abstracts.copy()

        new.preprocessor = self.preprocessor

        return new

    def __getitem__(
        self,
        key: str,
    ) -> pd.Series:
        """
        Return a column from the underlying DataFrame.

        Parameters
        ----------
        key : str
            Column name.

        Returns
        -------
        pandas.Series
            Requested column.
        """

        return self.data[key]

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return a copy of the underlying DataFrame.

        Returns
        -------
        pandas.DataFrame
            Corpus data.
        """

        return self.data.copy()
