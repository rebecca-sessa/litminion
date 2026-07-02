"""
base.py

Abstract base classes for document embedding extraction.

All embedding extractors in litminion should inherit from
BaseEmbeddingExtractor and implement the fit() and
transform() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BaseEmbeddingExtractor(ABC):
    """
    Abstract base class for document embedding extraction.
    """

    def __init__(self) -> None:
        """
        Initialize the embedding extractor.
        """

        self.embeddings: np.ndarray | None = None

    @abstractmethod
    def fit(
        self,
        documents: Sequence[str],
    ) -> "BaseEmbeddingExtractor":
        """
        Fit the embedding model.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        BaseEmbeddingExtractor
            Fitted embedding extractor.
        """
        ...

    @abstractmethod
    def transform(
        self,
        documents: Sequence[str],
    ) -> np.ndarray:
        """
        Generate document embeddings.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        numpy.ndarray
            Matrix of document embeddings with shape
            (n_documents, embedding_dimension).
        """
        ...

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> np.ndarray:
        """
        Fit the extractor and generate document embeddings.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        numpy.ndarray
            Matrix of document embeddings.
        """

        self.fit(documents)

        self.embeddings = self.transform(documents)

        return self.embeddings

    def get_embeddings(
        self,
    ) -> np.ndarray:
        """
        Return the generated document embeddings.

        Returns
        -------
        numpy.ndarray
            Matrix of document embeddings.

        Raises
        ------
        ValueError
            If embeddings have not been generated.
        """

        if self.embeddings is None:

            raise ValueError(
                "No embeddings available. "
                "Call 'fit()', 'transform()', or "
                "'fit_transform()' first."
            )

        return self.embeddings

    def __len__(
        self,
    ) -> int:
        """
        Return the number of embedded documents.

        Returns
        -------
        int
            Number of document embeddings.
        """

        if self.embeddings is None:
            return 0

        return len(self.embeddings)
