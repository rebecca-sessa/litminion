"""
base.py

Abstract base classes for topic modeling.

All topic models in litminion should inherit from
BaseTopicModel and implement the fit() and
transform() methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class BaseTopicModel(ABC):
    """
    Abstract base class for topic modeling algorithms.
    """

    @abstractmethod
    def fit(
        self,
        documents: Sequence[str],
    ) -> "BaseTopicModel":
        """
        Learn topics from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        BaseTopicModel
            The fitted topic model.
        """
        ...

    @abstractmethod
    def transform(
        self,
        documents: Sequence[str],
    ) -> list[int]:
        """
        Assign topics to a collection of documents.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        list[int]
            Topic assignment for each document.
        """
        ...

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> list[int]:
        """
        Fit the topic model and immediately assign topics.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of preprocessed documents.

        Returns
        -------
        list[int]
            Topic assignment for each document.
        """

        self.fit(documents)

        return self.transform(documents)

    @abstractmethod
    def get_topic(
        self,
        topic: int,
    ) -> list[tuple[str, float]]:
        """
        Return the representative words of a topic.

        Parameters
        ----------
        topic : int
            Topic identifier.

        Returns
        -------
        list[tuple[str, float]]
            List of (term, score) pairs describing the topic.
        """
        ...

    @abstractmethod
    def get_topic_info(
        self,
    ) -> pd.DataFrame:
        """
        Return summary information for all discovered topics.

        Returns
        -------
        pandas.DataFrame
            Topic summary table.
        """
        ...
