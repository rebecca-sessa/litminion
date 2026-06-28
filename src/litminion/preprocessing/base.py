"""
base.py

Abstract base classes for text preprocessing.

All preprocessors in LitMinion should inherit from BasePreprocessor
and implement the transform() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class BasePreprocessor(ABC):
    """
    Abstract base class for all text preprocessors.
    """

    @abstractmethod
    def transform(self, text: str) -> str:
        """
        Transform a single document.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Processed text.
        """
        pass

    def transform_corpus(
        self,
        texts: Iterable[str],
    ) -> list[str]:
        """
        Transform multiple documents.

        Parameters
        ----------
        texts : Iterable[str]
            Collection of input documents.

        Returns
        -------
        list[str]
            List of processed documents.
        """

        return [self.transform(text) for text in texts]
