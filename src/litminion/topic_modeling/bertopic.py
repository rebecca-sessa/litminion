"""
bertopic.py

Topic modeling using BERTopic.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import numpy as np

import numpy as np
from sklearn.utils.validation import check_is_fitted

from bertopic import BERTopic

from litminion.embeddings import (
    SentenceTransformerExtractor,
)

from litminion.topic_modeling.base import (
    BaseTopicModel,
)


class BERTopicModel(BaseTopicModel):
    """
    Discover semantic topics from biomedical documents using BERTopic.

    This class provides a high-level wrapper around BERTopic while
    exposing an API consistent with the rest of litminion.
    """

    def __init__(
        self,
        embedding_extractor: (
            SentenceTransformerExtractor | None
        ) = None,
        calculate_probabilities: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the BERTopic model.

        Parameters
        ----------
        embedding_extractor : SentenceTransformerExtractor | None
            Embedding extractor used internally. If None,
            a SentenceTransformerExtractor is created.

        calculate_probabilities : bool, default=True
            Whether to compute topic probabilities.

        verbose : bool, default=False
            Whether BERTopic should print progress messages.
        """

        self.embedding_extractor = (
            embedding_extractor
            if embedding_extractor is not None
            else SentenceTransformerExtractor()
        )

        self.calculate_probabilities = (
            calculate_probabilities
        )

        self.verbose = verbose

        self._model: BERTopic | None = None

        self.documents: list[str] = []

        self.topics: list[int] | None = None

        self.probabilities: np.ndarray | None = None

    def _load_model(
        self,
    ) -> None:
        """
        Initialize the BERTopic model.
        """
        @property
        def model(self,) -> SentenceTransformer:
            """
            Return the underlying SentenceTransformer model.
            """

            self._load_model()
            assert self._model is not None
            return self._model

        self._model = BERTopic(
            embedding_model=self.embedding_extractor.model,
            calculate_probabilities=self.calculate_probabilities,
            verbose=self.verbose,
        )

    def _check_fitted(
        self,
    ) -> None:
        """
        Check whether the topic model has been fitted.

        Raises
        ------
        ValueError
            If the topic model has not been fitted.
        """

        if self.topics is None:

            raise ValueError(
                "The topic model has not been fitted. "
                "Call 'fit()' or 'fit_transform()' first."
            )

    def _validate_document(
        self,
        document: int,
    ) -> None:
        """
        Validate a document index.

        Parameters
        ----------
        document : int
            Document index.

        Raises
        ------
        ValueError
            If the topic model has not been fitted.

        IndexError
            If the document index is out of range.
        """

        self._check_fitted()

        if (
            document < 0
            or document >= len(self.documents)
        ):

            raise IndexError(
                f"Document index {document} is out of range. "
                f"The corpus contains {len(self.documents)} documents."
            )

    def _validate_topic(
        self,
        topic: int,
    ) -> None:
        """
        Validate a topic identifier.

        Parameters
        ----------
        topic : int
            Topic identifier.

        Raises
        ------
        ValueError
            If the topic model has not been fitted.

        ValueError
            If the topic does not exist.
        """

        self._check_fitted()

        assert self._model is not None

        topics = self._model.get_topics()

        if topic not in topics:

            raise ValueError(
                f"Topic {topic} does not exist."
            )

    def fit(
        self,
        documents: Sequence[str],
    ) -> "BERTopicModel":
        """
        Learn topics from a corpus.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        BERTopicModel
            The fitted topic model.
        """

        self._load_model()

        assert self._model is not None

        self.documents = list(documents)

        if not self.documents:

            self.topics = []

            self.probabilities = np.empty((0,))

            return self

        (
            self.topics,
            self.probabilities,
        ) = self._model.fit_transform(
            self.documents,
        )

        return self

    def transform(
        self,
        documents: Sequence[str],
    ) -> list[int]:
        """
        Assign topics to new documents.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        list[int]
            Topic assignment for each document.
        """

        self._check_fitted()

        assert self._model is not None

        topics, _ = self._model.transform(
            list(documents),
        )

        return topics

    def fit_transform(
        self,
        documents: Sequence[str],
    ) -> list[int]:
        """
        Fit the topic model and immediately return the
        assigned topics.

        Parameters
        ----------
        documents : Sequence[str]
            Preprocessed documents.

        Returns
        -------
        list[int]
            Topic assignment for each document.
        """

        self.fit(documents)

        assert self.topics is not None

        return self.topics

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

        self._validate_topic(topic)

        assert self._model is not None

        topic_words = self._model.get_topic(
            topic,
        )

        if topic_words is None:

            return []

        return [
            (
                str(word),
                float(score),
            )
            for word, score in topic_words
        ]

    def get_topics(
        self,
    ) -> dict[int, list[tuple[str, float]]]:
        """
        Return all discovered topics.

        Returns
        -------
        dict[int, list[tuple[str, float]]]
            Dictionary mapping topic identifiers to
            representative words.
        """

        self._check_fitted()

        assert self._model is not None

        topics = self._model.get_topics()

        return {
            int(topic): [
                (
                    str(word),
                    float(score),
                )
                for word, score in words
            ]
            for topic, words in topics.items()
        }

    def get_topic_info(
        self,
    ) -> pd.DataFrame:
        """
        Return summary information for all topics.

        Returns
        -------
        pandas.DataFrame
            Topic summary table.
        """

        self._check_fitted()

        assert self._model is not None

        return self._model.get_topic_info()

    def get_document_topics(
        self,
    ) -> pd.DataFrame:
        """
        Return the assigned topic for every fitted document.

        Returns
        -------
        pandas.DataFrame
            Document-topic assignments.
        """

        self._check_fitted()

        assert self.topics is not None

        data = {
            "Document": list(
                range(
                    len(self.documents),
                )
            ),
            "Topic": self.topics,
        }

        if self.probabilities is not None:

            probabilities = np.asarray(
                self.probabilities,
            )

            if probabilities.ndim == 2:

                data["Probability"] = (
                    probabilities.max(
                        axis=1,
                    )
                )

        return pd.DataFrame(
            data,
        )

    def plot_topics(
        self,
    ):
        """
        Visualize the discovered topics.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive topic visualization.
        """

        self._check_fitted()

        assert self._model is not None

        return self._model.visualize_topics()

    def plot_topic_hierarchy(
        self,
    ):
        """
        Visualize the topic hierarchy.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive topic hierarchy.
        """

        self._check_fitted()

        assert self._model is not None

        return self._model.visualize_hierarchy()

    def plot_topic_similarity(
        self,
    ):
        """
        Visualize similarities between topics.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive topic similarity heatmap.
        """

        self._check_fitted()

        assert self._model is not None

        return self._model.visualize_heatmap()

    def plot_topic_sizes(
        self,
    ):
        """
        Visualize the distribution of topic sizes.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive bar chart of topic sizes.
        """

        self._check_fitted()

        assert self._model is not None

        return self._model.visualize_barchart()

    def save(
        self,
        path: str,
    ) -> None:
        """
        Save the fitted BERTopic model.

        Parameters
        ----------
        path : str
            Output directory.
        """

        self._check_fitted()

        assert self._model is not None

        self._model.save(
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "BERTopicModel":
        """
        Load a previously saved BERTopic model.

        Parameters
        ----------
        path : str
            Model directory.

        Returns
        -------
        BERTopicModel
            Loaded topic model.
        """

        model = cls()

        model._model = BERTopic.load(
            path,
        )

        return model

    def __len__(
        self,
    ) -> int:
        """
        Return the number of processed documents.

        Returns
        -------
        int
            Number of documents.
        """

        return len(self.documents)

    def __repr__(
        self,
    ) -> str:
        """
        Return a string representation of the topic model.

        Returns
        -------
        str
            Summary of the topic model.
        """

        fitted = self.topics is not None

        n_topics = (
            len(set(self.topics))
            if self.topics is not None
            else 0
        )

        return (
            f"BERTopicModel("
            f"documents={len(self)}, "
            f"topics={n_topics}, "
            f"fitted={fitted}"
            f")"
        )
