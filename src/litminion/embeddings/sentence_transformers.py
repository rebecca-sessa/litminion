"""
sentence_transformer.py

Sentence Transformer-based document embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from litminion.config import DEFAULT_SENTENCE_TRANSFORMER
from litminion.embeddings.base import BaseEmbeddingExtractor


class SentenceTransformerExtractor(BaseEmbeddingExtractor):
    """
    Generate document embeddings using Sentence Transformers.
    """

    def __init__(
        self,
        model: str = DEFAULT_SENTENCE_TRANSFORMER,
    ) -> None:
        """
        Initialize the embedding extractor.

        Parameters
        ----------
        model : str, default=DEFAULT_SENTENCE_TRANSFORMER
            Name of the Sentence Transformer model.
        """

        super().__init__()

        self.model_name = model

        self._model: SentenceTransformer | None = None

    def _load_model(
        self,
    ) -> None:
        """
        Load the Sentence Transformer model on demand.
        """

        if self._model is not None:
            return

        try:

            self._model = SentenceTransformer(
                self.model_name
            )

        except Exception as exc:

            raise RuntimeError(
                f"Unable to load Sentence Transformer "
                f"model '{self.model_name}'."
            ) from exc

    def fit(
        self,
        documents: Sequence[str],
    ) -> "SentenceTransformerExtractor":
        """
        Load the embedding model.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of documents.

        Returns
        -------
        SentenceTransformerExtractor
            Fitted extractor.
        """

        self._load_model()

        return self

    def transform(
        self,
        documents: Sequence[str],
    ) -> np.ndarray:
        """
        Generate document embeddings.

        Parameters
        ----------
        documents : Sequence[str]
            Collection of documents.

        Returns
        -------
        numpy.ndarray
            Matrix of document embeddings.
        """

        self._load_model()

        assert self._model is not None

        self.embeddings = self._model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return self.embeddings

    def get_embedding_dimension(
        self,
    ) -> int:
        """
        Return the embedding dimension.

        Returns
        -------
        int
            Embedding dimension.

        Raises
        ------
        ValueError
            If embeddings have not been generated.
        """

        embeddings = self.get_embeddings()

        return embeddings.shape[1]

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        """
        Return the shape of the embedding matrix.

        Returns
        -------
        tuple[int, int]
            Tuple containing the number of documents and the
            embedding dimension.

        Raises
        ------
        ValueError
            If embeddings have not been generated.
        """

        return self.get_embeddings().shape

    def get_n_documents(
        self,
    ) -> int:
        """
        Return the number of embedded documents.

        Returns
        -------
        int
            Number of embedded documents.
        """

        embeddings = self.get_embeddings()

        return embeddings.shape[0]

    def get_embedding(
        self,
        document: int,
    ) -> np.ndarray:
        """
        Return the embedding of a single document.

        Parameters
        ----------
        document : int
            Document index.

        Returns
        -------
        numpy.ndarray
            Embedding vector.

        Raises
        ------
        IndexError
            If the document index is out of range.
        """

        embeddings = self.get_embeddings()

        if document < 0 or document >= embeddings.shape[0]:
            raise IndexError(
                f"Document index {document} is out of range. "
                f"The corpus contains {embeddings.shape[0]} documents."
            )

        return embeddings[document]

    def similarity(
        self,
        document1: int,
        document2: int,
    ) -> float:
        """
        Compute cosine similarity between two documents.

        Parameters
        ----------
        document1 : int
            First document index.

        document2 : int
            Second document index.

        Returns
        -------
        float
            Cosine similarity.
        """

        emb1 = self.get_embedding(document1).reshape(1, -1)
        emb2 = self.get_embedding(document2).reshape(1, -1)

        score = cosine_similarity(
            emb1,
            emb2,
        )[0, 0]

        return float(score)

    def similarity_matrix(
        self,
    ) -> pd.DataFrame:
        """
        Compute the document-document cosine similarity matrix.

        Returns
        -------
        pandas.DataFrame
            Square similarity matrix.
        """

        embeddings = self.get_embeddings()

        matrix = cosine_similarity(
            embeddings,
        )

        return matrix

    def similarity_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.similarity_matrix())

    def most_similar(
        self,
        document: int,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Return the documents most similar to a query document.

        Parameters
        ----------
        document : int
            Query document index.

        n : int, default=10
            Number of similar documents.

        Returns
        -------
        list[tuple[int, float]]
            List of (document_index, similarity_score).
        """

        embeddings = self.get_embeddings()

        if document < 0 or document >= embeddings.shape[0]:
            raise IndexError(
                f"Document index {document} is out of range. "
                f"The corpus contains {embeddings.shape[0]} documents."
            )

        if n <= 0:
            raise ValueError(
                "'n' must be greater than zero."
            )

        similarities = self.similarity_matrix()

        scores = similarities[document]

        indices = np.argsort(scores)[::-1]

        results: list[tuple[int, float]] = []

        for idx in indices:

            if idx == document:
                continue

            results.append(
                (
                    int(idx),
                    float(scores[idx]),
                )
            )

            if len(results) == n:
                break

        return results

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return embeddings as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Embedding matrix.
        """

        embeddings = self.get_embeddings()

        return pd.DataFrame(embeddings)

    def search(
        self,
        query: str,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search for the documents most similar to a text query.

        Parameters
        ----------
        query : str
            Query text.

        n : int, default=10
            Number of documents to return.

        Returns
        -------
        list[tuple[int, float]]
            List of (document_index, similarity_score)
            sorted by decreasing similarity.
        """

        if not query.strip():
            raise ValueError(
                "'query' cannot be empty."
            )

        if n <= 0:
            raise ValueError(
                "'n' must be greater than zero."
            )

        self._load_model()

        assert self._model is not None

        embeddings = self.get_embeddings()

        query_embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).reshape(1, -1)

        similarities = cosine_similarity(
            query_embedding,
            embeddings,
        )[0]

        indices = np.argsort(similarities)[::-1]

        return [
            (
                int(idx),
                float(similarities[idx]),
            )
            for idx in indices[:n]
        ]

    @property
    def model(
        self,
    ):
        """
        Return the underlying SentenceTransformer model.

        Returns
        -------
        SentenceTransformer
            Loaded SentenceTransformer model.
        """

        self._load_model()

        assert self._model is not None

        return self._model
