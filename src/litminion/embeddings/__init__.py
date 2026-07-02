"""
Embedding extraction.
"""

from litminion.embeddings.base import (
    BaseEmbeddingExtractor,
)

from litminion.embeddings.sentence_transformers import (
    SentenceTransformerExtractor,
)

__all__ = [
    "BaseEmbeddingExtractor",
    "SentenceTransformerExtractor",
]
