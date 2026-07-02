"""
litminion

A modular Python framework for biomedical literature mining and
natural language processing.
"""

from litminion.corpus import Corpus

from litminion.data import (
    set_email,
    search_pubmed,
    fetch_pubmed,
    download_pubmed,
)

from litminion.preprocessing import (
    BasePreprocessor,
    ClassicalPreprocessor,
)

from litminion.features import (
    WordFrequencyExtractor,
    NGramExtractor,
    TfidfExtractor,
)

from litminion.keywords import (
    BaseKeywordExtractor,
    KeyBERTKeywordExtractor,
)

from litminion.embeddings import (
    BaseEmbeddingExtractor,
    SentenceTransformerExtractor,
)

from litminion.topic_modeling import (
    BaseTopicModel,
    BERTopicModel,
)

__all__ = [
    # data
    "set_email",
    "search_pubmed",
    "fetch_pubmed",
    "download_pubmed",

    # corpus
    "Corpus",

    # preprocessing
    "BasePreprocessor",
    "ClassicalPreprocessor",

    # features
    "WordFrequencyExtractor",
    "NGramExtractor",
    "TfidfExtractor",

    # keywords
    "BaseKeywordExtractor",
    "KeyBERTKeywordExtractor",

    # embeddings
    "BaseEmbeddingExtractor",
    "SentenceTransformerExtractor",

    # topic models
    "BaseTopicModel",
    "BERTopicModel",
]
