"""
LitMinion

A Python framework for biomedical literature mining and natural
language processing.
"""

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
    TfidfExtractor,
    NGramExtractor,
)

from litminion.corpus import Corpus

from litminion.keywords import (
    BaseKeywordExtractor,
    TfidfKeywordExtractor,
)

__version__ = "0.2.0"

__all__ = [

    # configuration
    "set_email",

    # data
    "download_pubmed",

    # corpus
    "Corpus",

    # preprocessing
    "BasePreprocessor",
    "ClassicalPreprocessor",

    # feature extraction
    "WordFrequencyExtractor",
    "TfidfExtractor",
    "NgramExtractor",

    # keyword extraction
    "BaseKeywordExtractor",
    "TfidfKeywordExtractor",

]
