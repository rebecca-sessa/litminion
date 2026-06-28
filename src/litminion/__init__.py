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

__version__ = "0.1.0"

__all__ = [
    "set_email",
    "search_pubmed",
    "fetch_pubmed",
    "download_pubmed",
    "BasePreprocessor",
    "ClassicalPreprocessor",
]
