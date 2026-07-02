"""
PubMed data acquisition.
"""

from litminion.data.api import (
    set_email,
    search_pubmed,
    fetch_pubmed,
)

from litminion.data.downloader import (
    download_pubmed,
)

__all__ = [
    "set_email",
    "search_pubmed",
    "fetch_pubmed",
    "download_pubmed",
]
