"""

downloader.py

High-level interface for downloading PubMed articles as a pandas DataFrame.

This module orchestrates the PubMed search, article retrieval,

and parsing pipeline.

"""

from __future__ import annotations

import pandas as pd

from litminion.data.api import fetch_pubmed, search_pubmed
from litminion.data.parser import parse_article


def download_pubmed(
        query: str,
        max_results: int = 100,
) -> pd.DataFrame:
    """

    Download PubMed articles matching a search query.

    Parameters

    ----------

    query : str

        PubMed search query.

    max_results : int, default=100

        Maximum number of articles to retrieve.

    Returns

    -------

    pandas.DataFrame

        DataFrame containing one row per article.

    """

    pmids = search_pubmed(
        query=query,
        max_results=max_results
    )

    articles = fetch_pubmed(pmids)

    parsed_articles = [
        parse_article(article)
        for article in articles
    ]

    return pd.DataFrame(parsed_articles)
