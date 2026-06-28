"""
Tests for the downloader module.
"""

import pandas as pd

import litminion as lm


EXPECTED_COLUMNS = {
    "PMID",
    "Title",
    "Abstract",
    "Journal",
    "Year",
    "Authors",
    "PublicationType",
}


def test_download_pubmed_returns_dataframe():
    """
    download_pubmed should return a non-empty DataFrame
    with the expected columns.
    """

    lm.set_email("your_email@example.com")

    df = lm.download_pubmed(
        query="JAK inhibitor",
        max_results=5,
    )

    assert isinstance(df, pd.DataFrame)

    assert not df.empty

    assert EXPECTED_COLUMNS.issubset(df.columns)
