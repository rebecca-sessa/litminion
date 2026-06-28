"""

parser.py

Utilities for converting raw PubMed records into structured Python

dictionaries suitable for downstream analysis.

"""

from typing import Any


def _extract_abstract(article_info: Any) -> str:
    """

    Extract and concatenate the abstract text.

    Parameters

    ----------

    article_info

        PubMed Article element.

    Returns

    -------

    str

        Full abstract as a single string.

    """

    if "Abstract" not in article_info:

        return ""

    sections = article_info["Abstract"]["AbstractText"]

    return " ".join(str(section) for section in sections)


def _extract_authors(article_info: Any) -> str:
    """

    Extract authors as a semicolon-separated string.

    Parameters

    ----------

    article_info

        PubMed Article element.

    Returns

    -------

    str

        Authors formatted as:

        'John Smith; Jane Doe'

    """

    authors = []

    for author in article_info.get("AuthorList", []):

        first = author.get("ForeName", "")

        last = author.get("LastName", "")

        full_name = f"{first} {last}".strip()

        if full_name:

            authors.append(full_name)

    return "; ".join(authors)


def parse_article(article: Any) -> dict[str, str]:
    """

    Parse a PubMed article into a structured dictionary.

    Parameters

    ----------

    article

        Raw PubMed article returned by Bio.Entrez.

    Returns

    -------

    dict[str, str]

        Parsed article metadata.

    """

    medline = article["MedlineCitation"]

    article_info = medline["Article"]

    publication_types = "; ".join(

        str(pt)

        for pt in article_info.get("PublicationTypeList", [])

    )

    year = (

        article_info["Journal"]

        .get("JournalIssue", {})

        .get("PubDate", {})

        .get("Year", "")

    )

    return {

        "PMID": str(medline["PMID"]),

        "Title": str(article_info.get("ArticleTitle", "")),

        "Abstract": _extract_abstract(article_info),

        "Journal": str(article_info["Journal"].get("Title", "")),

        "Year": year,

        "Authors": _extract_authors(article_info),

        "PublicationType": publication_types,

    }
