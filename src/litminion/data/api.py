"""

api.py

Functions for communicating with the PubMed database using the NCBI Entrez API.

This module is responsible only for interacting with PubMed.

It does not parse articles or build pandas DataFrames.

"""

from collections.abc import Sequence
from Bio import Entrez


def set_email(email: str) -> None:
    """

   Configure the email address required by the NCBI Entrez API.

   Parameters

   ----------

   email : str

       Email address used to identify the user when making requests

       to the PubMed API.

   Returns

   -------

   None

   """
    Entrez.email = email


def search_pubmed(query: str, max_results: int = 100) -> Sequence[str]:
    """

    Search PubMed and return a list of PubMed IDs (PMIDs).

    Parameters

    ----------

    query : str

        Search query using PubMed syntax.

    max_results : int, default=100

        Maximum number of PMIDs to retrieve.

    Returns

    -------

    Sequence[str]

        List of PubMed IDs.

    """

    handle = Entrez.esearch(

        db="pubmed",

        term=query,

        retmax=max_results,

    )

    results = Entrez.read(handle)

    handle.close()

    return results["IdList"]


def fetch_pubmed(pmids: Sequence[str]) -> Sequence:
    """

    Retrieve PubMed articles from a list of PMIDs.

    Parameters

    ----------

    pmids : list[str]

        List of PubMed IDs.

    Returns

    -------

    list

        List containing PubMedArticle objects returned by Entrez.

    """

    ids = ",".join(pmids)

    handle = Entrez.efetch(

        db="pubmed",

        id=ids,

        rettype="abstract",

        retmode="xml",

    )

    records = Entrez.read(handle)

    handle.close()

    return records["PubmedArticle"]
