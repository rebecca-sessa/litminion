import pandas as pd
import pytest

import litminion as lm


@pytest.fixture
def dataframe():

    return pd.DataFrame(
        {
            "PMID": [1, 2],
            "Title": [
                "Paper 1",
                "Paper 2",
            ],
            "Abstract": [
                "Janus kinase inhibitors improve rheumatoid arthritis.",
                "Clinical trials evaluated JAK inhibitors.",
            ],
            "Journal": [
                "Journal A",
                "Journal B",
            ],
            "Year": [
                2024,
                2025,
            ],
            "Authors": [
                "Author A",
                "Author B",
            ],
            "PublicationType": [
                "Journal Article",
                "Review",
            ],
        }
    )


@pytest.fixture
def corpus(dataframe):

    return lm.Corpus(dataframe)


def test_length(corpus):

    assert len(corpus) == 2


def test_shape(corpus):

    assert corpus.shape == (2, 7)


def test_columns(corpus):

    assert list(corpus.columns) == [
        "PMID",
        "Title",
        "Abstract",
        "Journal",
        "Year",
        "Authors",
        "PublicationType",
    ]


def test_head(corpus):

    head = corpus.head()

    assert isinstance(head, pd.DataFrame)
    assert len(head) == 2


def test_to_dataframe(corpus):

    df = corpus.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.equals(corpus.data)


def test_getitem(corpus):

    years = corpus["Year"]

    assert list(years) == [2024, 2025]


def test_get_documents_before_preprocessing(corpus):

    docs = corpus.get_documents()

    assert docs == corpus.abstracts


def test_preprocess(corpus):

    preprocessor = lm.ClassicalPreprocessor()

    corpus.preprocess(preprocessor)

    assert corpus.processed_abstracts is not None
    assert len(corpus.processed_abstracts) == 2


def test_get_documents_after_preprocessing(corpus):

    preprocessor = lm.ClassicalPreprocessor()

    corpus.preprocess(preprocessor)

    docs = corpus.get_documents()

    assert docs == corpus.processed_abstracts


def test_copy(corpus):

    copied = corpus.copy()

    assert copied is not corpus

    assert copied.data.equals(corpus.data)


def test_copy_is_independent(corpus):

    copied = corpus.copy()

    copied.data.loc[0, "Title"] = "Modified"

    assert corpus.data.loc[0, "Title"] == "Paper 1"


def test_repr(corpus):

    representation = repr(corpus)

    assert "Corpus" in representation
    assert "documents=2" in representation
