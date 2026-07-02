import matplotlib.axes
import pandas as pd
import pytest

import litminion as lm


@pytest.fixture
def documents():

    return [
        (
            "Janus kinase inhibitors improve rheumatoid arthritis "
            "outcomes and reduce inflammation."
        ),
        (
            "Clinical trials evaluated the safety of JAK inhibitors "
            "in autoimmune diseases."
        ),
        (
            "Upadacitinib demonstrated efficacy in ulcerative colitis "
            "patients with moderate disease."
        ),
    ]


def test_fit(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    result = extractor.fit(documents)

    assert result is extractor


def test_transform(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    keywords = extractor.transform()

    assert isinstance(keywords, list)

    assert len(keywords) == len(documents)

    assert all(
        isinstance(document, list)
        for document in keywords
    )


def test_fit_transform(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    keywords = extractor.fit_transform(documents)

    assert isinstance(keywords, list)

    assert len(keywords) == len(documents)

    assert keywords == extractor.transform()


def test_get_keywords(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    keywords = extractor.get_keywords(
        document=0,
        n=5,
    )

    assert isinstance(keywords, list)

    assert len(keywords) == 5

    assert all(
        isinstance(keyword, tuple)
        for keyword in keywords
    )

    assert isinstance(
        keywords[0][0],
        str,
    )

    assert isinstance(
        keywords[0][1],
        float,
    )


def test_dataframe(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    df = extractor.to_dataframe(
        document=0,
        n=5,
    )

    assert isinstance(df, pd.DataFrame)

    assert df.shape == (5, 2)

    assert list(df.columns) == [
        "Keyword",
        "Score",
    ]


def test_plot(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    ax = extractor.plot(
        document=0,
        n=5,
    )

    assert isinstance(
        ax,
        matplotlib.axes.Axes,
    )


def test_len(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    assert len(extractor) == len(documents)


def test_repr_before_fit():

    extractor = lm.KeyBERTKeywordExtractor()

    assert repr(extractor) == (
        "KeyBERTKeywordExtractor("
        "documents=0, "
        "fitted=False)"
    )


def test_repr_after_fit(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    assert repr(extractor) == (
        "KeyBERTKeywordExtractor("
        "documents=3, "
        "fitted=True)"
    )


def test_invalid_document(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    with pytest.raises(IndexError):

        extractor.get_keywords(
            document=100,
        )


def test_invalid_n(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    with pytest.raises(ValueError):

        extractor.get_keywords(
            document=0,
            n=0,
        )


def test_not_fitted_transform():

    extractor = lm.KeyBERTKeywordExtractor()

    with pytest.raises(ValueError):

        extractor.transform()


def test_not_fitted_get_keywords():

    extractor = lm.KeyBERTKeywordExtractor()

    with pytest.raises(ValueError):

        extractor.get_keywords()


def test_not_fitted_dataframe():

    extractor = lm.KeyBERTKeywordExtractor()

    with pytest.raises(ValueError):

        extractor.to_dataframe()


def test_not_fitted_plot():

    extractor = lm.KeyBERTKeywordExtractor()

    with pytest.raises(ValueError):

        extractor.plot()


def test_keyword_scores_are_sorted(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    keywords = extractor.get_keywords(
        document=0,
        n=5,
    )

    scores = [
        score
        for _, score in keywords
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_top_n_parameter(documents):

    extractor = lm.KeyBERTKeywordExtractor(
        top_n=10,
    )

    extractor.fit(documents)

    keywords = extractor.get_keywords(
        document=0,
        n=5,
    )

    assert len(keywords) == 5


def test_empty_documents():

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit([])

    assert len(extractor) == 0

    assert extractor.transform() == []


def test_transform_returns_cached_results(documents):

    extractor = lm.KeyBERTKeywordExtractor()

    extractor.fit(documents)

    first = extractor.transform()

    second = extractor.transform()

    assert first == second
