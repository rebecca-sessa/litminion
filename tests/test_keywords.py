import pytest

import litminion as lm


@pytest.fixture
def documents():
    return [
        "Janus kinase inhibitors improve rheumatoid arthritis outcomes.",
        "JAK inhibitors reduce inflammation in autoimmune diseases.",
        "Clinical trials evaluated the safety of JAK inhibitors.",
    ]


def test_fit(documents):

    extractor = lm.TfidfKeywordExtractor()

    result = extractor.fit(documents)

    assert result is extractor


def test_get_keywords(documents):

    extractor = lm.TfidfKeywordExtractor()

    extractor.fit(documents)

    keywords = extractor.get_keywords(5)

    assert isinstance(keywords, list)
    assert len(keywords) == 5
    assert isinstance(keywords[0][0], str)
    assert isinstance(keywords[0][1], float)


def test_dataframe(documents):

    extractor = lm.TfidfKeywordExtractor()

    extractor.fit(documents)

    df = extractor.to_dataframe(5)

    assert list(df.columns) == ["Keyword", "Score"]
    assert len(df) == 5


def test_invalid_n(documents):

    extractor = lm.TfidfKeywordExtractor()

    extractor.fit(documents)

    with pytest.raises(ValueError):
        extractor.get_keywords(0)


def test_not_fitted_keywords():

    extractor = lm.TfidfKeywordExtractor()

    with pytest.raises(ValueError):
        extractor.get_keywords()


def test_not_fitted_dataframe():

    extractor = lm.TfidfKeywordExtractor()

    with pytest.raises(ValueError):
        extractor.to_dataframe()


def test_scores_sorted(documents):

    extractor = lm.TfidfKeywordExtractor()

    extractor.fit(documents)

    keywords = extractor.get_keywords(10)

    scores = [score for _, score in keywords]

    assert scores == sorted(scores, reverse=True)


def test_plot(documents):

    extractor = lm.TfidfKeywordExtractor()

    extractor.fit(documents)

    ax = extractor.plot(5)

    assert ax is not None
