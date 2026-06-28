import pandas as pd

import pytest

import litminion as lm


def test_fit_transform(documents):

    extractor = lm.NGramExtractor()

    matrix = extractor.fit_transform(documents)

    assert matrix.shape[0] == len(documents)


def test_feature_names(documents):

    extractor = lm.NGramExtractor()

    extractor.fit(documents)

    assert len(extractor.get_feature_names()) > 0


def test_vocabulary_size(documents):

    extractor = lm.NGramExtractor()

    extractor.fit(documents)

    assert extractor.get_vocabulary_size() > 0


def test_ngram_counts(documents):

    extractor = lm.NGramExtractor()

    extractor.fit(documents)

    counts = extractor.get_ngram_counts()

    assert isinstance(counts, dict)

    assert len(counts) > 0


def test_top_ngrams(documents):

    extractor = lm.NGramExtractor()

    extractor.fit(documents)

    top = extractor.get_top_ngrams(5)

    assert len(top) == 5

    assert isinstance(top[0], tuple)


def test_dataframe(documents):

    extractor = lm.NGramExtractor()

    extractor.fit(documents)

    df = extractor.to_dataframe()

    assert isinstance(df, pd.DataFrame)

    assert list(df.columns) == ["N-gram", "Count"]


def test_invalid_n(documents):

    extractor = lm.NGramExtractor()

    extractor.fit(documents)

    with pytest.raises(ValueError):

        extractor.get_top_ngrams(0)


def test_not_fitted():

    extractor = lm.NGramExtractor()

    with pytest.raises(ValueError):

        extractor.transform(["hello world"])
