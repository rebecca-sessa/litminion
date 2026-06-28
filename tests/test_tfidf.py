import pandas as pd

import pytest

import litminion as lm


def test_fit_transform(documents):

    extractor = lm.TfidfExtractor()

    matrix = extractor.fit_transform(documents)

    assert matrix.shape[0] == len(documents)


def test_feature_names(documents):

    extractor = lm.TfidfExtractor()

    extractor.fit(documents)

    assert len(extractor.get_feature_names()) > 0


def test_vocabulary_size(documents):

    extractor = lm.TfidfExtractor()

    extractor.fit(documents)

    assert extractor.get_vocabulary_size() > 0


def test_dataframe(documents):

    extractor = lm.TfidfExtractor()

    extractor.fit(documents)

    df = extractor.to_dataframe(document=0)

    assert isinstance(df, pd.DataFrame)


def test_invalid_document(documents):

    extractor = lm.TfidfExtractor()

    extractor.fit(documents)

    with pytest.raises(IndexError):

        extractor.get_top_terms(100)


def test_invalid_n(documents):

    extractor = lm.TfidfExtractor()

    extractor.fit(documents)

    with pytest.raises(ValueError):

        extractor.get_top_terms(document=0, n=0)


def test_not_fitted():

    extractor = lm.TfidfExtractor()

    with pytest.raises(ValueError):

        extractor.transform(["hello"])
