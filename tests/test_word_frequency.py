import pandas as pd

import litminion as lm


def test_fit(documents):

    analyzer = lm.WordFrequencyExtractor()

    analyzer.fit(documents)

    assert analyzer.frequencies is not None


def test_vocabulary_size(documents):

    analyzer = lm.WordFrequencyExtractor()

    analyzer.fit(documents)

    assert analyzer.get_vocabulary_size() > 0


def test_top_words(documents):

    analyzer = lm.WordFrequencyExtractor()

    analyzer.fit(documents)

    words = analyzer.get_top_words(5)

    assert len(words) == 5

    assert isinstance(words[0], tuple)


def test_dataframe(documents):

    analyzer = lm.WordFrequencyExtractor()

    analyzer.fit(documents)

    df = analyzer.to_dataframe()

    assert isinstance(df, pd.DataFrame)

    assert list(df.columns) == ["Word", "Count"]


def test_invalid_n(documents):

    analyzer = lm.WordFrequencyExtractor()

    analyzer.fit(documents)

    import pytest

    with pytest.raises(ValueError):

        analyzer.get_top_words(0)
