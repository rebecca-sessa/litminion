import litminion as lm


def test_public_api():

    assert hasattr(lm, "download_pubmed")

    assert hasattr(lm, "ClassicalPreprocessor")

    assert hasattr(lm, "WordFrequencyExtractor")

    assert hasattr(lm, "TfidfExtractor")

    assert hasattr(lm, "NGramExtractor")
