import litminion as lm


def test_transform_returns_string():

    preprocessor = lm.ClassicalPreprocessor()

    text = "Patients received JAK inhibitors."

    processed = preprocessor.transform(text)

    assert isinstance(processed, str)


def test_transform_corpus(documents):

    preprocessor = lm.ClassicalPreprocessor()

    processed = preprocessor.transform_corpus(documents)

    assert isinstance(processed, list)

    assert len(processed) == len(documents)


def test_empty_document():

    preprocessor = lm.ClassicalPreprocessor()

    assert preprocessor.transform("") == ""
