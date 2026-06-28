"""
Tests for the preprocessing module.
"""

from litminion import ClassicalPreprocessor


def test_preprocessor_instantiation():
    preprocessor = ClassicalPreprocessor()
    assert preprocessor is not None


def test_transform_returns_string():
    preprocessor = ClassicalPreprocessor()

    result = preprocessor.transform(
        "Patients were treated with baricitinib."
    )

    assert isinstance(result, str)


def test_cleaning():
    preprocessor = ClassicalPreprocessor()

    text = "  Hello\n\nworld   "

    assert preprocessor._clean_text(text) == "Hello world"
