import tempfile

import pandas as pd
import plotly.graph_objects as go
import pytest

import litminion as lm


@pytest.fixture
def documents():

    return [
        "Janus kinase inhibitors improve rheumatoid arthritis outcomes.",
        "JAK inhibitors reduce inflammation in autoimmune diseases.",
        "Clinical trials evaluated the safety of JAK inhibitors.",
        "Upadacitinib demonstrated efficacy in ulcerative colitis.",
        "Tofacitinib improved disease activity in rheumatoid arthritis.",
        "Baricitinib is approved for atopic dermatitis.",
        "Filgotinib selectively inhibits JAK1.",
        "Inflammatory bowel disease includes Crohn disease.",
        "Ulcerative colitis affects the colon.",
        "Immune-mediated diseases involve cytokine signaling.",
        "Interleukin six activates the JAK STAT pathway.",
        "Targeted therapies improve patient outcomes.",
        "Autoimmune diseases require long-term treatment.",
        "Biologic drugs inhibit inflammatory pathways.",
        "Selective kinase inhibition reduces toxicity.",
        "Clinical studies evaluate efficacy and safety.",
        "Precision medicine guides therapeutic decisions.",
        "Synovial inflammation causes joint damage.",
        "Patients require monitoring during therapy.",
        "Modern immunology focuses on targeted treatments.",
    ]


# constructor


def test_constructor():

    model = lm.BERTopicModel()

    assert model.documents == []
    assert model.topics is None
    assert model.probabilities is None
    assert model._model is None


def test_default_embedding():

    model = lm.BERTopicModel()

    assert isinstance(
        model.embedding_extractor,
        lm.SentenceTransformerExtractor,
    )


def test_custom_embedding():

    embedding = lm.SentenceTransformerExtractor()

    model = lm.BERTopicModel(
        embedding_extractor=embedding,
    )

    assert model.embedding_extractor is embedding


def test_parameters():

    model = lm.BERTopicModel(
        calculate_probabilities=False,
        verbose=True,
    )

    assert model.calculate_probabilities is False
    assert model.verbose is True


# validation


def test_load_model():

    model = lm.BERTopicModel()

    model._load_model()

    assert model._model is not None


def test_check_fitted():

    model = lm.BERTopicModel()

    with pytest.raises(ValueError):
        model._check_fitted()


def test_validate_document():

    model = lm.BERTopicModel()

    with pytest.raises(ValueError):
        model._validate_document(0)


def test_validate_topic():

    model = lm.BERTopicModel()

    with pytest.raises(ValueError):
        model._validate_topic(0)


# fitting


def test_fit(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    assert len(model.documents) == len(documents)
    assert len(model.topics) == len(documents)


def test_fit_transform(documents):

    model = lm.BERTopicModel()

    topics = model.fit_transform(documents)

    assert topics == model.topics


def test_transform(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    topics = model.transform(documents)

    assert len(topics) == len(documents)


def test_transform_not_fitted(documents):

    model = lm.BERTopicModel()

    with pytest.raises(ValueError):
        model.transform(documents)


# retrieval


def test_get_topics(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    topics = model.get_topics()

    assert isinstance(topics, dict)


def test_get_topic(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    topic = next(iter(model.get_topics()))

    words = model.get_topic(topic)

    assert isinstance(words, list)


def test_get_topic_info(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    df = model.get_topic_info()

    assert isinstance(df, pd.DataFrame)


def test_get_document_topics(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    df = model.get_document_topics()

    assert isinstance(df, pd.DataFrame)

    assert len(df) == len(documents)


def test_invalid_topic(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    with pytest.raises(ValueError):
        model.get_topic(999)


# plotting


def test_plot_topics(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    assert isinstance(
        model.plot_topics(),
        go.Figure,
    )


def test_plot_topic_hierarchy(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    assert isinstance(
        model.plot_topic_hierarchy(),
        go.Figure,
    )


def test_plot_topic_similarity(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    assert isinstance(
        model.plot_topic_similarity(),
        go.Figure,
    )


def test_plot_topic_sizes(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    assert isinstance(
        model.plot_topic_sizes(),
        go.Figure,
    )


# save/load


def test_save_load(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    with tempfile.TemporaryDirectory() as directory:

        model.save(directory)

        loaded = lm.BERTopicModel.load(directory)

        assert isinstance(
            loaded,
            lm.BERTopicModel,
        )


# magic methods


def test_len(documents):

    model = lm.BERTopicModel()

    model.fit(documents)

    assert len(model) == len(documents)


def test_repr():

    model = lm.BERTopicModel()

    assert "BERTopicModel" in repr(model)
