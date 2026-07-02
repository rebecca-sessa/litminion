import numpy as np
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

    extractor = lm.SentenceTransformerExtractor()

    result = extractor.fit(documents)

    assert result is extractor


def test_fit_transform(documents):

    extractor = lm.SentenceTransformerExtractor()

    embeddings = extractor.fit_transform(documents)

    assert isinstance(embeddings, np.ndarray)

    assert embeddings.shape[0] == len(documents)


def test_transform(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit(documents)

    embeddings = extractor.transform(documents)

    assert isinstance(embeddings, np.ndarray)


def test_get_embeddings(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    embeddings = extractor.get_embeddings()

    assert isinstance(embeddings, np.ndarray)


def test_shape(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    assert extractor.shape == extractor.get_embeddings().shape


def test_embedding_dimension(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    assert extractor.get_embedding_dimension() > 0


def test_n_documents(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    assert extractor.get_n_documents() == len(documents)


def test_single_embedding(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    embedding = extractor.get_embedding(0)

    assert isinstance(embedding, np.ndarray)

    assert embedding.ndim == 1

    assert len(embedding) == extractor.get_embedding_dimension()


def test_similarity(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    score = extractor.similarity(
        0,
        1,
    )

    assert isinstance(score, float)

    assert -1.0 <= score <= 1.0


def test_similarity_matrix(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    matrix = extractor.similarity_matrix()

    assert isinstance(matrix, np.ndarray)

    assert matrix.shape == (
        len(documents),
        len(documents),
    )


def test_similarity_matrix_diagonal(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    matrix = extractor.similarity_matrix()

    assert np.allclose(
        np.diag(matrix),
        1.0,
    )


def test_similarity_matrix_is_symmetric(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    matrix = extractor.similarity_matrix()

    assert np.allclose(
        matrix,
        matrix.T,
    )


def test_most_similar(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    results = extractor.most_similar(
        0,
        n=2,
    )

    assert len(results) == 2

    assert isinstance(results[0][0], int)

    assert isinstance(results[0][1], float)


def test_search(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    results = extractor.search(
        "JAK inhibitors",
        n=2,
    )

    assert len(results) == 2

    assert isinstance(results[0][0], int)

    assert isinstance(results[0][1], float)


def test_search_is_sorted(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    results = extractor.search(
        "JAK inhibitor",
        n=3,
    )

    scores = [
        score
        for _, score in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_dataframe(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    df = extractor.to_dataframe()

    assert df.shape == extractor.shape


def test_invalid_document(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    with pytest.raises(IndexError):

        extractor.get_embedding(100)


def test_invalid_n(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    with pytest.raises(ValueError):

        extractor.most_similar(
            0,
            n=0,
        )


def test_invalid_query(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    with pytest.raises(ValueError):

        extractor.search(
            "",
            n=2,
        )


def test_search_invalid_n(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    with pytest.raises(ValueError):

        extractor.search(
            "JAK inhibitor",
            n=0,
        )


def test_not_fitted():

    extractor = lm.SentenceTransformerExtractor()

    with pytest.raises(ValueError):

        extractor.get_embeddings()


def test_search_not_fitted():

    extractor = lm.SentenceTransformerExtractor()

    with pytest.raises(ValueError):

        extractor.search(
            "JAK inhibitor",
        )


def test_len(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    assert len(extractor) == len(documents)


def test_self_similarity(documents):

    extractor = lm.SentenceTransformerExtractor()

    extractor.fit_transform(documents)

    score = extractor.similarity(
        0,
        0,
    )

    assert np.isclose(score, 1.0)
