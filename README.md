# 🤖 litminion

<p align="center">
  <img src="assets/logo.svg" width="120" alt="litminion logo">
</p>

<p align="center">
  <img src="assets/cover.png" alt="litminion cover">
</p>

<p align="center">
<b>A modular Python framework for biomedical literature mining and natural language processing.</b>
</p>

<p align="center">

Python 3.11+ • Alpha

</p>

---

# Overview

**litminion** is an open-source Python framework for biomedical literature mining and natural language processing.

Starting from PubMed publications, it provides reusable building blocks for retrieving scientific articles, preprocessing biomedical text, extracting statistical features, and supporting downstream NLP analyses.

Rather than offering monolithic workflows, **litminion** is designed around modular, interchangeable components that can be combined into reproducible literature-mining pipelines.

Current capabilities include:

- PubMed retrieval through the NCBI Entrez API
- Biomedical text preprocessing
- Word frequency analysis
- TF-IDF feature extraction
- N-gram extraction
- Visualization utilities

Future releases will expand the framework with semantic embeddings, keyword extraction, topic modeling, biomedical named entity recognition, and document exploration.

---

# Features

## Data acquisition

- PubMed search via the NCBI Entrez API
- Metadata retrieval
- Abstract retrieval
- XML parsing
- Download directly into pandas DataFrames

## Text preprocessing

- Classical preprocessing with spaCy
- Lemmatization
- Stopword removal
- Punctuation filtering
- Lazy loading of language models
- Batch corpus preprocessing

## Feature extraction

- Word frequency extraction
- TF-IDF vectorization
- N-gram extraction
- DataFrame export
- Built-in visualization

## Software engineering

- Modular package architecture
- Object-oriented design
- Type annotations
- Comprehensive documentation
- Automated testing with pytest
- Consistent public API

---

# Installation

Clone the repository

```bash
git clone https://github.com/rebecca-sessa/litminion.git
cd litminion
```

Create a virtual environment

```bash
conda create -n litminion python=3.11
conda activate litminion
```

Install the package

```bash
pip install -e .
```

Download the default spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

# Quick start

```python
import litminion as lm

lm.set_email("your_email@example.com")

# Download PubMed abstracts
df = lm.download_pubmed(
    query="JAK inhibitor",
    max_results=100,
)

# Preprocess the corpus
preprocessor = lm.ClassicalPreprocessor()

documents = preprocessor.transform_corpus(
    df["Abstract"]
)

# Extract TF-IDF features
tfidf = lm.TfidfExtractor()

tfidf.fit(documents)

print(tfidf.get_top_terms(document=0))

tfidf.plot(document=0)

# Extract Keywords
keywords = lm.TfidfKeywordExtractor()

keywords.fit(

    corpus.get_documents()

)

keywords.get_keywords(15)
```

---

# Project structure

```text
litminion/
│
├── assets/
├── notebooks/
├── tests/
├── src/
│   └── litminion/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── config.py
│       └── __init__.py
│
├── outputs/
├── storage/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

# Design principles

The architecture of **litminion** is guided by a small set of engineering principles:

- Modular architecture
- Object-oriented design
- Reusable components
- Extensible NLP pipelines
- Consistent public APIs
- Strong typing
- Comprehensive documentation
- Automated testing
- Reproducible scientific workflows

---

# Roadmap

## Completed

- ✅ PubMed API integration
- ✅ XML parser
- ✅ Literature downloader
- ✅ Classical preprocessing pipeline
- ✅ Word frequency extraction
- ✅ TF-IDF feature extraction
- ✅ N-gram extraction
- ✅ Built-in visualization
- ✅ Automated unit tests

## In progress

- ⏳ Corpus abstraction
- ⏳ Biomedical preprocessing with scispaCy

## Planned

### Natural language processing

- Keyword extraction
- Sentence embeddings
- Semantic similarity
- Document clustering
- BERTopic integration

### Biomedical NLP

- Named entity recognition
- Entity linking
- Drug recognition
- Disease recognition
- Gene and protein recognition

### Visualization

- Publication trends
- Topic visualization
- Interactive dashboards

---

# Development status

**litminion** is currently in **Alpha (v0.2.0)**.

The project provides a stable foundation for biomedical literature retrieval and classical NLP. Development is now focused on expanding support for modern biomedical NLP methods while maintaining a clean, consistent, and extensible API.

Contributions, suggestions, and issue reports are welcome.