# LitMinion

> A Python framework for biomedical literature mining and natural language processing.

LitMinion is an open-source Python framework designed to retrieve, preprocess, analyze, and explore biomedical literature from PubMed. The project combines modern software engineering practices with natural language processing (NLP) techniques to support scientific knowledge discovery in academia and industry.

---

## Overview

Biomedical literature is growing at an unprecedented rate, making manual exploration increasingly difficult. LitMinion aims to provide a modular toolkit for:

- Retrieving publications from PubMed
- Preprocessing biomedical text
- Extracting keywords and entities
- Discovering topics and trends
- Building reproducible literature mining workflows

The framework is designed to be extensible, allowing multiple preprocessing strategies and machine learning pipelines to coexist under a common interface.

---

## Current Features

- PubMed search through the NCBI Entrez API
- Retrieval of article metadata and abstracts
- Parsing PubMed XML into pandas DataFrames
- Object-oriented preprocessing architecture
- Classical NLP preprocessing with spaCy
- Lazy loading of NLP models
- Type hints and comprehensive docstrings

---

## Planned Features

### Data management

- Corpus class
- Dataset serialization
- Metadata management

### NLP

- Biomedical preprocessing (scispaCy)
- Transformer preprocessing
- Keyword extraction
- TF-IDF
- YAKE
- KeyBERT

### Machine Learning

- Topic modeling (BERTopic)
- Document embeddings
- Abstract classification
- Trend analysis

### Biomedical NLP

- Named Entity Recognition
- Gene and protein recognition
- Drug extraction
- Disease extraction

### Visualization

- Topic visualizations
- Publication trends
- Interactive dashboards

---

## Project Structure

```text
litminion/
│
├── src/
│   └── litminion/
│       ├── data/
│       ├── preprocessing/
│       ├── config.py
│       └── __init__.py
│
├── tests/
├── notebooks/
├── storage/
│   ├── raw/
│   └── processed/
│
├── outputs/
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/litminion.git
cd litminion
```

Create a virtual environment (recommended)

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

## Quick Start

```python
import litminion as lm

lm.set_email("your_email@example.com")

df = lm.download_pubmed(
    query="JAK inhibitor",
    max_results=20,
)

preprocessor = lm.ClassicalPreprocessor()

processed = preprocessor.transform(
    df.loc[0, "Abstract"]
)

print(processed)
```

---

## Development principles

- Modular architecture
- Object-oriented design
- Type annotations
- Comprehensive documentation
- Unit testing
- Reproducible analyses

---

## Roadmap

- [x] PubMed API
- [x] XML parser
- [x] Downloader
- [x] Classical preprocessing
- [ ] Corpus abstraction
- [ ] Keyword extraction
- [ ] Biomedical preprocessing
- [ ] Topic modeling
- [ ] Named entity recognition
- [ ] Trend analysis
- [ ] Interactive dashboard