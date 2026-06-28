"""
Feature extraction methods for biomedical text.

This package contains algorithms for extracting numerical and
statistical features from preprocessed biomedical documents.
"""

from litminion.features.word_frequency import WordFrequencyAnalyzer
from litminion.features.tfidf import TfidfExtractor

__all__ = [
    "WordFrequencyAnalyzer",
    "TfidfExtractor",
]
