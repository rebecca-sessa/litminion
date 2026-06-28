"""
config.py

Central configuration for LitMinion.

This module defines default values used throughout the package.
Changing a configuration here propagates automatically to all
components that rely on it.
"""

# Package
PACKAGE_NAME = "litminion"
PACKAGE_VERSION = "0.1.0"

# PubMed
DEFAULT_MAX_RESULTS = 100

# NLP
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_LANGUAGE = "en"

# Randomness
DEFAULT_RANDOM_STATE = 42
