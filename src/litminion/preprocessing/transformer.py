"""
transformer.py

Transformer-based preprocessing.

This module will contain preprocessing pipelines based on
Hugging Face transformers and biomedical language models.
"""

from __future__ import annotations

from litminion.preprocessing.base import BasePreprocessor


class TransformerPreprocessor(BasePreprocessor):
    """
    Base class for transformer-based preprocessing.

    Notes
    -----
    This class is currently under development.
    """

    def transform(
        self,
        text: str,
    ) -> str:
        """
        Transform a document using a transformer model.
        """

        raise NotImplementedError(
            "TransformerPreprocessor is not implemented yet."
        )
