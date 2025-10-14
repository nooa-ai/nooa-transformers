"""
Grammatical Models - Infrastructure Layer (INFRA-001 Pattern)

Linguistic Role: ADVERBS - Specific implementations using external technology
Clean Architecture: Infrastructure adapters for Hugging Face Transformers

This module provides constituency-aware attention mechanisms
that respect grammatical boundaries.
"""

from .attention import (
    GrammaticalAttention,
    ConstituencyAwareAttention,
    GrammaticalMultiHeadAttention
)

from .grammatical_bert import (
    GrammaticalBertConfig,
    GrammaticalBertModel,
    GrammaticalBertForSequenceClassification,
    GrammaticalBertForTokenClassification
)

__all__ = [
    # Attention mechanisms
    "GrammaticalAttention",
    "ConstituencyAwareAttention",
    "GrammaticalMultiHeadAttention",

    # Model implementations
    "GrammaticalBertConfig",
    "GrammaticalBertModel",
    "GrammaticalBertForSequenceClassification",
    "GrammaticalBertForTokenClassification",
]
