"""
Grammatical Transformers - Chomsky's Universal Grammar for Neural Networks

Implements Universal Grammar principles into Hugging Face Transformers.

Key Insight (O(1)):
    Clean Architecture IS grammatical mapping
    Grammar IS compression - no overhead, just revealing structure

Architecture:
    Domain (chomsky/):      Grammatical structures (SyntacticObject, Constituent)
    Data (chomsky/):        Operations (Merge, Parse, Symmetry)
    Infrastructure (models/): Transformer implementations (GrammaticalBERT)

Usage:
    from grammatical_transformers import (
        GrammaticalBertModel,
        GrammaticalBertConfig,
        ChomskyParser
    )

    # Create model
    config = GrammaticalBertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        constituency_penalty=0.5
    )

    model = GrammaticalBertModel(config)

    # Use like standard BERT
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
"""

__version__ = "0.1.0"
__author__ = "Thiago Butignon"
__description__ = "Universal Grammar for Transformers"

# Domain: Grammatical structures
from .chomsky.structures import (
    SyntacticObject,
    Constituent,
    ConstituencyTree,
    FeatureBundle,
    MergeResult,
    SymmetryMetrics,
)

# Data: Operations
from .chomsky.parser import (
    ChomskyParser,
    MergeOperation,
    ConstituencyParser,
)

from .chomsky.symmetry import (
    SymmetryMeasure,
    GrammaticalSymmetry,
    compute_symmetry,
    SymmetryLoss,
)

# Infrastructure: Models
from .models.attention import (
    GrammaticalAttention,
    ConstituencyAwareAttention,
    GrammaticalMultiHeadAttention,
)

from .models.grammatical_bert import (
    GrammaticalBertConfig,
    GrammaticalBertModel,
    GrammaticalBertForSequenceClassification,
    GrammaticalBertForTokenClassification,
)

__all__ = [
    # Structures (Domain - Nouns)
    "SyntacticObject",
    "Constituent",
    "ConstituencyTree",
    "FeatureBundle",
    "MergeResult",
    "SymmetryMetrics",

    # Operations (Data - Verbs)
    "ChomskyParser",
    "MergeOperation",
    "ConstituencyParser",
    "SymmetryMeasure",
    "GrammaticalSymmetry",
    "compute_symmetry",
    "SymmetryLoss",

    # Models (Infrastructure - Implementations)
    "GrammaticalAttention",
    "ConstituencyAwareAttention",
    "GrammaticalMultiHeadAttention",
    "GrammaticalBertConfig",
    "GrammaticalBertModel",
    "GrammaticalBertForSequenceClassification",
    "GrammaticalBertForTokenClassification",
]
