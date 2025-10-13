"""
Chomsky Grammar Module for Transformers

This module implements Chomsky's Universal Grammar principles
into transformer architectures using Clean Architecture patterns.

Linguistic Mapping:
- Domain: Grammatical structures (Merge, Constituency)
- Data: Parsing operations
- Infrastructure: PyTorch tensor operations
"""

from .structures import (
    SyntacticObject,
    Constituent,
    ConstituencyTree,
    FeatureBundle,
    MergeResult
)

from .parser import (
    ChomskyParser,
    MergeOperation,
    ConstituencyParser
)

from .symmetry import (
    SymmetryMeasure,
    GrammaticalSymmetry,
    compute_symmetry
)

__all__ = [
    # Structures (Domain - Nouns)
    "SyntacticObject",
    "Constituent",
    "ConstituencyTree",
    "FeatureBundle",
    "MergeResult",

    # Parser (Data - Verbs)
    "ChomskyParser",
    "MergeOperation",
    "ConstituencyParser",

    # Symmetry (Data - Verification)
    "SymmetryMeasure",
    "GrammaticalSymmetry",
    "compute_symmetry",
]

__version__ = "0.1.0"
__author__ = "Thiago Butignon"
__description__ = "Universal Grammar for Transformers - O(1) because grammar IS compression"
