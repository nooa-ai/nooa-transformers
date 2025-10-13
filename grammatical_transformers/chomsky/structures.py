"""
Grammatical Structures - Domain Layer (DOM-001 Pattern)

Linguistic Role: NOUNS - Entities representing syntactic objects
Clean Architecture: Domain entities with no dependencies

These structures represent the core concepts from Chomsky's Minimalist Program:
- Syntactic Objects: Minimal units of syntax
- Constituents: Grouped syntactic objects
- Feature Bundles: Grammatical features (number, person, case)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, Dict, Any
from enum import Enum


class FeatureType(Enum):
    """Grammatical feature types following Chomsky's feature theory"""
    # φ-features (agreement features)
    PERSON = "person"
    NUMBER = "number"
    GENDER = "gender"

    # Case features
    NOMINATIVE = "nominative"
    ACCUSATIVE = "accusative"
    GENITIVE = "genitive"

    # Tense/Aspect features
    TENSE = "tense"
    ASPECT = "aspect"

    # Structural features
    CATEGORY = "category"  # N, V, A, P
    HEAD = "head"  # Is this the head of the constituent?


@dataclass(frozen=True)
class FeatureBundle:
    """
    Feature Bundle - represents grammatical features

    Example:
        FeatureBundle(
            features={
                FeatureType.PERSON: "3rd",
                FeatureType.NUMBER: "singular",
                FeatureType.CATEGORY: "V"
            }
        )
    """
    features: Dict[FeatureType, Any] = field(default_factory=dict)

    def matches(self, other: 'FeatureBundle') -> bool:
        """Check if features are compatible (for Agree operation)"""
        for feature_type, value in self.features.items():
            if feature_type in other.features:
                if other.features[feature_type] != value:
                    return False
        return True

    def merge_features(self, other: 'FeatureBundle') -> 'FeatureBundle':
        """Merge two feature bundles (union of features)"""
        merged = {**self.features, **other.features}
        return FeatureBundle(features=merged)


@dataclass
class SyntacticObject:
    """
    Minimal syntactic unit following Chomsky's Minimalist Program

    Properties:
    - label: Category label (N, V, D, etc.)
    - features: Feature bundle
    - token_index: Position in sequence
    - is_head: Whether this is the head of its constituent

    Linguistic parallel:
    - Like a word with its grammatical properties
    """
    label: str
    features: FeatureBundle
    token_index: int
    is_head: bool = False
    word: Optional[str] = None  # Surface form

    def __hash__(self):
        return hash((self.label, self.token_index))

    def __repr__(self):
        head_mark = "*" if self.is_head else ""
        word_repr = f"'{self.word}'" if self.word else ""
        return f"{head_mark}{self.label}_{self.token_index}{word_repr}"


@dataclass
class Constituent:
    """
    Constituent - result of Merge operation

    Following Chomsky's Merge:
    - Merge(X, Y) = {X, Y} with one as head
    - Head determines label of constituent

    Example:
        V + NP → VP (V is head)
        D + NP → DP (D is head)
    """
    head: SyntacticObject
    complement: Optional[SyntacticObject]
    specifier: Optional['Constituent'] = None  # For more complex structures
    span: Tuple[int, int] = field(init=False)  # Token range covered
    label: str = field(init=False)

    def __post_init__(self):
        """Compute span and label after initialization"""
        # Span is from leftmost to rightmost token
        indices = [self.head.token_index]
        if self.complement:
            indices.append(self.complement.token_index)
        if self.specifier:
            indices.extend(range(self.specifier.span[0], self.specifier.span[1] + 1))

        self.span = (min(indices), max(indices))

        # Label comes from head (following X-bar theory)
        self.label = f"{self.head.label}P"  # VP, NP, DP, etc.

    def contains(self, token_index: int) -> bool:
        """Check if token is within this constituent"""
        return self.span[0] <= token_index <= self.span[1]

    def __repr__(self):
        comp_repr = f" + {self.complement}" if self.complement else ""
        spec_repr = f" [{self.specifier}]" if self.specifier else ""
        return f"[{self.label} {self.head}{comp_repr}{spec_repr}]"


@dataclass
class MergeResult:
    """
    Result of Merge operation

    Contains:
    - constituent: The resulting constituent
    - merged_objects: Objects that were merged
    - operation_type: Type of merge (Internal/External)
    """
    constituent: Constituent
    merged_objects: Tuple[SyntacticObject, SyntacticObject]
    operation_type: str  # "external" or "internal"

    @property
    def head(self) -> SyntacticObject:
        return self.constituent.head

    @property
    def complement(self) -> Optional[SyntacticObject]:
        return self.constituent.complement


class ConstituencyTree:
    """
    Constituency parse tree representing hierarchical structure

    Linguistic Role: Complete syntactic structure

    Properties:
    - constituents: All constituents in the tree
    - root: Top-level constituent (typically CP or TP)
    - token_count: Number of tokens in sequence

    O(1) insight: Tree IS the grammatical compression
    """

    def __init__(self, token_count: int):
        self.token_count = token_count
        self.constituents: List[Constituent] = []
        self.root: Optional[Constituent] = None
        self._token_to_constituents: Dict[int, List[Constituent]] = {}

    def add_constituent(self, constituent: Constituent) -> None:
        """Add a constituent to the tree"""
        self.constituents.append(constituent)

        # Index for fast lookup
        for token_idx in range(constituent.span[0], constituent.span[1] + 1):
            if token_idx not in self._token_to_constituents:
                self._token_to_constituents[token_idx] = []
            self._token_to_constituents[token_idx].append(constituent)

    def get_constituents_for_token(self, token_index: int) -> List[Constituent]:
        """Get all constituents containing this token"""
        return self._token_to_constituents.get(token_index, [])

    def get_constituency_matrix(self) -> List[List[bool]]:
        """
        Create binary constituency matrix

        Matrix[i][j] = True if tokens i and j are in the same constituent

        This is used to create constituency-aware attention masks
        """
        matrix = [[False] * self.token_count for _ in range(self.token_count)]

        # All tokens attend to themselves
        for i in range(self.token_count):
            matrix[i][i] = True

        # Tokens in same constituent can attend to each other
        for constituent in self.constituents:
            for i in range(constituent.span[0], constituent.span[1] + 1):
                for j in range(constituent.span[0], constituent.span[1] + 1):
                    matrix[i][j] = True

        return matrix

    def get_depth_for_token(self, token_index: int) -> int:
        """Get syntactic depth (how many constituents contain this token)"""
        return len(self.get_constituents_for_token(token_index))

    def __repr__(self):
        return f"ConstituencyTree(tokens={self.token_count}, constituents={len(self.constituents)})"


@dataclass
class SymmetryMetrics:
    """
    Metrics for grammatical symmetry between input and output

    Following Glass framework for hallucination detection:
    - entity_overlap: Preservation of named entities
    - predicate_overlap: Preservation of action verbs
    - negation_consistency: Negation markers preserved
    """
    entity_overlap: float = 0.0
    predicate_overlap: float = 0.0
    negation_consistency: float = 1.0  # 1.0 = consistent, 0.0 = inconsistent
    structural_similarity: float = 0.0

    @property
    def overall_symmetry(self) -> float:
        """
        Compute overall symmetry score

        σ = α·entity + β·predicate + γ·negation + δ·structure
        """
        alpha, beta, gamma, delta = 0.3, 0.3, 0.2, 0.2
        return (
            alpha * self.entity_overlap +
            beta * self.predicate_overlap +
            gamma * self.negation_consistency +
            delta * self.structural_similarity
        )


# Type aliases for clarity
TokenIndex = int
ConstituentLabel = str
FeatureMatrix = Dict[TokenIndex, FeatureBundle]
