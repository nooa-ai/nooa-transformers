"""
Grammatical Symmetry - Data Layer (Verification)

Linguistic Role: Grammar checker for input-output consistency
Clean Architecture: Verification operations

Measures structural symmetry between input and output to detect hallucinations.
Based on Glass framework: hallucinations are grammatical inconsistencies.

O(n) symmetry computation where n = sequence length
"""

from typing import List, Set, Dict, Tuple
import torch
from .structures import (
    ConstituencyTree,
    SyntacticObject,
    SymmetryMetrics,
    FeatureType
)


class SymmetryMeasure:
    """
    Measures for grammatical symmetry

    Following Glass framework components:
    - Entity preservation: Named entities should be preserved
    - Predicate preservation: Action verbs should be preserved
    - Negation consistency: Negation markers should be consistent
    - Structural similarity: Parse tree similarity
    """

    @staticmethod
    def entity_overlap(
        input_entities: Set[str],
        output_entities: Set[str]
    ) -> float:
        """
        Measure entity preservation

        Args:
            input_entities: Entities in input
            output_entities: Entities in output

        Returns:
            Overlap ratio [0.0, 1.0]
        """
        if not input_entities:
            return 1.0  # No entities to preserve

        # Intersection over input size
        preserved = input_entities.intersection(output_entities)
        return len(preserved) / len(input_entities)

    @staticmethod
    def predicate_overlap(
        input_predicates: Set[str],
        output_predicates: Set[str]
    ) -> float:
        """
        Measure predicate (verb) preservation

        Args:
            input_predicates: Verbs in input
            output_predicates: Verbs in output

        Returns:
            Overlap ratio [0.0, 1.0]
        """
        if not input_predicates:
            return 1.0  # No predicates to preserve

        preserved = input_predicates.intersection(output_predicates)
        return len(preserved) / len(input_predicates)

    @staticmethod
    def negation_consistency(
        input_has_negation: bool,
        output_has_negation: bool
    ) -> float:
        """
        Check negation consistency

        Args:
            input_has_negation: Whether input contains negation
            output_has_negation: Whether output contains negation

        Returns:
            1.0 if consistent, 0.0 if inconsistent
        """
        return 1.0 if input_has_negation == output_has_negation else 0.0

    @staticmethod
    def structural_similarity(
        input_tree: ConstituencyTree,
        output_tree: ConstituencyTree
    ) -> float:
        """
        Measure parse tree similarity

        Uses tree edit distance (simplified version)

        Args:
            input_tree: Input constituency tree
            output_tree: Output constituency tree

        Returns:
            Similarity score [0.0, 1.0]
        """
        # Compare number of constituents (simple measure)
        input_const_count = len(input_tree.constituents)
        output_const_count = len(output_tree.constituents)

        if input_const_count == 0 and output_const_count == 0:
            return 1.0

        # Similarity based on constituent count difference
        max_count = max(input_const_count, output_const_count)
        diff = abs(input_const_count - output_const_count)

        return 1.0 - (diff / max_count)


class GrammaticalSymmetry:
    """
    Complete grammatical symmetry computation

    Combines all symmetry measures into overall score
    Used to compute symmetry loss
    """

    def __init__(
        self,
        alpha: float = 0.3,  # Entity weight
        beta: float = 0.3,   # Predicate weight
        gamma: float = 0.2,  # Negation weight
        delta: float = 0.2   # Structure weight
    ):
        """
        Args:
            alpha: Weight for entity overlap
            beta: Weight for predicate overlap
            gamma: Weight for negation consistency
            delta: Weight for structural similarity
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.measure = SymmetryMeasure()

    def compute(
        self,
        input_tree: ConstituencyTree,
        output_tree: ConstituencyTree,
        input_tokens: List[str],
        output_tokens: List[str],
        input_pos: List[str],
        output_pos: List[str]
    ) -> SymmetryMetrics:
        """
        Compute full symmetry metrics

        Args:
            input_tree: Input constituency tree
            output_tree: Output constituency tree
            input_tokens: Input tokens
            output_tokens: Output tokens
            input_pos: Input POS tags
            output_pos: Output POS tags

        Returns:
            SymmetryMetrics with all components
        """
        # Extract entities (proper nouns, etc.)
        input_entities = self._extract_entities(input_tokens, input_pos)
        output_entities = self._extract_entities(output_tokens, output_pos)

        # Extract predicates (verbs)
        input_predicates = self._extract_predicates(input_tokens, input_pos)
        output_predicates = self._extract_predicates(output_tokens, output_pos)

        # Check negation
        input_has_neg = self._has_negation(input_tokens)
        output_has_neg = self._has_negation(output_tokens)

        # Compute all measures
        entity_overlap = self.measure.entity_overlap(input_entities, output_entities)
        predicate_overlap = self.measure.predicate_overlap(input_predicates, output_predicates)
        negation_consistency = self.measure.negation_consistency(input_has_neg, output_has_neg)
        structural_similarity = self.measure.structural_similarity(input_tree, output_tree)

        return SymmetryMetrics(
            entity_overlap=entity_overlap,
            predicate_overlap=predicate_overlap,
            negation_consistency=negation_consistency,
            structural_similarity=structural_similarity
        )

    def _extract_entities(self, tokens: List[str], pos_tags: List[str]) -> Set[str]:
        """Extract named entities (proper nouns)"""
        entities = set()
        for token, pos in zip(tokens, pos_tags):
            if pos in ("NNP", "NNPS"):  # Proper nouns
                entities.add(token.lower())
        return entities

    def _extract_predicates(self, tokens: List[str], pos_tags: List[str]) -> Set[str]:
        """Extract predicates (verbs)"""
        predicates = set()
        for token, pos in zip(tokens, pos_tags):
            if pos.startswith("VB"):  # All verb forms
                predicates.add(token.lower())
        return predicates

    def _has_negation(self, tokens: List[str]) -> bool:
        """Check for negation markers"""
        negation_words = {"not", "n't", "no", "never", "none", "neither", "nor"}
        return any(token.lower() in negation_words for token in tokens)


def compute_symmetry(
    input_tree: ConstituencyTree,
    output_tree: ConstituencyTree,
    input_tokens: List[str],
    output_tokens: List[str],
    input_pos: List[str],
    output_pos: List[str],
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2
) -> float:
    """
    Convenience function to compute overall symmetry score

    Args:
        input_tree: Input parse tree
        output_tree: Output parse tree
        input_tokens: Input tokens
        output_tokens: Output tokens
        input_pos: Input POS tags
        output_pos: Output POS tags
        alpha: Entity weight
        beta: Predicate weight
        gamma: Negation weight
        delta: Structure weight

    Returns:
        Overall symmetry score [0.0, 1.0]
    """
    symmetry = GrammaticalSymmetry(alpha, beta, gamma, delta)
    metrics = symmetry.compute(
        input_tree=input_tree,
        output_tree=output_tree,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_pos=input_pos,
        output_pos=output_pos
    )
    return metrics.overall_symmetry


class SymmetryLoss(torch.nn.Module):
    """
    Symmetry Loss for training

    Loss = 1.0 - symmetry_score

    High symmetry (1.0) → Low loss (0.0)
    Low symmetry (0.0) → High loss (1.0)

    This encourages model to preserve grammatical structure
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.2
    ):
        super().__init__()
        self.symmetry_computer = GrammaticalSymmetry(alpha, beta, gamma, delta)

    def forward(
        self,
        input_tree: ConstituencyTree,
        output_tree: ConstituencyTree,
        input_tokens: List[str],
        output_tokens: List[str],
        input_pos: List[str],
        output_pos: List[str]
    ) -> torch.Tensor:
        """
        Compute symmetry loss

        Args:
            input_tree: Input parse tree
            output_tree: Output parse tree
            input_tokens: Input tokens
            output_tokens: Output tokens
            input_pos: Input POS tags
            output_pos: Output POS tags

        Returns:
            Loss tensor (scalar)
        """
        metrics = self.symmetry_computer.compute(
            input_tree=input_tree,
            output_tree=output_tree,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_pos=input_pos,
            output_pos=output_pos
        )

        symmetry_score = metrics.overall_symmetry
        loss = 1.0 - symmetry_score

        return torch.tensor(loss, dtype=torch.float32)


# Export convenience function
__all__ = [
    "SymmetryMeasure",
    "GrammaticalSymmetry",
    "compute_symmetry",
    "SymmetryLoss"
]
