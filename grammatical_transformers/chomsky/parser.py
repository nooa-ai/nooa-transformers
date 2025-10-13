"""
Chomsky Parser - Data Layer (DATA-001 Pattern)

Linguistic Role: VERBS - Actions that build syntactic structures
Clean Architecture: Implementation of domain operations

This implements Chomsky's Merge operation:
- External Merge: Combines two separate objects
- Internal Merge: Movement (creates copies for displacement)

O(n) complexity where n = sequence length
Not O(n²) like full CFG parsing
"""

from typing import List, Optional, Tuple, Set
import torch
from .structures import (
    SyntacticObject,
    Constituent,
    ConstituencyTree,
    FeatureBundle,
    MergeResult,
    FeatureType
)


class MergeOperation:
    """
    Core Merge operation from Chomsky's Minimalist Program

    Merge(X, Y) → {X, Y} with head determination

    Head selection rules:
    1. Functional heads (D, T, C) project over lexical heads (N, V, A)
    2. If both lexical, leftmost projects (head-initial languages)
    """

    @staticmethod
    def merge(X: SyntacticObject, Y: SyntacticObject) -> MergeResult:
        """
        External Merge: Combine two syntactic objects

        Args:
            X: First syntactic object
            Y: Second syntactic object

        Returns:
            MergeResult containing the new constituent
        """
        # Determine head (functional categories project)
        functional_categories = {'D', 'T', 'C', 'Comp'}

        if X.label in functional_categories:
            head, complement = X, Y
        elif Y.label in functional_categories:
            head, complement = Y, X
        elif X.is_head:
            head, complement = X, Y
        else:
            # Default: leftmost is head
            head = X if X.token_index < Y.token_index else Y
            complement = Y if head == X else X

        # Create constituent
        constituent = Constituent(
            head=head,
            complement=complement
        )

        return MergeResult(
            constituent=constituent,
            merged_objects=(X, Y),
            operation_type="external"
        )

    @staticmethod
    def internal_merge(constituent: Constituent, mover: SyntacticObject) -> MergeResult:
        """
        Internal Merge: Movement operation (e.g., wh-movement)

        Creates a copy of the mover in specifier position

        Example:
            "What did you see __?"
            → "what" moves from complement to specifier
        """
        # Create new constituent with mover in specifier position
        new_constituent = Constituent(
            head=constituent.head,
            complement=constituent.complement,
            specifier=Constituent(
                head=mover,
                complement=None
            )
        )

        return MergeResult(
            constituent=new_constituent,
            merged_objects=(constituent.head, mover),
            operation_type="internal"
        )


class ConstituencyParser:
    """
    Lightweight constituency parser using Merge

    O(n) parsing by using:
    1. Attention weights as merge probabilities
    2. Greedy bottom-up construction
    3. Feature agreement to guide merging

    NOT a full CKY parser (which would be O(n³))
    """

    def __init__(self, max_depth: int = 5):
        """
        Args:
            max_depth: Maximum tree depth (prevents infinite recursion)
        """
        self.max_depth = max_depth
        self.merge_op = MergeOperation()

    def parse_from_attention(
        self,
        attention_weights: torch.Tensor,
        feature_bundles: List[FeatureBundle],
        tokens: Optional[List[str]] = None
    ) -> ConstituencyTree:
        """
        Build constituency tree from attention weights

        Insight: High attention = likely in same constituent

        Args:
            attention_weights: [seq_len, seq_len] attention matrix
            feature_bundles: Feature bundle for each token
            tokens: Optional surface forms

        Returns:
            ConstituencyTree representing parse
        """
        seq_len = attention_weights.shape[0]

        # Create leaf nodes (words as syntactic objects)
        syntactic_objects = self._create_leaf_nodes(
            seq_len=seq_len,
            feature_bundles=feature_bundles,
            tokens=tokens
        )

        # Build tree bottom-up
        tree = ConstituencyTree(token_count=seq_len)

        # Greedy merging based on attention weights
        constituents = self._greedy_merge(
            syntactic_objects=syntactic_objects,
            attention_weights=attention_weights,
            tree=tree
        )

        # Set root to highest-level constituent
        if constituents:
            tree.root = max(constituents, key=lambda c: c.span[1] - c.span[0])

        return tree

    def _create_leaf_nodes(
        self,
        seq_len: int,
        feature_bundles: List[FeatureBundle],
        tokens: Optional[List[str]]
    ) -> List[SyntacticObject]:
        """Create leaf syntactic objects (words)"""
        syntactic_objects = []

        for i in range(seq_len):
            features = feature_bundles[i] if i < len(feature_bundles) else FeatureBundle()
            word = tokens[i] if tokens and i < len(tokens) else None

            # Infer category from features or use default
            category = features.features.get(FeatureType.CATEGORY, "X")

            obj = SyntacticObject(
                label=category,
                features=features,
                token_index=i,
                is_head=features.features.get(FeatureType.HEAD, False),
                word=word
            )
            syntactic_objects.append(obj)

        return syntactic_objects

    def _greedy_merge(
        self,
        syntactic_objects: List[SyntacticObject],
        attention_weights: torch.Tensor,
        tree: ConstituencyTree
    ) -> List[Constituent]:
        """
        Greedy bottom-up merging

        Algorithm:
        1. Find pair with highest attention
        2. Check feature compatibility
        3. Merge if compatible
        4. Repeat until convergence
        """
        constituents: List[Constituent] = []
        merged_indices: Set[int] = set()

        # Get sorted pairs by attention weight
        pairs = self._get_merge_candidates(attention_weights, merged_indices)

        for (i, j, weight) in pairs:
            if i in merged_indices or j in merged_indices:
                continue

            # Check feature compatibility
            obj_i = syntactic_objects[i]
            obj_j = syntactic_objects[j]

            if not self._are_compatible(obj_i, obj_j):
                continue

            # Merge
            merge_result = self.merge_op.merge(obj_i, obj_j)
            constituent = merge_result.constituent

            # Add to tree
            tree.add_constituent(constituent)
            constituents.append(constituent)

            # Mark as merged
            merged_indices.add(i)
            merged_indices.add(j)

        return constituents

    def _get_merge_candidates(
        self,
        attention_weights: torch.Tensor,
        merged_indices: Set[int]
    ) -> List[Tuple[int, int, float]]:
        """
        Get merge candidates sorted by attention weight

        Only considers adjacent or nearly-adjacent pairs (local constituency)
        """
        seq_len = attention_weights.shape[0]
        candidates = []

        for i in range(seq_len):
            if i in merged_indices:
                continue

            # Only look at nearby tokens (window = 3)
            for j in range(max(0, i - 3), min(seq_len, i + 4)):
                if j == i or j in merged_indices:
                    continue

                weight = attention_weights[i, j].item()

                # Only consider high-attention pairs
                if weight > 0.1:  # Threshold
                    candidates.append((i, j, weight))

        # Sort by weight descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:10]  # Top 10 candidates

    def _are_compatible(
        self,
        obj1: SyntacticObject,
        obj2: SyntacticObject
    ) -> bool:
        """
        Check if two objects can merge (feature agreement)

        Compatibility rules:
        1. Feature bundles must be compatible
        2. Should be adjacent or nearly adjacent
        """
        # Check feature compatibility
        if not obj1.features.matches(obj2.features):
            return False

        # Check proximity (shouldn't be too far apart)
        distance = abs(obj1.token_index - obj2.token_index)
        if distance > 3:  # Max distance for direct merge
            return False

        return True


class ChomskyParser:
    """
    Main parser interface (Factory pattern - MAIN-001)

    Usage:
        parser = ChomskyParser()
        tree = parser.parse(
            attention_weights=attention_matrix,
            features=feature_list,
            tokens=["the", "dog", "barks"]
        )
    """

    def __init__(self, max_depth: int = 5):
        self.constituency_parser = ConstituencyParser(max_depth=max_depth)

    def parse(
        self,
        attention_weights: torch.Tensor,
        features: List[FeatureBundle],
        tokens: Optional[List[str]] = None
    ) -> ConstituencyTree:
        """
        Parse sequence into constituency tree

        Args:
            attention_weights: [seq_len, seq_len] attention matrix
            features: Feature bundle for each token
            tokens: Optional surface forms

        Returns:
            ConstituencyTree
        """
        return self.constituency_parser.parse_from_attention(
            attention_weights=attention_weights,
            feature_bundles=features,
            tokens=tokens
        )

    def create_constituency_mask(
        self,
        tree: ConstituencyTree
    ) -> torch.Tensor:
        """
        Create constituency-aware attention mask

        Tokens in same constituent can attend freely
        Cross-constituent attention is penalized

        Returns:
            [seq_len, seq_len] mask (0 = allowed, -inf = blocked)
        """
        matrix = tree.get_constituency_matrix()

        # Convert to PyTorch tensor
        mask = torch.zeros(tree.token_count, tree.token_count)

        for i in range(tree.token_count):
            for j in range(tree.token_count):
                if not matrix[i][j]:
                    # Block cross-constituent attention
                    mask[i, j] = float('-inf')

        return mask

    @staticmethod
    def create_default_features(
        seq_len: int,
        pos_tags: Optional[List[str]] = None
    ) -> List[FeatureBundle]:
        """
        Create default feature bundles from POS tags

        Maps POS tags to categories:
        - NN, NNS → N (noun)
        - VB, VBD, VBZ → V (verb)
        - DT → D (determiner)
        - etc.
        """
        features = []

        for i in range(seq_len):
            pos = pos_tags[i] if pos_tags and i < len(pos_tags) else "X"

            # Map POS to category
            if pos.startswith("NN"):
                category = "N"
            elif pos.startswith("VB"):
                category = "V"
            elif pos.startswith("JJ"):
                category = "A"  # Adjective
            elif pos in ("DT", "THE", "A"):
                category = "D"  # Determiner
            else:
                category = "X"  # Unknown

            feature_bundle = FeatureBundle(features={
                FeatureType.CATEGORY: category
            })

            features.append(feature_bundle)

        return features
