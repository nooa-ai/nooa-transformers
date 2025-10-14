"""
Grammatical Attention Mechanisms

Implements constituency-aware attention that respects grammatical boundaries.

Key insight: Attention should be stronger within constituents,
weaker across constituent boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from ..chomsky.structures import ConstituencyTree
from ..chomsky.parser import ChomskyParser


class ConstituencyAwareAttention(nn.Module):
    """
    Attention mechanism that respects constituency boundaries

    Modifies standard attention with constituency mask:
    - High attention within constituents
    - Penalized attention across constituents
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        constituency_penalty: float = 0.5,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.constituency_penalty = constituency_penalty

        # Standard attention projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        # Chomsky parser for constituency detection
        self.parser = ChomskyParser()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape for multi-head attention

        [batch_size, seq_len, hidden_size] →
        [batch_size, num_heads, seq_len, head_size]
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def create_constituency_penalty_mask(
        self,
        attention_scores: torch.Tensor,
        batch_size: int,
        seq_length: int,
    ) -> torch.Tensor:
        """
        Create constituency-based penalty mask

        Args:
            attention_scores: [batch, heads, seq_len, seq_len]
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Penalty mask [batch, heads, seq_len, seq_len]
        """
        # Average attention across heads for parsing
        avg_attention = attention_scores.mean(dim=1)  # [batch, seq_len, seq_len]

        penalty_masks = []

        for batch_idx in range(batch_size):
            # Get attention for this example
            attn = avg_attention[batch_idx]  # [seq_len, seq_len]

            # Parse using averaged attention
            # Create dummy features
            features = self.parser.create_default_features(seq_length)

            try:
                # Parse into constituency tree
                tree = self.parser.parse(
                    attention_weights=attn,
                    features=features,
                    tokens=None
                )

                # Create constituency mask
                # 0 for same constituent, 1 for different constituents
                constituency_matrix = tree.get_constituency_matrix()

                # Convert to penalty: 0 (same) → no penalty, 1 (diff) → penalty
                penalty = torch.zeros(seq_length, seq_length)
                for i in range(seq_length):
                    for j in range(seq_length):
                        if not constituency_matrix[i][j]:
                            # Different constituents → apply penalty
                            penalty[i, j] = self.constituency_penalty

                penalty_masks.append(penalty)

            except Exception:
                # If parsing fails, use no penalty
                penalty_masks.append(torch.zeros(seq_length, seq_length))

        # Stack into batch
        penalty_mask = torch.stack(penalty_masks).to(attention_scores.device)

        # Expand for heads: [batch, seq, seq] → [batch, heads, seq, seq]
        penalty_mask = penalty_mask.unsqueeze(1).expand(
            batch_size, self.num_attention_heads, seq_length, seq_length
        )

        return penalty_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with constituency-aware attention

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            output_attentions: Whether to return attention weights

        Returns:
            context_layer: [batch_size, seq_len, hidden_size]
            attention_probs: Optional[batch_size, heads, seq_len, seq_len]
        """
        batch_size, seq_length, _ = hidden_states.size()

        # Standard attention computation
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask (padding)
        if attention_mask is not None:
            # Ensure mask has correct shape
            if attention_mask.dim() == 2:
                # [batch, seq_len] → [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [batch, seq_len, seq_len] → [batch, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)

            # Convert to additive mask (0 = keep, -inf = mask)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # Apply constituency penalty
        # Note: We apply this AFTER padding mask but BEFORE softmax
        penalty_mask = self.create_constituency_penalty_mask(
            attention_scores=attention_scores,
            batch_size=batch_size,
            seq_length=seq_length
        )

        # Apply penalty (subtract from scores to reduce attention)
        attention_scores = attention_scores - penalty_mask

        # Normalize to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Dropout
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class GrammaticalAttention(nn.Module):
    """
    Complete grammatical attention with output projection

    This is the full attention layer including:
    - Multi-head constituency-aware attention
    - Output projection
    - Residual connection support
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        constituency_penalty: float = 0.5,
    ):
        super().__init__()
        self.self_attention = ConstituencyAwareAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            constituency_penalty=constituency_penalty,
        )

        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with residual connection and layer norm

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights

        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_probs: Optional attention weights
        """
        # Self-attention
        self_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        attention_output = self_outputs[0]

        # Output projection
        attention_output = self.output(attention_output)
        attention_output = self.dropout(attention_output)

        # Residual connection + Layer norm
        attention_output = self.layer_norm(attention_output + hidden_states)

        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (self_outputs[1],)

        return outputs


class GrammaticalMultiHeadAttention(nn.Module):
    """
    Wrapper for compatibility with Hugging Face Transformers

    Can be used as drop-in replacement for BertSelfAttention
    """

    def __init__(
        self,
        config,  # Hugging Face config object
        constituency_penalty: float = 0.5,
    ):
        super().__init__()
        self.attention = GrammaticalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            constituency_penalty=constituency_penalty,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward compatible with Hugging Face"""
        # Note: head_mask not yet supported
        return self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
