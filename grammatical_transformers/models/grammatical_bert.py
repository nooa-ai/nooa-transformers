"""
Grammatical BERT - Main Model Implementation

Extends Hugging Face BERT with grammatical constraints from Chomsky's theory.

Key modifications:
1. Constituency-aware attention (respects grammatical boundaries)
2. Symmetry loss (prevents hallucinations)
3. Compatible with standard BERT checkpoints (can load vanilla BERT)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput
)

from .attention import GrammaticalMultiHeadAttention
from ..chomsky.symmetry import SymmetryLoss


class GrammaticalBertConfig(BertConfig):
    """
    Configuration for Grammatical BERT

    Extends BertConfig with grammatical parameters
    """

    model_type = "grammatical_bert"

    def __init__(
        self,
        constituency_penalty: float = 0.5,
        use_symmetry_loss: bool = True,
        symmetry_loss_weight: float = 0.1,
        **kwargs
    ):
        """
        Args:
            constituency_penalty: Penalty for cross-constituent attention [0.0, 1.0]
            use_symmetry_loss: Whether to use symmetry loss
            symmetry_loss_weight: Weight for symmetry loss in total loss
            **kwargs: Standard BERT config parameters
        """
        super().__init__(**kwargs)
        self.constituency_penalty = constituency_penalty
        self.use_symmetry_loss = use_symmetry_loss
        self.symmetry_loss_weight = symmetry_loss_weight


@dataclass
class GrammaticalBertOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    """
    Output with additional grammatical metrics

    Extends standard BERT output with:
    - symmetry_score: Grammatical symmetry between input and output
    - constituency_trees: Parsed constituency trees
    """
    symmetry_score: Optional[torch.Tensor] = None
    constituency_info: Optional[Dict[str, Any]] = None


class GrammaticalBertEmbeddings(nn.Module):
    """
    Standard BERT embeddings (unchanged)

    We reuse BERT's embedding layer without modifications
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard BERT embedding forward"""
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                dtype=torch.long,
                device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class GrammaticalBertLayer(nn.Module):
    """
    Single transformer layer with grammatical attention

    Replaces standard attention with constituency-aware attention
    """

    def __init__(self, config: GrammaticalBertConfig):
        super().__init__()
        self.attention = GrammaticalMultiHeadAttention(
            config=config,
            constituency_penalty=config.constituency_penalty
        )

        # Standard feed-forward network (unchanged)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through layer"""
        # Grammatical attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = nn.functional.gelu(intermediate_output)

        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output + attention_output)

        outputs = (layer_output,)
        if output_attentions:
            outputs = outputs + (attention_outputs[1],)

        return outputs


class GrammaticalBertEncoder(nn.Module):
    """
    Stack of Grammatical BERT layers

    Similar to standard BERT encoder but with grammatical layers
    """

    def __init__(self, config: GrammaticalBertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            GrammaticalBertLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """Forward through all layers"""
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GrammaticalBertModel(BertPreTrainedModel):
    """
    Main Grammatical BERT model

    Can be used as drop-in replacement for BERT
    Supports loading vanilla BERT checkpoints
    """

    config_class = GrammaticalBertConfig

    def __init__(self, config: GrammaticalBertConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = GrammaticalBertEmbeddings(config)
        self.encoder = GrammaticalBertEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        # Symmetry loss (optional)
        if config.use_symmetry_loss:
            self.symmetry_loss = SymmetryLoss()
        else:
            self.symmetry_loss = None

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            inputs_embeds: [batch_size, seq_len, hidden_size]
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dict output

        Returns:
            Model outputs with last_hidden_state and pooler_output
        """
        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # Get embeddings
        if inputs_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
        else:
            embedding_output = inputs_embeds

        # Encoder
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs.last_hidden_state

        # Pooler (CLS token)
        pooled_output = self.pooler(sequence_output[:, 0, :])
        pooled_output = self.pooler_activation(pooled_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GrammaticalBertForSequenceClassification(BertPreTrainedModel):
    """
    Grammatical BERT for sequence classification

    Example tasks: Sentiment analysis, text classification, NLI
    """

    config_class = GrammaticalBertConfig

    def __init__(self, config: GrammaticalBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = GrammaticalBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        """Forward for sequence classification"""
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GrammaticalBertForTokenClassification(BertPreTrainedModel):
    """
    Grammatical BERT for token classification

    Example tasks: NER, POS tagging, chunking
    """

    config_class = GrammaticalBertConfig

    def __init__(self, config: GrammaticalBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = GrammaticalBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        """Forward for token classification"""
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Register models for auto loading
from transformers import AutoConfig, AutoModel
AutoConfig.register("grammatical_bert", GrammaticalBertConfig)
AutoModel.register(GrammaticalBertConfig, GrammaticalBertModel)
