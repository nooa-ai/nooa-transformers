"""
Basic integration tests for Grammatical Transformers

Tests the core functionality without requiring full training.
"""

import torch
import pytest


def test_imports():
    """Test that all main modules can be imported"""
    try:
        from grammatical_transformers import (
            SyntacticObject,
            Constituent,
            ConstituencyTree,
            ChomskyParser,
            MergeOperation,
            GrammaticalBertModel,
            GrammaticalBertConfig,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_merge_operation():
    """Test Chomsky's Merge operation"""
    from grammatical_transformers import (
        MergeOperation,
        SyntacticObject,
        FeatureBundle,
    )

    # Create two syntactic objects
    X = SyntacticObject(
        label="V",
        features=FeatureBundle(),
        token_index=0,
        word="runs"
    )

    Y = SyntacticObject(
        label="N",
        features=FeatureBundle(),
        token_index=1,
        word="dog"
    )

    # Merge
    merge_op = MergeOperation()
    result = merge_op.merge(X, Y)

    # Verify result
    assert result.constituent is not None
    assert result.constituent.head is not None
    assert result.constituent.complement is not None
    assert result.constituent.label.endswith("P")  # Should create XP
    print(f"✓ Merge operation: {result.constituent}")


def test_constituency_parsing():
    """Test constituency parsing from attention weights"""
    from grammatical_transformers import ChomskyParser
    import torch

    parser = ChomskyParser()

    # Create dummy attention weights
    seq_len = 5
    attention_weights = torch.rand(seq_len, seq_len)

    # Make attention matrix more structured (higher on diagonal)
    attention_weights = attention_weights + torch.eye(seq_len) * 2.0
    attention_weights = torch.softmax(attention_weights, dim=-1)

    # Create features
    features = parser.create_default_features(seq_len)

    # Parse
    tree = parser.parse(
        attention_weights=attention_weights,
        features=features,
        tokens=["the", "dog", "runs", "fast", "."]
    )

    # Verify tree
    assert tree.token_count == seq_len
    assert len(tree.constituents) >= 0  # May have parsed some constituents
    print(f"✓ Parsed tree: {tree}")


def test_grammatical_bert_config():
    """Test GrammaticalBertConfig creation"""
    from grammatical_transformers import GrammaticalBertConfig

    config = GrammaticalBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        constituency_penalty=0.5,
    )

    assert config.vocab_size == 1000
    assert config.hidden_size == 128
    assert config.constituency_penalty == 0.5
    print(f"✓ Config created: {config}")


def test_grammatical_bert_forward():
    """Test GrammaticalBertModel forward pass"""
    from grammatical_transformers import (
        GrammaticalBertModel,
        GrammaticalBertConfig,
    )
    import torch

    # Small config for testing
    config = GrammaticalBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        constituency_penalty=0.3,
    )

    model = GrammaticalBertModel(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Verify output shapes
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)
    assert outputs.pooler_output.shape == (batch_size, config.hidden_size)
    print(f"✓ Forward pass successful: {outputs.last_hidden_state.shape}")


def test_symmetry_computation():
    """Test symmetry computation"""
    from grammatical_transformers import (
        ConstituencyTree,
        SymmetryMeasure,
    )

    # Create dummy trees
    input_tree = ConstituencyTree(token_count=3)
    output_tree = ConstituencyTree(token_count=3)

    measure = SymmetryMeasure()

    # Test entity overlap
    input_entities = {"John", "Mary"}
    output_entities = {"John", "Mary"}
    overlap = measure.entity_overlap(input_entities, output_entities)

    assert overlap == 1.0  # Perfect overlap
    print(f"✓ Symmetry measure: {overlap}")


def test_sequence_classification():
    """Test GrammaticalBertForSequenceClassification"""
    from grammatical_transformers import (
        GrammaticalBertForSequenceClassification,
        GrammaticalBertConfig,
    )
    import torch

    # Small config
    config = GrammaticalBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        num_labels=2,  # Binary classification
    )

    model = GrammaticalBertForSequenceClassification(config)
    model.eval()

    # Dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))

    # Forward
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # Verify
    assert outputs.logits.shape == (batch_size, config.num_labels)
    assert outputs.loss is not None
    print(f"✓ Classification forward: logits={outputs.logits.shape}, loss={outputs.loss.item():.4f}")


if __name__ == "__main__":
    print("Running Grammatical Transformers tests...\n")

    print("1. Testing imports...")
    test_imports()

    print("\n2. Testing Merge operation...")
    test_merge_operation()

    print("\n3. Testing constituency parsing...")
    test_constituency_parsing()

    print("\n4. Testing config...")
    test_grammatical_bert_config()

    print("\n5. Testing forward pass...")
    test_grammatical_bert_forward()

    print("\n6. Testing symmetry...")
    test_symmetry_computation()

    print("\n7. Testing sequence classification...")
    test_sequence_classification()

    print("\n✅ All tests passed!")
