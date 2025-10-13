# Mission: GrammaticalTransformers

## Context
You are implementing Chomsky's Universal Grammar into Hugging Face Transformers.

## Big O(1) Insight
Clean Architecture IS grammatical mapping:
- Domain ↔ Data ↔ Infra = grammatical transformations
- "Send message" ↔ "Receive message" = symmetric structure
- O(1) because grammar is ALREADY compressed representation

## Your Task

### Phase 1: Study (4h)
1. Read Chomsky's Minimalist Program core concepts (docs/referentials/)
2. Analyze huggingface/transformers attention mechanism:
   - `src/transformers/models/bert/modeling_bert.py`
   - Focus on `BertSelfAttention` class
3. Identify injection points for grammatical constraints

### Phase 2: Design (2h)
Create architecture for:
- `ChomskyParser`: Lightweight Merge-based constituency parser
- `GrammaticalAttention`: Constituency-aware attention
- `SymmetryLoss`: Grammatical consistency loss

### Phase 3: Implement (10h)
Build in this structure:

```
grammatical_transformers/
├── chomsky/
│   ├── __init__.py
│   ├── parser.py           # Merge operation implementation
│   ├── structures.py       # Constituency tree structures
│   └── symmetry.py         # Symmetry computation
├── models/
│   ├── __init__.py
│   ├── grammatical_bert.py # GrammaticalBertModel
│   └── attention.py        # GrammaticalAttention layer
├── benchmarks/
│   ├── glue_test.py
│   ├── hallucination_test.py
│   └── compare_vanilla.py
├── tests/
│   └── test_*.py
└── README.md
```

### Phase 4: Benchmark (4h)
Compare vanilla BERT vs GrammaticalBERT on:
- GLUE tasks
- Hallucination rate (use Glass framework)
- Interpretability metrics

### Phase 5: Documentation (2h)
- README with examples
- RFC for Hugging Face PR
- Architecture diagram

## Key Principles

1. **O(1) Thinking**: Grammar IS compression. Don't add complexity, reveal existing structure.

2. **Clean Architecture**:
   - Domain = Grammatical rules (Chomsky)
   - Data = Token sequences
   - Infra = Attention mechanism

3. **Symmetry Preservation**:
   Input and output must maintain grammatical symmetry
   Like "send" ↔ "receive" in domain modeling

4. **No Overhead**:
   Grammatical parsing should be O(n) where n = sequence length
   Not O(n²) like full syntactic parsing

## Implementation Hints

### Merge Operation (Chomsky)
```python
def merge(X, Y):
    """
    Basic Merge: combines two syntactic objects
    Returns: {X, Y} with one as head
    """
    if is_head(X):
        return {"head": X, "complement": Y}
    else:
        return {"head": Y, "complement": Y}
```

### Constituency Mask
```python
def create_constituency_mask(constituents, seq_len):
    """
    Creates attention mask respecting constituency boundaries
    Tokens in same constituent can attend freely
    Cross-constituent attention is penalized
    """
    mask = torch.zeros(seq_len, seq_len)
    for constituent in constituents:
        start, end = constituent.span
        mask[start:end, start:end] = 0  # Free attention
        # Cross-constituent gets penalty
    return mask
```

### Symmetry Loss
```python
def symmetry_loss(input_structure, output_structure):
    """
    Measures grammatical consistency between input and output
    High symmetry = low hallucination risk

    Based on Glass framework:
    σ = α·entity_overlap + β·predicate_overlap + γ·negation_consistency
    """
    return 1.0 - compute_symmetry(input_structure, output_structure)
```

## Success Criteria

- ✅ Code runs and trains
- ✅ GrammaticalBERT achieves ≥ vanilla BERT accuracy
- ✅ Hallucination rate reduced by ≥20%
- ✅ Interpretability: can visualize constituency structure
- ✅ Production-ready code (tests, docs, types)

## Constraints

- Use PyTorch (Hugging Face standard)
- Compatible with transformers>=4.30.0
- No external dependencies beyond HF ecosystem
- Code style: black, flake8, mypy

## Timeline

Total: 22 hours
- Can be done in ~3 overnight sessions
- Each session: 6-8h of autonomous work

## Reference Papers (Available Locally)

- docs/referentials/ChomskyMinimalistProgram.pdf
- docs/referentials/chomsky1965-ch1.pdf
- docs/referentials/1905.05950v2.pdf (Transformers paper)
- docs/referentials/N19-1419.pdf
- docs/grammar/ (Universal Grammar analysis)

## Context Available

Full Universal Grammar analysis available at:
- docs/grammar/UNIVERSAL_GRAMMAR_PROOF.md
- docs/grammar/CLEAN_ARCHITECTURE_GRAMMAR_ANALYSIS.md
- docs/grammar/grammar-patterns.yml

Use these as theoretical foundation.

---

You have full autonomy. Build it. Make it work. Prove the concept.

When done, output:
1. Full code in /grammatical_transformers
2. Benchmark results in RESULTS.md
3. RFC draft in RFC.md

LFG.
