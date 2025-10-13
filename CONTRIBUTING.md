# Contributing to Grammatical Transformers

Thank you for your interest in contributing to Grammatical Transformers! This project implements Chomsky's Universal Grammar in neural transformers, and we welcome contributions from linguists, ML researchers, and developers alike.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Architecture Guidelines](#architecture-guidelines)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Contribution Workflow](#contribution-workflow)
8. [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Background in linguistics or ML
- Experience level
- Identity or expression

### Expected Behavior

- Be respectful and constructive in discussions
- Focus on ideas, not individuals
- Accept constructive criticism gracefully
- Help newcomers get started

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Publishing others' private information

Report violations to: [thiago.butignon@example.com]

---

## Getting Started

### Prerequisites

**Required Knowledge**:
- Python 3.10+
- PyTorch basics
- Basic understanding of transformers (BERT, attention)

**Helpful Knowledge**:
- Chomsky's linguistic theory (we'll help you learn!)
- Clean Architecture principles
- Hugging Face Transformers library

### Quick Start

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/nooa-transformers.git
   cd nooa-transformers/grammatical_transformers
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run tests**
   ```bash
   python -m pytest tests/
   ```

4. **Try examples**
   ```python
   from grammatical_transformers import GrammaticalBertModel
   # ... explore the code
   ```

---

## Development Setup

### Environment Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

   This includes:
   - pytest (testing)
   - black (formatting)
   - flake8 (linting)
   - mypy (type checking)
   - jupyter (notebooks)

3. **Configure pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Project Structure

```
grammatical_transformers/
├── chomsky/                    # Domain Layer (Grammar)
│   ├── structures.py          # Entities: SyntacticObject, Constituent
│   ├── parser.py              # Use cases: Merge, Parse
│   └── symmetry.py            # Use cases: Symmetry measurement
│
├── models/                     # Infrastructure Layer (Transformers)
│   ├── attention.py           # Constituency-aware attention
│   └── grammatical_bert.py    # Main model implementation
│
├── tests/                      # Test suite
│   ├── test_basic.py          # Unit tests
│   └── test_integration.py    # Integration tests
│
├── benchmarks/                 # Benchmark suite
│   ├── glue_test.py           # GLUE tasks
│   ├── hallucination_test.py  # Hallucination detection
│   └── compare_vanilla.py     # Comparison with vanilla BERT
│
└── examples/                   # Usage examples
    └── quickstart.ipynb       # Jupyter notebook tutorial
```

---

## Architecture Guidelines

### Clean Architecture Principles

**CRITICAL**: This project follows Clean Architecture strictly. All contributions MUST respect the dependency rule:

```
Infrastructure → Data → Domain
(models/)     (parser/) (structures/)
```

#### Rule 1: Domain has ZERO dependencies
```python
# ❌ BAD - Domain depends on PyTorch
class SyntacticObject:
    def __init__(self, embedding: torch.Tensor):
        self.embedding = embedding

# ✅ GOOD - Domain is pure Python
class SyntacticObject:
    def __init__(self, label: str, features: FeatureBundle):
        self.label = label
        self.features = features
```

#### Rule 2: Data depends only on Domain
```python
# ❌ BAD - Data depends on Infrastructure
from ..models.attention import GrammaticalAttention

class ChomskyParser:
    def __init__(self):
        self.attention = GrammaticalAttention()

# ✅ GOOD - Data depends on Domain only
from .structures import SyntacticObject, Constituent

class ChomskyParser:
    def parse(self, X: SyntacticObject, Y: SyntacticObject) -> Constituent:
        return self.merge(X, Y)
```

#### Rule 3: Infrastructure depends on Data + Domain
```python
# ✅ GOOD - Infrastructure uses both layers
from ..chomsky.parser import ChomskyParser
from ..chomsky.structures import ConstituencyTree

class ConstituencyAwareAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.parser = ChomskyParser()
```

### Linguistic Correctness

When implementing grammatical features, follow Chomsky's theory:

1. **Merge is binary**: `Merge(X, Y) → {X, Y}` (not ternary)
2. **Head projection**: Head determines label
3. **Feature checking**: Merge only if features compatible
4. **Minimal operations**: Don't add unnecessary structure

**If unsure, ask!** We're happy to explain linguistic concepts.

---

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

1. **Line length**: 100 characters (not 79)
2. **Imports**: Grouped and sorted
   ```python
   # Standard library
   import os
   from typing import Dict, List

   # Third-party
   import torch
   import numpy as np

   # Local
   from ..chomsky.structures import SyntacticObject
   ```

3. **Type hints**: Required for all public functions
   ```python
   def merge(X: SyntacticObject, Y: SyntacticObject) -> Constituent:
       """Merge two syntactic objects"""
       ...
   ```

4. **Docstrings**: Google style
   ```python
   def parse(attention: torch.Tensor, features: List[FeatureBundle]) -> ConstituencyTree:
       """
       Parse constituency structure from attention weights

       Args:
           attention: Attention weights [seq_len, seq_len]
           features: Feature bundle for each token

       Returns:
           Complete constituency tree

       Raises:
           ValueError: If attention shape doesn't match features length
       """
   ```

### Formatting

Use **black** for automatic formatting:
```bash
black grammatical_transformers/
```

Use **flake8** for linting:
```bash
flake8 grammatical_transformers/ --max-line-length=100
```

Use **mypy** for type checking:
```bash
mypy grammatical_transformers/
```

### Naming Conventions

- **Classes**: PascalCase (`SyntacticObject`, `ChomskyParser`)
- **Functions**: snake_case (`merge`, `compute_symmetry`)
- **Constants**: UPPER_CASE (`MAX_DEPTH`, `FUNCTIONAL_CATEGORIES`)
- **Private**: Prefix with underscore (`_internal_helper`)

**Linguistic terms**: Use standard linguistic terminology
- ✅ `merge`, `constituent`, `head`, `complement`
- ❌ `combine`, `group`, `main`, `child`

---

## Testing Requirements

### Test Coverage

**Minimum requirement**: 80% test coverage for all new code.

Check coverage:
```bash
pytest --cov=grammatical_transformers tests/
```

### Test Structure

1. **Unit tests**: Test individual functions
   ```python
   def test_merge_operation():
       """Test Merge(X, Y) creates correct constituent"""
       X = SyntacticObject(label='N', features=noun_features)
       Y = SyntacticObject(label='D', features=det_features)

       result = merge(X, Y)

       assert isinstance(result, Constituent)
       assert result.head == Y  # D is functional, should be head
       assert result.complement == X
   ```

2. **Integration tests**: Test component interactions
   ```python
   def test_parse_and_attend():
       """Test parsing from attention and applying back to attention"""
       model = GrammaticalBertModel(config)
       outputs = model(input_ids=input_ids)

       # Should produce valid output
       assert outputs.last_hidden_state.shape == expected_shape
   ```

3. **Linguistic validity tests**: Verify grammatical correctness
   ```python
   def test_head_projection():
       """Test that head determines constituent label"""
       # Functional category (D) should project over lexical (N)
       dp = merge(D('the'), N('dog'))
       assert dp.label == 'D'  # DP, not NP

       # Verb should project over object
       vp = merge(V('eat'), N('food'))
       assert vp.label == 'V'  # VP
   ```

### Running Tests

```bash
# All tests
python -m pytest tests/

# Specific file
python -m pytest tests/test_basic.py

# Specific test
python -m pytest tests/test_basic.py::test_merge_operation

# With coverage
python -m pytest --cov=grammatical_transformers tests/

# Verbose
python -m pytest -v tests/
```

---

## Contribution Workflow

### 1. Create an Issue

Before starting work:
1. Check existing issues
2. Create new issue describing:
   - What you want to add/fix
   - Why it's needed
   - How you plan to implement it
3. Wait for maintainer feedback

### 2. Fork and Branch

```bash
# Fork on GitHub, then:
git clone https://github.com/yourusername/nooa-transformers.git
cd nooa-transformers

# Create feature branch
git checkout -b feature/add-probabilistic-parser
```

**Branch naming**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring

### 3. Make Changes

1. **Write code** following guidelines above
2. **Add tests** for new functionality
3. **Update docs** if needed
4. **Run tests** to ensure nothing breaks

```bash
# Format code
black grammatical_transformers/

# Run tests
pytest tests/

# Check types
mypy grammatical_transformers/
```

### 4. Commit

Write clear commit messages:

```bash
git commit -m "feat: Add probabilistic constituency parser

- Implement PCFG parser as alternative to greedy parser
- Add learnable grammar rules
- Maintain O(n³) complexity (acceptable for smaller sequences)
- Add tests for probabilistic parsing

Closes #42"
```

**Commit message format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement

### 5. Push and Pull Request

```bash
git push origin feature/add-probabilistic-parser
```

Then create PR on GitHub with:
- **Title**: Clear, concise description
- **Description**:
  - What changes were made
  - Why they're needed
  - How to test them
- **Link to issue**: "Closes #42"

**PR Checklist**:
- [ ] Tests pass
- [ ] Code formatted (black)
- [ ] Type hints added
- [ ] Docstrings added
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

### 6. Review and Iterate

- Respond to reviewer feedback
- Make requested changes
- Push updates (will update PR automatically)

### 7. Merge

Once approved, maintainer will merge your PR. Congratulations! 🎉

---

## Areas for Contribution

### High Priority

#### 1. Benchmark Results
**Status**: Framework ready, needs actual runs

**Tasks**:
- Run GLUE benchmarks on trained models
- Measure hallucination rates
- Compare with vanilla BERT baseline
- Write up results

**Skills needed**: ML experimentation, data analysis

#### 2. Model Pre-training
**Status**: Architecture complete, needs pre-training

**Tasks**:
- Pre-train GrammaticalBERT on large corpus
- Compare with vanilla BERT pre-training
- Tune hyperparameters (constituency_penalty, symmetry_loss_weight)
- Release pre-trained checkpoints

**Skills needed**: Large-scale ML training, compute resources

#### 3. Visualization Tools
**Status**: Not started

**Tasks**:
- Visualize constituency trees
- Visualize attention patterns
- Interactive constituency explorer
- Integration with tools like BertViz

**Skills needed**: Python visualization (matplotlib, plotly), web dev

#### 4. Multi-language Support
**Status**: English-centric now

**Tasks**:
- Support Universal Dependencies tags
- Add language-specific parsers (Japanese, Arabic, etc.)
- Test on multilingual benchmarks
- Documentation for different languages

**Skills needed**: Linguistics, multilingual NLP

### Medium Priority

#### 5. Performance Optimization
**Tasks**:
- Constituency caching across layers
- Faster parsing algorithms
- CUDA kernels for constituency operations
- Profiling and bottleneck analysis

**Skills needed**: Performance optimization, CUDA

#### 6. Additional Model Variants
**Tasks**:
- GrammaticalGPT (decoder-only)
- GrammaticalT5 (encoder-decoder)
- GrammaticalRoBERTa (improved pre-training)

**Skills needed**: Deep understanding of transformer architectures

#### 7. Constituency Tree Bank
**Tasks**:
- Collect/annotate constituency trees
- Create training dataset for supervised parsing
- Evaluate learned vs. greedy parsing

**Skills needed**: Linguistic annotation, data curation

#### 8. Integration with Hugging Face Hub
**Tasks**:
- Upload pre-trained models to Hub
- Create model cards
- Add to Transformers library
- Documentation and tutorials

**Skills needed**: Hugging Face ecosystem knowledge

### Low Priority (Nice to Have)

#### 9. Additional Symmetry Metrics
**Tasks**:
- Domain-specific symmetry measures
- Learned symmetry weights
- Symmetry visualization

#### 10. Educational Resources
**Tasks**:
- Tutorial notebooks
- Video explanations
- Blog posts
- Conference talks/posters

#### 11. Constituency Probing Tasks
**Tasks**:
- Design probing tasks for constituency knowledge
- Analyze what model learns
- Compare layers

---

## Documentation Standards

### Code Documentation

Every public function/class needs:
1. **Docstring** with description
2. **Args** section with types
3. **Returns** section
4. **Examples** (for complex functions)
5. **Linguistic explanation** (for grammar-related code)

Example:
```python
def merge(X: SyntacticObject, Y: SyntacticObject) -> Constituent:
    """
    Chomsky's Merge operation: Combine two syntactic objects

    Merge is the fundamental operation of syntax (Chomsky 1995).
    It takes two syntactic objects and combines them into a
    constituent, with one object as the head.

    Head selection follows the functional hierarchy:
        Functional categories (D, T, C) > Lexical categories (N, V, A)

    Args:
        X: First syntactic object
        Y: Second syntactic object

    Returns:
        Constituent with X and Y, head determined by category

    Raises:
        FeatureConflictError: If X and Y have incompatible features

    Example:
        >>> D = SyntacticObject('D', det_features, 'the')
        >>> N = SyntacticObject('N', noun_features, 'dog')
        >>> dp = merge(D, N)
        >>> dp.label
        'D'  # D projects, creating DP
        >>> dp.head == D
        True

    References:
        Chomsky, N. (1995). The Minimalist Program. MIT Press.
    """
```

### README Updates

When adding features, update relevant READMEs:
- Main README (if user-facing)
- Module README (if internal)
- Examples (if new capability)

### Changelog

Update CHANGELOG.md:
```markdown
## [Unreleased]

### Added
- Probabilistic constituency parser (#42)
- Support for Japanese grammar (#43)

### Fixed
- Bug in symmetry computation for empty sequences (#44)

### Changed
- Improved parsing speed by 2x (#45)
```

---

## Getting Help

### Resources

1. **Documentation**:
   - README.md - Quick start
   - ARCHITECTURE.md - Deep dive into design
   - RFC.md - Full proposal and theory

2. **Linguistic Background**:
   - Chomsky (1995): The Minimalist Program
   - Chomsky (1965): Aspects of the Theory of Syntax
   - `docs/grammar/UNIVERSAL_GRAMMAR_PROOF.md`

3. **Code Examples**:
   - `tests/` - See how to use each component
   - `examples/` - End-to-end examples

### Contact

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: thiago.butignon@example.com

### Office Hours

We hold virtual office hours every Friday 2-3pm UTC:
- Discuss contributions
- Get help with linguistics
- Pair programming sessions

Join via: [link to video chat]

---

## Recognition

All contributors are recognized in:
- **CONTRIBUTORS.md** - List of all contributors
- **Paper acknowledgments** - If we publish
- **Release notes** - For significant contributions

Thank you for contributing to bringing Universal Grammar to neural networks! 🧠🌍

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
