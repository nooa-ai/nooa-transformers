# Training GrammaticalBERT - Hardware Guide

## TL;DR

- **Development/Testing**: Mac M1/M2/M3 ✅
- **Fine-tuning small tasks**: Mac (slow but works) ⚠️
- **Full GLUE benchmarks**: NVIDIA GPU recommended 🚀
- **Pre-training from scratch**: NVIDIA GPU required ❌ (or cloud)

---

## Scenarios

### 1️⃣ I Just Want to Test the Code

**Hardware**: Mac, CPU, anything
**Time**: Minutes
**Cost**: $0

```bash
# Install
pip install torch transformers

# Run tests
cd grammatical_transformers
python -m pytest tests/ -v

# Quick inference test
python -c "
from models.grammatical_bert import GrammaticalBertModel, GrammaticalBertConfig
import torch

config = GrammaticalBertConfig(vocab_size=1000, hidden_size=128, num_hidden_layers=2)
model = GrammaticalBertModel(config)
input_ids = torch.randint(0, 1000, (2, 10))
outputs = model(input_ids)
print('✅ Model works!', outputs.last_hidden_state.shape)
"
```

**Result**: ✅ Works perfectly on Mac

---

### 2️⃣ I Want to Fine-tune on Small Task (SST-2)

**Hardware**: Mac M1/M2 works, GPU recommended
**Time**: 30-60 min (Mac) vs 5-10 min (GPU)
**Cost**: $0 (Mac) or $0.50 (cloud GPU)

#### Option A: Mac (Apple Silicon)

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Fine-tune SST-2 (sentiment analysis, ~67K examples)
python benchmarks/glue_test.py \
  --task sst2 \
  --epochs 3 \
  --batch_size 16 \
  --device mps  # Use Apple Metal

# Expected: 30-60 minutes
```

#### Option B: Google Colab (Free GPU)

```python
# In Colab notebook:
!git clone https://github.com/yourusername/nooa-transformers
!cd nooa-transformers/grammatical_transformers && pip install -e .

!python benchmarks/glue_test.py --task sst2 --device cuda
# Expected: 5-10 minutes
```

**Result**: ✅ Both work, GPU much faster

---

### 3️⃣ I Want to Run Full GLUE Benchmarks

**Hardware**: NVIDIA GPU recommended (8+ GB VRAM)
**Time**: 4-8 hours (Mac) vs 30-60 min (GPU)
**Cost**: $0 (if you have GPU) or $2-5 (cloud)

#### Recommended: Cloud GPU

**RunPod** (cheapest):
```bash
# 1. Sign up: runpod.io
# 2. Create pod: RTX 3090 ($0.29/h)
# 3. SSH into pod

git clone https://github.com/yourusername/nooa-transformers
cd nooa-transformers/grammatical_transformers
pip install -e .

# Run all GLUE tasks
python benchmarks/glue_test.py --all-tasks --device cuda

# Expected time: ~1 hour
# Cost: ~$0.30
```

**Google Colab Pro** ($10/month):
- Get A100 GPU (sometimes)
- Run all benchmarks
- ~30 min total

**Result**: ✅ Best bang for buck

---

### 4️⃣ I Want to Pre-train from Scratch

**Hardware**: NVIDIA A100 (40GB+) or multiple GPUs
**Time**: Days to weeks
**Cost**: $100-1000+ depending on setup

#### Reality Check:

Pre-training BERT from scratch requires:
- **Data**: 16GB+ text corpus (Wikipedia, BookCorpus)
- **Compute**: 4-8x A100 GPUs for 3-7 days
- **Cost**: $500-2000
- **Memory**: 40GB+ VRAM per GPU
- **Expertise**: Distributed training setup

#### Alternatives:

**Option A: Start from pretrained BERT**
```python
# Load vanilla BERT weights
from transformers import BertModel
vanilla_bert = BertModel.from_pretrained("bert-base-uncased")

# Transfer to GrammaticalBERT
grammatical_bert = GrammaticalBertModel(config)
grammatical_bert.load_state_dict(vanilla_bert.state_dict(), strict=False)

# Only train grammatical components (much faster!)
# Time: Hours instead of days
```

**Option B: Use Lambda Labs / CoreWeave**
```bash
# Lambda Labs: $0.60/h for A100 40GB
# CoreWeave: $0.80/h for A100 40GB

# Total cost for pre-training: $400-800
```

**Result**: ⚠️ Doable but expensive, transfer learning recommended

---

## Hardware Comparison

| Task | Mac M1 | RTX 3090 | A100 40GB |
|------|--------|----------|-----------|
| **Inference** | Fast ✅ | Fastest ✅ | Fastest ✅ |
| **Unit Tests** | Seconds ✅ | Seconds ✅ | Seconds ✅ |
| **Fine-tune (small)** | 30-60 min ⚠️ | 5-10 min ✅ | 3-5 min ✅ |
| **GLUE (all tasks)** | 4-8 hours ❌ | 1 hour ✅ | 30 min ✅ |
| **Pre-training** | Months ❌ | 1-2 weeks ⚠️ | 3-7 days ✅ |

---

## Recommended Setup

### For Development (what you have now):
```bash
# Mac M1/M2/M3
pip install torch torchvision torchaudio
pip install transformers datasets

# Works for:
✅ Code development
✅ Unit tests
✅ Small experiments
✅ Inference
```

### For Training (when needed):
```bash
# Option 1: Google Colab (free tier)
# - Good for: Learning, small experiments
# - Limits: 12h sessions, slower T4 GPU

# Option 2: Google Colab Pro ($10/month)
# - Good for: GLUE benchmarks, fine-tuning
# - Pros: A100 access (sometimes), 24h sessions

# Option 3: RunPod/Lambda Labs (pay-as-you-go)
# - Good for: Serious training
# - Cost: $0.30-2/hour
# - Pros: Choose GPU, persistent storage
```

---

## Quick Start on Mac (Right Now)

```bash
cd /Users/thiagobutignon/dev/nooa-transformers/grammatical_transformers

# Install dependencies
pip install torch transformers datasets pytest

# Run tests (should work immediately)
python -m pytest tests/ -v

# Try quick inference
python -c "
from models.grammatical_bert import GrammaticalBertModel, GrammaticalBertConfig
import torch

print('Creating model...')
config = GrammaticalBertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)
model = GrammaticalBertModel(config)

print('Running inference...')
input_ids = torch.randint(0, 30522, (2, 128))
attention_mask = torch.ones(2, 128)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)
print(f'✅ Success! Output shape: {outputs.last_hidden_state.shape}')
"
```

---

## Cloud Setup Guide

### Google Colab (Easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. New notebook
3. Runtime → Change runtime type → GPU (T4 or A100)
4. Run:

```python
!git clone https://github.com/yourusername/nooa-transformers
%cd nooa-transformers/grammatical_transformers
!pip install -e .
!python -m pytest tests/ -v
```

### RunPod (Best Value)

1. Sign up: [runpod.io](https://runpod.io)
2. Pods → Deploy
3. Choose: RTX 3090 ($0.29/h) or A100 ($1.99/h)
4. Template: PyTorch
5. SSH into pod:

```bash
git clone https://github.com/yourusername/nooa-transformers
cd nooa-transformers/grammatical_transformers
pip install -e .
python benchmarks/glue_test.py --task sst2
```

---

## Cost Estimation

### Small Experiment (1 GLUE task)
- **Mac**: Free, 30-60 min
- **Colab Free**: Free, 5-10 min
- **RunPod**: $0.05 (10 min × $0.29/h)

### Full GLUE Benchmarks (8 tasks)
- **Mac**: Free, 4-8 hours
- **Colab Pro**: $0 (included in $10/month)
- **RunPod**: $0.30 (1h × $0.29/h)

### Pre-training from Scratch
- **Mac**: Not viable
- **4x A100**: $400-800 (3-7 days)
- **Transfer learning**: $10-50 (hours instead of days)

---

## My Recommendation for You

**Phase 1 (Now - Development)**:
- Use your Mac M1/M2/M3
- Run tests, develop, experiment
- Cost: $0

**Phase 2 (When ready to train)**:
- Get Google Colab Pro ($10/month)
- Run GLUE benchmarks
- Validate approach
- Cost: $10

**Phase 3 (Serious training)**:
- Use RunPod/Lambda Labs
- Pre-train or fine-tune extensively
- Release checkpoints
- Cost: $50-500 depending on scope

---

## Bottom Line

**Your Mac is perfect for:**
✅ Development
✅ Testing
✅ Inference
✅ Small experiments

**You'll need GPU for:**
- Full GLUE benchmarks (optional, Mac works but slow)
- Pre-training (required)
- Large-scale evaluation (required)

**Best first step:**
```bash
# Try it on your Mac RIGHT NOW:
cd grammatical_transformers
pip install torch transformers datasets
python -m pytest tests/ -v
```

If tests pass, you're good to develop on Mac!

When you're ready to train seriously, spin up a cloud GPU for $0.30-2/hour.

**LFG!** 🚀
