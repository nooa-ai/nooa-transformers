# Datasets Guide - O que usar e como preparar

## 🎯 TL;DR

**Para começar**: Use GLUE (não precisa preparar nada!)
**Para research sério**: Pre-training (precisa preparar corpus)

---

## 📊 Opção A: Fine-tuning (Recomendado)

### Sem Preparação (Zero Setup)

Estes datasets baixam automaticamente via Hugging Face:

#### 1. GLUE Benchmark (Melhor para começar)

```python
from datasets import load_dataset

# Sentiment Analysis - SST-2
dataset = load_dataset("glue", "sst2")
# 67K treino, 872 validação
# Input: "this movie is great"
# Output: 0 (negative) ou 1 (positive)
# Tempo: ~20 min (GPU free)

# Gramaticalidade - CoLA
dataset = load_dataset("glue", "cola")
# 8.5K treino, 1K validação
# Input: "The dog barked." vs "Dog the barked."
# Output: 0 (ungrammatical) ou 1 (grammatical)
# Tempo: ~10 min (GPU free)
# ⚡ Perfeito para GrammaticalBERT!

# Natural Language Inference - MNLI
dataset = load_dataset("glue", "mnli")
# 393K treino, 20K validação
# Input: premise + hypothesis
# Output: entailment/contradiction/neutral
# Tempo: ~2 horas (GPU free)

# Semantic Similarity - STS-B
dataset = load_dataset("glue", "stsb")
# 7K treino, 1.5K validação
# Input: sentence1 + sentence2
# Output: similarity score (0-5)
# Tempo: ~15 min (GPU free)
```

#### 2. Question Answering

```python
# SQuAD v2
dataset = load_dataset("squad_v2")
# 130K treino, 12K validação
# Input: context + question
# Output: answer span
# Tempo: ~3 horas (GPU free)
```

#### 3. Summarization

```python
# CNN/DailyMail
dataset = load_dataset("cnn_dailymail", "3.0.0")
# 287K treino, 13K validação
# Input: article
# Output: summary
# Tempo: ~8 horas (GPU free)
```

#### 4. Translation

```python
# WMT14 English-German
dataset = load_dataset("wmt14", "de-en")
# 4.5M pares
# Tempo: Muito longo (precisa GPU paga)
```

### Como Usar (Exemplo SST-2)

```python
from datasets import load_dataset
from transformers import BertTokenizer

# 1. Carregar dataset
dataset = load_dataset("glue", "sst2")

# 2. Ver estrutura
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['sentence', 'label', 'idx'],
#         num_rows: 67349
#     })
#     validation: Dataset({
#         features: ['sentence', 'label', 'idx'],
#         num_rows: 872
#     })
# })

# 3. Ver exemplo
print(dataset['train'][0])
# {
#   'sentence': 'hide new secretions from the parental units',
#   'label': 0,  # negative
#   'idx': 0
# }

# 4. Tokenizar
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['sentence'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Usar com PyTorch DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=32, shuffle=True)

# Pronto! Sem preparação manual.
```

---

## 🔬 Opção B: Pre-training do Zero (Avançado)

### Quando fazer pre-training?

✅ **Faça pre-training se**:
- Tem domínio específico (medicina, direito, código)
- Tem corpus proprietário
- Quer publicar novo modelo
- Tem budget ($100-500)

❌ **Não faça pre-training se**:
- Só quer testar o modelo (use fine-tuning)
- Não tem GPU potente (A100)
- Orçamento limitado

### Datasets para Pre-training

#### 1. Wikipedia + BookCorpus (Padrão BERT)

```bash
# Wikipedia (16GB)
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Processar Wikipedia
pip install wikiextractor
python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 \
  --processes 8 \
  --output wikipedia_text \
  --bytes 100M \
  --compress \
  --json

# BookCorpus (11GB)
# Use biblioteca books-dataset
pip install datasets
from datasets import load_dataset
books = load_dataset("bookcorpus", split="train")
```

**Tamanho total**: ~27GB de texto
**Tempo de processamento**: 4-8 horas
**Tempo de training**: 3-7 dias (4x A100)
**Custo**: $400-800

#### 2. C4 (Colossal Clean Crawled Corpus)

```python
from datasets import load_dataset

# C4 (750GB!)
dataset = load_dataset("c4", "en", streaming=True)
# Streaming = não baixa tudo de uma vez
```

**Tamanho**: 750GB
**Tempo de training**: 1-2 semanas (múltiplas A100s)
**Custo**: $1,000-2,000

#### 3. The Pile (Diverse Dataset)

```bash
# 825GB de dados diversos
# https://pile.eleuther.ai/
wget https://the-eye.eu/public/AI/pile/train/00.jsonl.zst
# ... (muitos arquivos)
```

**Tamanho**: 825GB
**Tempo**: Semanas
**Custo**: $2,000+

### Formato de Dados para Pre-training

#### Estrutura de arquivos

```
corpus/
├── train/
│   ├── file001.txt
│   ├── file002.txt
│   └── ...
└── valid/
    ├── file001.txt
    └── file002.txt
```

#### Formato de texto

```text
# file001.txt
This is sentence one. This is sentence two.

This is a new paragraph.

Another paragraph here.
```

**Requisitos**:
- UTF-8 encoding
- Um parágrafo por linha (separado por linha vazia)
- Sem marcação especial
- Limpo (sem HTML, códigos, etc)

#### Preparar corpus customizado

```python
# Script para preparar corpus
import os
from pathlib import Path

def prepare_corpus(input_dir, output_file):
    """
    Consolidar múltiplos arquivos em um corpus para pre-training
    """
    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in Path(input_dir).rglob('*.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    out.write(text + '\n\n')

    print(f"✅ Corpus criado: {output_file}")
    # Calcular tamanho
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"📊 Tamanho: {size_mb:.1f} MB")

# Uso
prepare_corpus('raw_texts/', 'corpus.txt')
```

#### Script de Pre-training (básico)

```python
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 1. Carregar corpus
dataset = load_dataset('text', data_files={'train': 'corpus.txt'})

# 2. Tokenizar
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

# 3. Data collator para MLM (Masked Language Modeling)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 4. Configurar modelo
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)
model = BertForMaskedLM(config)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir='./grammatical_bert_pretrain',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    logging_steps=1000,
    learning_rate=1e-4,
    warmup_steps=10000,
    fp16=True,  # Mixed precision
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train']
)

# 7. Train!
trainer.train()
```

**Tempo estimado**: 3-7 dias (depende do tamanho do corpus e GPU)

---

## 📊 Comparação: Fine-tuning vs Pre-training

| Aspecto | Fine-tuning | Pre-training |
|---------|-------------|--------------|
| **Dataset** | Pronto (GLUE) | Precisa preparar (Wikipedia) |
| **Preparação** | Zero | 4-8 horas |
| **Tamanho** | 67K-400K exemplos | 16GB-800GB texto |
| **Tempo training** | 20 min - 2 horas | 3-7 dias |
| **GPU necessária** | T4 (grátis) | A100 (paga) |
| **Custo** | $0 | $400-2000 |
| **Quando usar** | Testar, avaliar, research | Modelo novo, domínio específico |

---

## 🎓 Recomendação para Você

### Fase 1: Aprender (Agora) ✅
```bash
# Use GLUE - sem preparação
python benchmarks/glue_test.py --task sst2
# 20 minutos, $0, aprende como funciona
```

### Fase 2: Validar (Depois de Fase 1) ✅
```bash
# Rode todos os benchmarks GLUE
python benchmarks/glue_test.py --all-tasks
# 2-4 horas, $0 (Colab free), valida abordagem
```

### Fase 3: Comparar (Depois de Fase 2) ✅
```bash
# Compare com vanilla BERT
python benchmarks/compare_vanilla.py
# Mede overhead, accuracy, hallucinations
```

### Fase 4: Publicar (Se resultados bons) 📝
- Documente descobertas
- Atualize RESULTS.md
- Considere pre-training se tiver budget

### Fase 5: Pre-training (Opcional, $$$) 💰
- Só se resultados de fine-tuning forem promissores
- Só se tiver budget ($400-800)
- Só se quiser publicar modelo completo

---

## 🆘 FAQ

**Q: Preciso preparar dataset para começar?**
A: ❌ NÃO! Use GLUE (automático).

**Q: Qual dataset usar primeiro?**
A: SST-2 (sentiment) - mais simples, 20 min.

**Q: Quando fazer pre-training?**
A: Só depois de validar com fine-tuning. Pre-training é caro.

**Q: Quantos dados preciso para pre-training?**
A: Mínimo 16GB de texto (Wikipedia). Ideal: 50GB+.

**Q: Posso usar meu próprio dataset?**
A: ✅ SIM! Para fine-tuning, prepare CSV:
```csv
text,label
"This is great",1
"This is bad",0
```

Depois:
```python
dataset = load_dataset('csv', data_files='my_data.csv')
```

**Q: Quanto custa rodar GLUE completo?**
A: $0 no Colab free (T4). ~4 horas.

**Q: Quanto custa pre-training?**
A: $400-800 (A100 por 3-7 dias).

---

## 📚 Recursos

- **Hugging Face Datasets**: https://huggingface.co/datasets
- **GLUE Benchmark**: https://gluebenchmark.com
- **Pre-training tutorial**: https://huggingface.co/course/chapter7/6

---

## ✅ Resumo

**Começar agora** (0 preparação):
```python
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
# Pronto!
```

**Pre-training futuro** (horas de preparação):
```bash
# Baixar Wikipedia
wget https://dumps.wikimedia.org/...
# Processar
python -m wikiextractor...
# Treinar (dias)
python pretrain.py
```

**Recomendação**: Comece com GLUE! Pre-training é para depois. 🚀
