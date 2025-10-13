# Google Colab - Guia Rápido

## 🎯 O que vamos fazer?

Treinar o GrammaticalBERT no Google Colab (GPU grátis!) usando o dataset GLUE que já vem pronto.

**Boa notícia**: Não precisa preparar dataset! O Hugging Face baixa automaticamente. 🎉

---

## 📋 Passo a Passo

### 1️⃣ Abrir o Google Colab

1. Acesse: https://colab.research.google.com
2. Faça login com sua conta Google
3. Clique em **File → New Notebook**

### 2️⃣ Ativar GPU Grátis

1. No notebook, clique em **Runtime → Change runtime type**
2. Em **Hardware accelerator**, selecione **GPU** (T4 grátis)
3. Clique **Save**

### 3️⃣ Copiar e Colar o Código Abaixo

Copie todo o código do notebook `colab_training.ipynb` que eu vou criar agora.

---

## 🚀 Opções de Treinamento

### Opção A: Fine-tuning em GLUE (Recomendado para começar)

**O que é**: Pegar um modelo BERT já treinado e ajustar para tarefas específicas (sentiment analysis, etc)

**Dataset**: GLUE (já vem pronto via Hugging Face)
- SST-2: 67K exemplos (sentiment)
- MNLI: 393K exemplos (entailment)
- CoLA: 8.5K exemplos (gramaticalidade)
- ... mais 5 tarefas

**Tempo**: 10-30 minutos por tarefa (GPU grátis T4)

**Custo**: $0 (grátis)

**Preparar dataset?**: ❌ NÃO! Baixa automaticamente

```python
# Tudo automático:
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")  # Pronto!
```

### Opção B: Pre-training do Zero (Avançado)

**O que é**: Treinar um modelo completamente do zero em texto genérico

**Dataset**: Precisa preparar corpus grande (Wikipedia, livros)
- Tamanho: 16GB+ de texto
- Formato: arquivos .txt

**Tempo**: Dias a semanas (mesmo com GPU)

**Custo**: $100-500 (precisa GPU paga tipo A100)

**Preparar dataset?**: ✅ SIM! Precisa corpus grande

**Recomendação**: Comece com fine-tuning (Opção A)! Pre-training é muito caro e demorado.

---

## 📊 Datasets Disponíveis (Zero Setup)

### GLUE Benchmark (Recomendado)

```python
from datasets import load_dataset

# Sentiment Analysis (mais fácil)
sst2 = load_dataset("glue", "sst2")
# 67K exemplos
# Input: "this movie is great"
# Output: positive/negative

# Natural Language Inference
mnli = load_dataset("glue", "mnli")
# 393K exemplos
# Input: premise + hypothesis
# Output: entailment/contradiction/neutral

# Gramaticalidade (interessante para GrammaticalBERT!)
cola = load_dataset("glue", "cola")
# 8.5K exemplos
# Input: sentence
# Output: grammatical/ungrammatical
```

### Outros Datasets Prontos

```python
# Question Answering
squad = load_dataset("squad")

# Translation
wmt = load_dataset("wmt14", "de-en")

# Summarization
cnn_dailymail = load_dataset("cnn_dailymail", "3.0.0")
```

**Tudo baixa automaticamente!** Sem preparação manual.

---

## ⚡ Colab Free vs Pro

### Free (T4 GPU)
- **Custo**: $0
- **GPU**: Tesla T4 (16GB VRAM)
- **Tempo**: 12 horas por sessão
- **Velocidade**:
  - SST-2: ~20 min
  - MNLI: ~2 horas
- **Limitação**: Desconecta após 12h ou inatividade

### Pro ($10/mês)
- **GPU**: A100 (às vezes), melhor prioridade
- **Tempo**: 24 horas por sessão
- **Velocidade**:
  - SST-2: ~5 min
  - MNLI: ~30 min
- **Vantagem**: Menos desconexões

**Recomendação**: Comece com Free! É suficiente para aprender.

---

## 🎓 Próximos Passos

1. ✅ Abrir Colab e ativar GPU
2. ✅ Usar notebook que vou criar (`colab_training.ipynb`)
3. ✅ Rodar fine-tuning no SST-2 (20 min)
4. ✅ Ver os resultados
5. ✅ Testar outras tarefas GLUE

Depois você pode:
- Comparar GrammaticalBERT vs vanilla BERT
- Medir redução de hallucinations
- Publicar resultados

---

## 📝 Resumo

**Precisa preparar dataset?**
- Fine-tuning: ❌ NÃO (usa GLUE, já vem pronto)
- Pre-training: ✅ SIM (precisa corpus grande)

**Recomendação**:
1. Comece com fine-tuning no SST-2 (sentiment)
2. Use Colab Free (suficiente)
3. Leva ~20 minutos
4. Depois teste outras tarefas

**Próximo arquivo**: Vou criar `colab_training.ipynb` com código pronto para copiar e colar! 🚀
