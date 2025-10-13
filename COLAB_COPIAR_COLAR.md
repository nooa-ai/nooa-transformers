# Google Colab - Copiar e Colar (5 minutos)

## 🎯 Objetivo
Treinar GrammaticalBERT no SST-2 (sentiment analysis) em 20 minutos usando GPU grátis.

**Resposta rápida**: ❌ NÃO precisa preparar dataset! Tudo automático.

---

## 📋 Passo a Passo

### 1. Abrir Google Colab
👉 https://colab.research.google.com

### 2. Ativar GPU Grátis
1. No menu: **Runtime → Change runtime type**
2. **Hardware accelerator**: GPU
3. **Save**

### 3. Copiar e Colar (uma célula de cada vez)

#### Célula 1: Verificar GPU
```python
import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ Não disponível'}")
```

#### Célula 2: Clonar repo
```python
!git clone https://github.com/nooa-ai/nooa-transformers.git
%cd nooa-transformers/grammatical_transformers
```

#### Célula 3: Instalar
```python
!pip install -e . -q
!pip install datasets accelerate -q
print("✅ Pronto!")
```

#### Célula 4: Treinar (aguarde ~20 min)
```python
!python benchmarks/glue_test.py \
  --task sst2 \
  --epochs 3 \
  --batch_size 32 \
  --device cuda
```

#### Célula 5: Ver resultados
```python
# Resultados aparecem automaticamente na célula anterior
# Procure por:
# - Accuracy: ~91-93%
# - F1 Score: ~91-93%
# - Training time: ~20 minutos
```

---

## 🎉 Pronto!

Você acabou de treinar o GrammaticalBERT!

### 📊 O que aconteceu?

1. ✅ **Dataset baixado automaticamente**: 67K exemplos do SST-2
2. ✅ **Modelo treinado**: 3 epochs (~20 min)
3. ✅ **Resultados**: Accuracy e F1 score

### 🔥 Próximos Passos

#### Opção A: Comparar com vanilla BERT
```python
!python benchmarks/compare_vanilla.py --task sst2 --num_samples 1000
```

#### Opção B: Testar hallucination detection
```python
!python benchmarks/hallucination_test.py
```

#### Opção C: Testar outras tarefas GLUE
```python
# Gramaticalidade (interessante para GrammaticalBERT!)
!python benchmarks/glue_test.py --task cola

# Natural Language Inference (mais desafiador)
!python benchmarks/glue_test.py --task mnli
```

---

## 🆘 Problemas Comuns

### "GPU not available"
**Solução**: Runtime → Change runtime type → GPU → Save

### "Out of memory"
**Solução**: Reduzir batch_size:
```python
!python benchmarks/glue_test.py --task sst2 --batch_size 16  # ou 8
```

### "Session disconnected"
**Causa**: Colab free desconecta após 12h ou inatividade
**Solução**: Reabrir e rodar novamente (progresso salvo se você salvou o modelo)

---

## 💾 Salvar Modelo Treinado

```python
# Salvar localmente no Colab
from grammatical_transformers import GrammaticalBertForSequenceClassification
model.save_pretrained("./my_model")

# Copiar para Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./my_model /content/drive/MyDrive/grammatical_bert_sst2
```

Depois baixe do Drive ou use em outro notebook!

---

## 📖 Mais Informações

- **Notebook completo**: `colab_training.ipynb` (mais exemplos)
- **Guia detalhado**: `COLAB_QUICKSTART.md`
- **Hardware**: `TRAINING_GUIDE.md`
- **Documentação**: `README.md`

---

## 🚀 LFG!

**Tempo total**: ~25 minutos (5 min setup + 20 min training)
**Custo**: $0 (grátis!)
**Preparação de dataset**: Zero, tudo automático!
