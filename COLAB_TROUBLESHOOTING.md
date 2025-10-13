# Google Colab - Troubleshooting

Problemas comuns e soluções ao treinar GrammaticalBERT no Colab.

---

## ❌ Erro: "No module named 'grammatical_transformers'"

### Causa
Você pulou a célula de instalação ou ela falhou.

### Solução 1: Execute a célula de instalação
```python
# Célula 3: Instalar Dependências
!pip install -e .
```

Aguarde até ver "✅ Instalação completa!"

### Solução 2: Verificar diretório
```python
import os
print(os.getcwd())
# Deve ser: /content/nooa-transformers/grammatical_transformers
```

Se não estiver no diretório correto:
```python
%cd /content/nooa-transformers/grammatical_transformers
```

### Solução 3: Instalar manualmente
```python
import sys
sys.path.insert(0, '/content/nooa-transformers/grammatical_transformers')

# Tentar importar novamente
from grammatical_transformers import GrammaticalBertModel
```

### Solução 4: Reinstalar tudo
```python
# Limpar instalação anterior
!pip uninstall grammatical-transformers -y

# Reinstalar
%cd /content/nooa-transformers/grammatical_transformers
!pip install -e .
```

---

## ❌ Erro: "GPU not available"

### Causa
GPU não está ativada ou sessão perdeu GPU.

### Solução 1: Ativar GPU
1. Menu: **Runtime → Change runtime type**
2. **Hardware accelerator**: GPU
3. **Save**
4. Reiniciar runtime: **Runtime → Restart runtime**

### Solução 2: Verificar disponibilidade
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Solução 3: Colab Free esgotou GPU
**Problema**: Colab free tem limite de uso de GPU.

**Opções**:
- Aguardar algumas horas
- Usar CPU (muito lento): `--device cpu`
- Upgrade para Colab Pro ($10/mês)

---

## ❌ Erro: "CUDA out of memory"

### Causa
Modelo muito grande para VRAM disponível.

### Solução 1: Reduzir batch_size
```python
# Em vez de batch_size=32
!python benchmarks/glue_test.py --task sst2 --batch_size 16 --device cuda

# Se ainda der erro, tente 8
!python benchmarks/glue_test.py --task sst2 --batch_size 8 --device cuda
```

### Solução 2: Reduzir tamanho do modelo
```python
config = GrammaticalBertConfig(
    vocab_size=30522,
    hidden_size=512,      # Reduzido de 768
    num_hidden_layers=6,  # Reduzido de 12
    num_attention_heads=8,  # Reduzido de 12
)
```

### Solução 3: Limpar cache CUDA
```python
import torch
torch.cuda.empty_cache()
```

### Solução 4: Reiniciar runtime
**Runtime → Restart runtime** e execute células novamente.

---

## ❌ Erro: "Session disconnected"

### Causa
Colab free desconecta após:
- 12 horas de uso
- Inatividade prolongada
- Uso excessivo de recursos

### Solução 1: Prevenir desconexão
Execute este código em uma célula:
```python
# Anti-desconexão (execute em célula separada)
import time
from IPython.display import clear_output

for i in range(1000):
    time.sleep(60)  # Espera 1 minuto
    clear_output(wait=True)
    print(f"Mantendo sessão ativa: {i+1} minuto(s)")
```

### Solução 2: Salvar checkpoints periodicamente
Modifique o treinamento para salvar a cada epoch:
```python
# No seu script de treinamento
for epoch in range(num_epochs):
    train_epoch()
    model.save_pretrained(f"checkpoint_epoch_{epoch}")
```

### Solução 3: Usar Google Drive
Salve modelo no Drive automaticamente:
```python
from google.colab import drive
drive.mount('/content/drive')

# Após treinar
!cp -r ./model /content/drive/MyDrive/checkpoints/
```

---

## ❌ Erro: "Repository not found" ao clonar

### Causa
Repositório ainda não está público ou URL errada.

### Solução 1: Verificar URL
```python
# URL correta
!git clone https://github.com/nooa-ai/nooa-transformers.git
```

### Solução 2: Verificar se repositório existe
Abra no navegador: https://github.com/nooa-ai/nooa-transformers

### Solução 3: Clonar de fork
Se você tem um fork:
```python
!git clone https://github.com/SEU-USUARIO/nooa-transformers.git
```

---

## ❌ Erro: "Dataset download failed"

### Causa
Problema de conexão ou dataset não encontrado.

### Solução 1: Tentar novamente
```python
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
```

### Solução 2: Cache pode estar corrompido
```python
# Limpar cache
!rm -rf ~/.cache/huggingface/datasets/

# Baixar novamente
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
```

### Solução 3: Verificar conexão
```python
!ping google.com -c 3
!curl https://huggingface.co
```

---

## ❌ Erro: "ModuleNotFoundError: torch"

### Causa
PyTorch não instalado ou versão incompatível.

### Solução
```python
# Desinstalar versão antiga
!pip uninstall torch torchvision torchaudio -y

# Instalar versão compatível com CUDA
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar instalação
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

---

## ⚠️ Aviso: "Running on CPU"

### Causa
GPU não ativada ou não disponível.

### Solução
Veja seção "GPU not available" acima.

### Alternativa: Aceitar CPU (lento)
```python
# Treinar em CPU (vai levar 5-10x mais tempo)
!python benchmarks/glue_test.py --task sst2 --device cpu --batch_size 8
```

---

## ⚠️ Aviso: Training muito lento

### Diagnóstico
```python
import torch
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Se T4:  ~20 min para SST-2 (normal)
# Se P100: ~15 min (rápido)
# Se CPU:  ~2-3 horas (muito lento)
```

### Solução 1: Verificar que está usando GPU
```python
# No código de treinamento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Solução 2: Usar batch_size maior
```python
# Se VRAM permitir
!python benchmarks/glue_test.py --task sst2 --batch_size 64 --device cuda
```

### Solução 3: Reduzir logging
```python
# Menos prints = mais rápido
!python benchmarks/glue_test.py --task sst2 --quiet
```

---

## 🔧 Debugging Geral

### Verificar ambiente completo
```python
import sys
import torch
from transformers import __version__ as transformers_version

print("=== Ambiente ===" )
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers_version}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Working dir: {sys.path[0]}")

# Verificar instalação do pacote
try:
    import grammatical_transformers
    print(f"\n✅ grammatical_transformers: v{grammatical_transformers.__version__}")
except ImportError as e:
    print(f"\n❌ grammatical_transformers: {e}")
```

### Teste mínimo
```python
# Teste mais simples possível
from grammatical_transformers import GrammaticalBertConfig
import torch

config = GrammaticalBertConfig(vocab_size=100, hidden_size=64, num_hidden_layers=1)
print(f"✅ Config criado: {config}")
```

### Habilitar modo debug
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📞 Pedir Ajuda

Se nenhuma solução funcionou:

### 1. Criar Issue no GitHub
https://github.com/nooa-ai/nooa-transformers/issues

**Inclua**:
- Erro completo (screenshot ou texto)
- Código que você executou
- Output de `verificar ambiente completo` (acima)
- Qual célula falhou

### 2. Copiar este template

```
## Problema
[Descreva o problema]

## Erro
```
[Cole o erro completo aqui]
```

## Ambiente
- GPU: [T4/P100/None]
- Colab: [Free/Pro]
- Python: [versão]
- PyTorch: [versão]

## Código
```python
[Cole o código que causou o erro]
```

## O que já tentei
- [ ] Reinstalei pacote
- [ ] Reiniciei runtime
- [ ] Verifiquei GPU está ativa
- [ ] ...
```

---

## 🎓 Dicas Gerais

### ✅ Sempre execute as células em ordem
Não pule células, especialmente instalação.

### ✅ Aguarde instalação completa
A célula 3 (instalação) pode levar 2-3 minutos.

### ✅ Reinicie runtime após instalar
Às vezes é necessário: **Runtime → Restart runtime**

### ✅ Salve checkpoints frequentemente
Use Google Drive para não perder progresso.

### ✅ Monitore recursos
```python
# Ver uso de GPU
!nvidia-smi

# Ver uso de RAM
!free -h

# Ver uso de disco
!df -h
```

---

## 📚 Recursos Adicionais

- **Colab FAQ**: https://research.google.com/colaboratory/faq.html
- **PyTorch Troubleshooting**: https://pytorch.org/get-started/locally/
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets/

---

**Última atualização**: 2025-10-13
**Problemas não listados?** Abra issue: https://github.com/nooa-ai/nooa-transformers/issues
