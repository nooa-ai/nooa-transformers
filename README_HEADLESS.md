# 🤖 Claude Code Headless Mode - GrammaticalTransformers

Automação completa para implementar Chomsky's Universal Grammar em Hugging Face Transformers.

## 🎯 O que isso faz?

Roda Claude Code em modo headless (sem interface) para:
1. Estudar teoria de Chomsky (docs/referentials/)
2. Analisar BERT attention mechanism
3. Implementar GrammaticalBERT com constituency-aware attention
4. Criar benchmarks comparando com BERT vanilla
5. Gerar documentação e RFC para PR no Hugging Face

**Tudo de forma autônoma. Você dorme. Claude trabalha.** 💤

---

## 📋 Pré-requisitos

```bash
# 1. Claude Code instalado
npm install -g @anthropic/claude-code

# 2. tmux (para modo background)
brew install tmux  # Mac
# ou
apt install tmux   # Linux

# 3. Python + PyTorch (para executar o código gerado depois)
pip install torch transformers datasets
```

---

## 🚀 Quick Start

### Opção 1: Execução Direta (Interativa)
```bash
./run_headless.sh
```

- Roda Claude Code com a task
- Você vê o progresso em tempo real
- Pede confirmações (mais seguro)

### Opção 2: Background com tmux (Recomendado para overnight)
```bash
./run_headless_tmux.sh
```

- Roda Claude Code em sessão tmux background
- **YOLO MODE**: `--dangerously-skip-permissions` (auto-aprova tudo)
- Você pode desconectar/dormir
- Logs salvos automaticamente

### Opção 3: Watch Mode (Monitoramento)
```bash
watch -n 10 ./monitor.sh
```

Atualiza a cada 10s com:
- Status da sessão
- Arquivos criados
- Linhas de código
- Últimos logs

---

## 🛠️ Scripts Disponíveis

| Script | Descrição |
|--------|-----------|
| `run_headless.sh` | Execução direta (foreground) |
| `run_headless_tmux.sh` | Execução background (tmux) |
| `monitor.sh` | Monitor de progresso |
| `stop.sh` | Para execução |
| `TASK.md` | Especificação da task |

---

## 📊 Monitoramento

### Em tempo real:
```bash
# Opção 1: Attach na sessão tmux
tmux attach -t claude_grammatical_*

# Opção 2: Tail dos logs
tail -f claude_headless.log

# Opção 3: Monitor script
./monitor.sh
```

### Auto-refresh:
```bash
watch -n 10 ./monitor.sh  # Atualiza a cada 10s
```

---

## 🎯 Output Esperado

```
grammatical_transformers/
├── chomsky/
│   ├── __init__.py
│   ├── parser.py           (~800 LOC) - Merge operation
│   ├── structures.py       (~600 LOC) - Constituency trees
│   └── symmetry.py         (~400 LOC) - Symmetry computation
├── models/
│   ├── __init__.py
│   ├── grammatical_bert.py (~1200 LOC) - GrammaticalBERT
│   └── attention.py        (~900 LOC) - Grammatical attention
├── benchmarks/
│   ├── glue_test.py        (~500 LOC) - GLUE benchmark
│   ├── hallucination_test.py (~400 LOC) - Hallucination metrics
│   └── compare_vanilla.py  (~300 LOC) - Comparison script
├── tests/
│   └── test_*.py           (~1500 LOC) - Unit tests
├── README.md               (~300 LOC) - Documentação
├── RFC.md                  (~500 LOC) - Hugging Face PR draft
└── RESULTS.md              (~200 LOC) - Benchmark results

TOTAL: ~7,600 LOC
```

---

## ⚡ Flags do Claude Code

### Flags usadas neste projeto:

| Flag | Descrição |
|------|-----------|
| `-p "prompt"` | Modo headless (non-interactive) |
| `--dangerously-skip-permissions` | **YOLO MODE** - Auto-aprova tudo ⚠️ |
| `--output-format stream-json` | Output estruturado JSON |
| `--continue` | Continua última sessão |
| `--resume <id>` | Resume sessão específica |
| `--allowedTools` | Limita ferramentas disponíveis |

### ⚠️ YOLO Mode - Quando usar?

**Use `--dangerously-skip-permissions` APENAS quando:**
- ✅ Ambiente isolado (container/VM)
- ✅ Backup feito
- ✅ Confia 100% na task
- ✅ Rodando overnight sem supervisão

**NÃO use quando:**
- ❌ Ambiente de produção
- ❌ Dados sensíveis
- ❌ Primeira vez testando
- ❌ Task complexa/arriscada

---

## 🔧 Troubleshooting

### Problema: Claude Code não encontrado
```bash
# Instalar
npm install -g @anthropic/claude-code

# Verificar
which claude
claude --version
```

### Problema: tmux não encontrado
```bash
# Mac
brew install tmux

# Linux
sudo apt install tmux
```

### Problema: Sessão tmux não inicia
```bash
# Listar sessões ativas
tmux ls

# Matar todas
tmux kill-server

# Tentar novamente
./run_headless_tmux.sh
```

### Problema: Logs não aparecem
```bash
# Verificar se arquivo existe
ls -lah claude_headless.log

# Ver em tempo real
tail -f claude_headless.log

# Ver últimas 50 linhas
tail -50 claude_headless.log
```

---

## 🎓 Fundamentação Teórica

### Papers Disponíveis (docs/referentials/):
- `ChomskyMinimalistProgram.pdf` - Programa Minimalista
- `chomsky1965-ch1.pdf` - Aspects of Syntax (1965)
- `1905.05950v2.pdf` - Attention is All You Need
- `N19-1419.pdf` - Grammatical structures in NLP

### Análise Universal Grammar (docs/grammar/):
- `UNIVERSAL_GRAMMAR_PROOF.md` - Prova formal
- `CLEAN_ARCHITECTURE_GRAMMAR_ANALYSIS.md` - Análise Clean Arch
- `grammar-patterns.yml` - Padrões gramaticais

Claude Code usa esses recursos como base teórica.

---

## 🏆 Success Criteria

A task será considerada completa quando:

- ✅ Código roda e treina sem erros
- ✅ GrammaticalBERT ≥ vanilla BERT accuracy (GLUE)
- ✅ Hallucination rate reduzido ≥20% (Glass framework)
- ✅ Visualização de constituency structure funcional
- ✅ Código production-ready (tests, docs, types)
- ✅ RFC pronto para PR no Hugging Face

---

## 📈 Próximos Passos (Após Execução)

1. **Review do código**
```bash
cd grammatical_transformers
tree -L 2
```

2. **Rodar testes**
```bash
cd grammatical_transformers
pytest tests/
```

3. **Ver benchmark results**
```bash
cat RESULTS.md
```

4. **Review RFC**
```bash
cat RFC.md
```

5. **Treinar modelo**
```bash
cd grammatical_transformers
python benchmarks/compare_vanilla.py
```

6. **PR para Hugging Face**
- Usar RFC.md como template
- Incluir RESULTS.md
- Adicionar visualizações

---

## 🔗 Referências

### Documentação Oficial:
- [Claude Code Headless Mode](https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)

### Papers:
- Chomsky, N. (1995). The Minimalist Program
- Vaswani et al. (2017). Attention is All You Need
- Glass Framework for Hallucination Detection

---

## 💎 Pro Tips

### 1. Checkpoint automático
```bash
# Adicione ao crontab para backup a cada hora
0 * * * * tar -czf ~/backups/grammatical_$(date +\%H).tar.gz ~/dev/nooa-transformers/grammatical_transformers/
```

### 2. Notificação quando terminar
```bash
# Adicione ao final do run_headless_tmux.sh
; osascript -e 'display notification "Claude Code finished!" with title "GrammaticalTransformers"'
```

### 3. Git auto-commit
```bash
cd grammatical_transformers
git init
watch -n 1800 'git add . && git commit -m "Progress: $(date +\%H:\%M)"'
```

### 4. Resource monitoring
```bash
# Em outro terminal
watch -n 5 'ps aux | grep claude'
```

---

## 🇧🇷 Comando Copy-Paste (Execução Completa)

```bash
# Setup + Execução + Monitoramento
cd /Users/thiagobutignon/dev/nooa-transformers

# Background execution
./run_headless_tmux.sh

# Auto-refresh monitoring (outro terminal)
watch -n 10 './monitor.sh'

# Ou só rode e vá dormir:
./run_headless_tmux.sh && echo "✅ Claude trabalhando. Vai dormir! 😴"
```

---

## ⏱️ Timeline Estimado

| Fase | Tempo Estimado | Atividade |
|------|---------------|-----------|
| **Phase 1** | 4h | Study (Chomsky + BERT analysis) |
| **Phase 2** | 2h | Design (Architecture) |
| **Phase 3** | 10h | Implementation (Coding) |
| **Phase 4** | 4h | Benchmarks (Testing) |
| **Phase 5** | 2h | Documentation (RFC + README) |
| **TOTAL** | **22h** | **~3 overnight sessions** |

---

**Bora transformar Chomsky em código. LFG.** 🚀🔥

---

*Generated with Claude Code Headless Mode*
*October 2025*
