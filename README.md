# 🤖 GrammaticalTransformers - Claude Code Headless Automation

**Implementação automática de Chomsky's Universal Grammar em Hugging Face Transformers usando Claude Code em modo headless.**

---

## 🎯 TL;DR

```bash
# 1. Verificar setup
./test_setup.sh

# 2. Executar (modo background overnight)
./run_headless_tmux.sh

# 3. Acordar e ver resultados
./monitor.sh
```

**Isso é tudo. Claude trabalha enquanto você dorme.** 😴

---

## 📚 Documentação

| Arquivo | Descrição | Quando Usar |
|---------|-----------|-------------|
| **[INDEX.md](INDEX.md)** | 📖 Índice completo | Navegação geral |
| **[QUICKSTART.md](QUICKSTART.md)** | ⚡ Início rápido (3 comandos) | Primeira vez |
| **[README_HEADLESS.md](README_HEADLESS.md)** | 📚 Docs completas | Entender tudo |
| **[ADVANCED.md](ADVANCED.md)** | 🔥 Técnicas avançadas | Otimização |
| **[TASK.md](TASK.md)** | 📋 Especificação do projeto | Ver o que será feito |

---

## 🛠️ Scripts

| Script | Função |
|--------|--------|
| `./test_setup.sh` | 🧪 Verifica dependências |
| `./run_headless.sh` | ▶️ Executa (foreground) |
| `./run_headless_tmux.sh` | 🎭 Executa (background) |
| `./monitor.sh` | 📊 Monitora progresso |
| `./stop.sh` | 🛑 Para execução |

---

## 🚀 Quick Start

### Pré-requisitos:
```bash
npm install -g @anthropic/claude-code
brew install tmux  # ou apt install tmux
```

### Executar:
```bash
./run_headless_tmux.sh
```

### Monitorar:
```bash
watch -n 10 ./monitor.sh
```

---

## 📊 O que será gerado?

```
grammatical_transformers/
├── chomsky/
│   ├── parser.py           (~800 LOC)
│   ├── structures.py       (~600 LOC)
│   └── symmetry.py         (~400 LOC)
├── models/
│   ├── grammatical_bert.py (~1200 LOC)
│   └── attention.py        (~900 LOC)
├── benchmarks/
│   ├── glue_test.py
│   ├── hallucination_test.py
│   └── compare_vanilla.py
├── tests/                  (~1500 LOC)
├── README.md
├── RFC.md
└── RESULTS.md

Total: ~7,600 LOC
```

---

## 🎓 Fundamentação Teórica

### Papers Disponíveis (`docs/referentials/`):
- `ChomskyMinimalistProgram.pdf`
- `chomsky1965-ch1.pdf`
- `1905.05950v2.pdf` (Attention is All You Need)
- `N19-1419.pdf`

### Universal Grammar Analysis (`docs/grammar/`):
- `UNIVERSAL_GRAMMAR_PROOF.md`
- `CLEAN_ARCHITECTURE_GRAMMAR_ANALYSIS.md`
- `grammar-patterns.yml`

---

## ✅ Success Criteria

- ✅ GrammaticalBERT ≥ vanilla BERT accuracy
- ✅ Hallucination rate -20%
- ✅ Test coverage >80%
- ✅ Production-ready code
- ✅ RFC pronto para Hugging Face PR

---

## 🔗 Links

- [Claude Code Docs](https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless)
- [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 📈 Timeline

| Fase | Tempo | Output |
|------|-------|--------|
| Study | 4h | Analysis docs |
| Design | 2h | Architecture |
| Implement | 10h | Code (~7.6k LOC) |
| Benchmark | 4h | Results |
| Document | 2h | RFC + README |
| **Total** | **22h** | **Complete project** |

---

**Bora! Execute e durma.** 🚀😴

**Acordar:** Transformers com Gramática Universal implementada. ✅

---

*October 2025 - Claude Code Headless Automation*
