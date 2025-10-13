# ⚡ QUICKSTART - Claude Code Headless

**3 comandos. Claude trabalha. Você dorme.** 😴

---

## 🎯 TL;DR

```bash
# 1. Executar (modo background)
./run_headless_tmux.sh

# 2. Monitorar (outro terminal - opcional)
watch -n 10 ./monitor.sh

# 3. Acordar e ver resultados
./monitor.sh
cat grammatical_transformers/RESULTS.md
```

**Isso é tudo.** 🔥

---

## 🚀 Setup Inicial (Só uma vez)

### Mac/Linux:
```bash
# Instalar dependências
npm install -g @anthropic/claude-code
brew install tmux  # ou: apt install tmux

# Verificar instalação
claude --version
tmux -V
```

### Primeira execução:
```bash
cd /Users/thiagobutignon/dev/nooa-transformers

# Dar permissões (se ainda não fez)
chmod +x *.sh

# Rodar!
./run_headless_tmux.sh
```

---

## 📊 O que vai acontecer?

```
[22:00] 🚀 ./run_headless_tmux.sh
[22:01] 📚 Claude lê Chomsky papers (docs/referentials/)
[22:30] 🔍 Claude analisa BERT source code
[23:00] 💻 Claude começa a codar GrammaticalBERT
[02:00] 🧪 Claude roda benchmarks
[04:00] 📝 Claude escreve docs + RFC
[06:00] ✅ DONE! ~7.6k LOC prontos
```

**Você:** 😴😴😴

---

## 🔍 Monitoramento (Enquanto roda)

### Terminal 1: Execução
```bash
./run_headless_tmux.sh
# Deixa rodando...
```

### Terminal 2: Monitoramento (opcional)
```bash
# Opção A: Auto-refresh
watch -n 10 ./monitor.sh

# Opção B: Manual
./monitor.sh  # Roda quando quiser

# Opção C: Live logs
tail -f claude_headless.log

# Opção D: Attach tmux (ver em tempo real)
tmux attach -t claude_grammatical_*
# (Ctrl+B, D para detach)
```

---

## 🛑 Parar Execução

```bash
./stop.sh

# Ou manual:
tmux kill-session -t claude_grammatical_*
```

---

## 📈 Ver Resultados

```bash
# Estrutura gerada
tree grammatical_transformers -L 2

# Código principal
ls -lah grammatical_transformers/models/

# Benchmarks
cat grammatical_transformers/RESULTS.md

# RFC para Hugging Face
cat grammatical_transformers/RFC.md

# Logs completos
cat claude_headless.log
```

---

## 🎯 Arquivos Importantes

```
.
├── TASK.md                     # ← Especificação da task
├── run_headless_tmux.sh        # ← Executar (background)
├── run_headless.sh             # ← Executar (foreground)
├── monitor.sh                  # ← Monitorar progresso
├── stop.sh                     # ← Parar execução
├── README_HEADLESS.md          # ← Docs completos
├── QUICKSTART.md               # ← Este arquivo
│
├── docs/                       # ← Teoria (Chomsky, Grammar)
│   ├── referentials/*.pdf
│   └── grammar/*.md
│
└── grammatical_transformers/   # ← Output (gerado pelo Claude)
    ├── chomsky/
    ├── models/
    ├── benchmarks/
    ├── tests/
    ├── README.md
    ├── RFC.md
    └── RESULTS.md
```

---

## ⚙️ Modos de Execução

### Modo 1: Foreground (Interativo)
```bash
./run_headless.sh
```
- Roda no terminal atual
- Você vê output em tempo real
- **Pede confirmações** (mais seguro)
- Precisa manter terminal aberto

### Modo 2: Background (Tmux + YOLO)
```bash
./run_headless_tmux.sh
```
- Roda em sessão tmux separada
- **Auto-aprova tudo** (`--dangerously-skip-permissions`)
- Pode fechar terminal
- ⚠️ **Menos seguro** - use em ambiente isolado

### Modo 3: Custom
```bash
# Edite run_headless.sh e descomente opções
# Escolha flags específicas
```

---

## 🔥 Pro Tips

### 1. Múltiplas tasks em paralelo
```bash
# Terminal 1
./run_headless_tmux.sh

# Terminal 2 (outra task)
claude -p "Implement benchmarks only" \
  --output-format stream-json \
  2>&1 | tee benchmarks.log
```

### 2. Continuar de onde parou
```bash
# Se interrompeu, retoma última sessão
claude --continue

# Ou sessão específica
claude --resume <session-id>
```

### 3. Checkpoint manual
```bash
# Durante execução, em outro terminal
tar -czf checkpoint_$(date +%H%M).tar.gz grammatical_transformers/
```

### 4. Notificação macOS quando terminar
```bash
# Adicione ao final do tmux command em run_headless_tmux.sh:
; osascript -e 'display notification "Done!" with title "Claude Code"'
```

---

## 🐛 Debug

### Problema: Nada acontece
```bash
# Verificar se tmux rodando
tmux ls

# Ver logs
cat claude_headless.log

# Tentar foreground mode
./run_headless.sh
```

### Problema: Erro de permissão
```bash
chmod +x *.sh
./run_headless_tmux.sh
```

### Problema: Claude não encontrado
```bash
npm install -g @anthropic/claude-code
which claude
```

---

## 📚 Next Steps

Depois que Claude terminar:

### 1. Review
```bash
cd grammatical_transformers
code .  # Abrir no VS Code
```

### 2. Test
```bash
cd grammatical_transformers
pytest tests/ -v
```

### 3. Run
```bash
python benchmarks/compare_vanilla.py
```

### 4. Deploy
```bash
# Ver RFC.md para instruções de PR
cat RFC.md
```

---

## 🎓 Entendendo a Task

### O que Claude vai fazer?

**Objetivo:** Implementar Chomsky's Universal Grammar em Transformers

**Como:**
1. **ChomskyParser** - Merge operation (constituency parsing)
2. **GrammaticalAttention** - Constituency-aware attention masks
3. **SymmetryLoss** - Grammatical consistency (reduz hallucination)

**Por quê:**
- Transformers não têm estrutura gramatical explícita
- Chomsky provou que linguagem tem deep structure universal
- Aplicar isso pode melhorar interpretability e reduzir hallucination

**Base teórica:**
- docs/referentials/ChomskyMinimalistProgram.pdf
- docs/grammar/UNIVERSAL_GRAMMAR_PROOF.md

---

## 🇧🇷 Copy-Paste Final

**Mac:**
```bash
cd /Users/thiagobutignon/dev/nooa-transformers
./run_headless_tmux.sh && echo "✅ Rodando! Vai dormir 😴"
```

**Linux:**
```bash
cd ~/dev/nooa-transformers
./run_headless_tmux.sh && echo "✅ Rodando! Vai dormir 😴"
```

---

## ⏱️ Expectativa de Tempo

- **Setup:** 5 min (só primeira vez)
- **Execução:** 22h (overnight x3)
- **Review:** 30 min
- **Testing:** 1h
- **Deploy:** 2h

**Total:** ~3-4 dias (maioria rodando sozinho)

---

**Pronto. É isso.**

**Execute. Durma. Acorde com 7.6k LOC prontos.**

**LFG.** 🚀💎

---

*Claude Code Headless - Because sleep is productive* 😴🤖
