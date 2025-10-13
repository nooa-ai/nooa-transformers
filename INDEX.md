# 📚 Claude Code Headless - Complete Index

**Tudo que você precisa saber sobre execução headless do Claude Code para GrammaticalTransers.**

---

## 🚀 Quick Navigation

### Começando (Start Here):
1. **[QUICKSTART.md](QUICKSTART.md)** ⚡ - 3 comandos para começar
2. **[test_setup.sh](test_setup.sh)** 🧪 - Verifica se tudo está pronto

### Documentação Completa:
3. **[README_HEADLESS.md](README_HEADLESS.md)** 📖 - Documentação completa
4. **[ADVANCED.md](ADVANCED.md)** 🔥 - Técnicas avançadas

### Execução:
5. **[run_headless.sh](run_headless.sh)** ▶️ - Execução foreground
6. **[run_headless_tmux.sh](run_headless_tmux.sh)** 🎭 - Execução background (overnight)

### Monitoramento:
7. **[monitor.sh](monitor.sh)** 📊 - Ver progresso
8. **[stop.sh](stop.sh)** 🛑 - Parar execução

### Task:
9. **[TASK.md](TASK.md)** 📋 - Especificação do projeto

---

## 📖 Docs Overview

### [QUICKSTART.md](QUICKSTART.md)
**O que é:** Início rápido
**Quando usar:** Primeira vez
**Tempo de leitura:** 2 min

**Contém:**
- 3 comandos básicos
- Setup inicial
- O que esperar
- Troubleshooting comum

### [README_HEADLESS.md](README_HEADLESS.md)
**O que é:** Documentação completa
**Quando usar:** Para entender tudo
**Tempo de leitura:** 10 min

**Contém:**
- Pré-requisitos
- Todas as opções de execução
- Flags do Claude Code
- Output esperado
- Fundamentação teórica
- Success criteria
- Próximos passos

### [ADVANCED.md](ADVANCED.md)
**O que é:** Técnicas avançadas
**Quando usar:** Depois de dominar o básico
**Tempo de leitura:** 15 min

**Contém:**
- Multi-task parallel execution
- Docker/VM isolation
- CI/CD integration
- Custom templates
- Claude Agent SDK
- Meta-agent automation
- Observability & metrics

---

## 🛠️ Scripts Overview

### [test_setup.sh](test_setup.sh)
```bash
./test_setup.sh
```
**O que faz:** Verifica setup
**Output:** Status de todas dependências
**Quando rodar:** Antes de começar

**Checa:**
- ✅ Claude Code instalado
- ✅ tmux instalado
- ✅ Scripts executáveis
- ✅ TASK.md existe
- ✅ Docs disponíveis
- ⚠️ Python/PyTorch (opcional)

---

### [run_headless.sh](run_headless.sh)
```bash
./run_headless.sh
```
**O que faz:** Executa Claude Code (foreground)
**Modo:** Interativo (pede confirmações)
**Quando usar:** Primeira execução, debugging

**Características:**
- ✅ Output em tempo real
- ✅ Logs salvos
- ✅ Mais seguro (pede aprovação)
- ❌ Precisa manter terminal aberto

---

### [run_headless_tmux.sh](run_headless_tmux.sh)
```bash
./run_headless_tmux.sh
```
**O que faz:** Executa Claude Code (background)
**Modo:** YOLO (auto-aprova tudo)
**Quando usar:** Execução overnight, produção

**Características:**
- ✅ Roda em background (tmux)
- ✅ Auto-aprova (`--dangerously-skip-permissions`)
- ✅ Pode fechar terminal
- ⚠️ Menos seguro (ambiente isolado recomendado)

---

### [monitor.sh](monitor.sh)
```bash
./monitor.sh
# ou
watch -n 10 ./monitor.sh
```
**O que faz:** Mostra progresso
**Quando usar:** Durante execução

**Mostra:**
- Status da sessão tmux
- Arquivos criados
- Linhas de código
- Estrutura de diretórios
- Últimas linhas do log

---

### [stop.sh](stop.sh)
```bash
./stop.sh
```
**O que faz:** Para execução
**Quando usar:** Para abortar ou após conclusão

**Ações:**
- Mata sessão tmux
- Preserva logs
- Preserva output gerado

---

## 📋 Task Specification

### [TASK.md](TASK.md)

**Objetivo:** Implementar Chomsky's Universal Grammar em Hugging Face Transformers

**Fases:**
1. **Study** (4h) - Chomsky papers + BERT analysis
2. **Design** (2h) - Architecture
3. **Implement** (10h) - Coding
4. **Benchmark** (4h) - Testing
5. **Document** (2h) - RFC + README

**Output esperado:** ~7.6k LOC
- `chomsky/` - Parser, structures, symmetry
- `models/` - GrammaticalBERT, attention
- `benchmarks/` - GLUE, hallucination tests
- `tests/` - Unit tests
- Docs - README, RFC, RESULTS

---

## 🎯 Workflows

### Workflow 1: First Time Setup
```bash
# 1. Verificar setup
./test_setup.sh

# 2. Executar (foreground, ver o que acontece)
./run_headless.sh
```

---

### Workflow 2: Overnight Production
```bash
# 1. Verificar setup
./test_setup.sh

# 2. Backup (segurança)
tar -czf backup_$(date +%Y%m%d).tar.gz .

# 3. Executar background
./run_headless_tmux.sh

# 4. (Opcional) Monitorar em outro terminal
watch -n 10 ./monitor.sh

# 5. Ir dormir 😴

# 6. Acordar e verificar
./monitor.sh
```

---

### Workflow 3: Incremental Development
```bash
# 1. Executar fase 1 apenas
claude -p "Execute only Phase 1 (Study) from TASK.md"

# 2. Revisar
cat claude_headless.log

# 3. Executar fase 2
claude -p "Execute only Phase 2 (Design) from TASK.md"

# 4. Repetir...
```

---

### Workflow 4: Parallel Execution
```bash
# Terminal 1
claude -p "Implement chomsky parser" &

# Terminal 2
claude -p "Implement attention mechanism" &

# Terminal 3
claude -p "Write tests" &

# Wait all
wait
```

---

## 🔍 Troubleshooting Guide

### Issue: Setup test fails
**Solution:**
```bash
# Ver qual passo falhou
./test_setup.sh

# Instalar Claude Code
npm install -g @anthropic/claude-code

# Instalar tmux
brew install tmux  # Mac
sudo apt install tmux  # Linux
```

---

### Issue: Execução não inicia
**Solution:**
```bash
# 1. Verificar se Claude Code funciona
claude --version

# 2. Testar comando manual
claude -p "Hello world"

# 3. Ver logs
cat claude_headless.log

# 4. Tentar foreground mode
./run_headless.sh
```

---

### Issue: Sessão tmux não aparece
**Solution:**
```bash
# 1. Listar todas sessões
tmux ls

# 2. Verificar se arquivo .current_session existe
cat .current_session

# 3. Matar todas sessões e tentar novamente
tmux kill-server
./run_headless_tmux.sh
```

---

### Issue: Nenhum arquivo criado
**Solution:**
```bash
# 1. Ver logs
tail -100 claude_headless.log

# 2. Verificar se TASK.md está correto
cat TASK.md

# 3. Tentar com task simples
claude -p "Create a hello.txt file with 'Hello World'"
```

---

### Issue: Processo travou
**Solution:**
```bash
# 1. Ver última atividade
tail -f claude_headless.log

# 2. Attach na sessão tmux
tmux attach -t $(cat .current_session)

# 3. Ver recursos
ps aux | grep claude

# 4. Se necessário, parar
./stop.sh
```

---

## 📊 Expected Timeline

| Time | Activity | Progress |
|------|----------|----------|
| **00:00** | `./run_headless_tmux.sh` | 🚀 Start |
| **00:30** | Reading Chomsky papers | 📚 Study phase |
| **01:00** | Analyzing BERT source | 🔍 Analysis |
| **02:00** | Designing architecture | 🎨 Design phase |
| **03:00** | Coding ChomskyParser | 💻 Implementation |
| **05:00** | Coding GrammaticalBERT | 💻 Implementation |
| **08:00** | Writing tests | 🧪 Testing |
| **10:00** | Running benchmarks | 📊 Benchmarking |
| **12:00** | Writing docs | 📝 Documentation |
| **14:00** | Final review | ✅ Complete |

**Total:** ~14-22h (depending on complexity)

---

## 🎓 Learning Path

### Beginner:
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `./test_setup.sh`
3. Execute `./run_headless.sh` (foreground)
4. Monitor with `./monitor.sh`

### Intermediate:
1. Read [README_HEADLESS.md](README_HEADLESS.md)
2. Execute `./run_headless_tmux.sh` (background)
3. Try `watch -n 10 ./monitor.sh`
4. Experiment with flags

### Advanced:
1. Read [ADVANCED.md](ADVANCED.md)
2. Multi-task parallel execution
3. Docker/VM isolation
4. Claude Agent SDK integration
5. CI/CD automation

---

## 🔗 External Resources

### Official Docs:
- [Claude Code Headless Mode](https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless)
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)

### Papers (Available locally):
- `docs/referentials/ChomskyMinimalistProgram.pdf`
- `docs/referentials/chomsky1965-ch1.pdf`
- `docs/referentials/1905.05950v2.pdf` (Transformers)
- `docs/referentials/N19-1419.pdf`

### Internal Docs:
- `docs/grammar/UNIVERSAL_GRAMMAR_PROOF.md`
- `docs/grammar/CLEAN_ARCHITECTURE_GRAMMAR_ANALYSIS.md`
- `docs/grammar/grammar-patterns.yml`

---

## ✅ Checklist

### Before Running:
- [ ] Claude Code installed (`claude --version`)
- [ ] tmux installed (`tmux -V`)
- [ ] Scripts executable (`./test_setup.sh`)
- [ ] TASK.md reviewed
- [ ] Backup created (if production)
- [ ] Isolated environment (if YOLO mode)

### During Execution:
- [ ] Monitor logs (`tail -f claude_headless.log`)
- [ ] Check progress (`./monitor.sh`)
- [ ] Verify files being created (`ls grammatical_transformers/`)
- [ ] Monitor resources (`ps aux | grep claude`)

### After Completion:
- [ ] Review output (`tree grammatical_transformers/`)
- [ ] Check results (`cat RESULTS.md`)
- [ ] Run tests (`pytest grammatical_transformers/tests/`)
- [ ] Review RFC (`cat RFC.md`)
- [ ] Benchmark (`python benchmarks/compare_vanilla.py`)

---

## 🇧🇷 Copy-Paste Commands

### Complete Workflow:
```bash
# 1. Setup
cd /Users/thiagobutignon/dev/nooa-transformers
./test_setup.sh

# 2. Backup
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz .

# 3. Execute (background)
./run_headless_tmux.sh

# 4. Monitor (outro terminal)
watch -n 10 ./monitor.sh

# 5. Quando terminar
./monitor.sh
cat grammatical_transformers/RESULTS.md
cat grammatical_transformers/RFC.md

# 6. Test
cd grammatical_transformers
pytest tests/ -v

# 7. Benchmark
python benchmarks/compare_vanilla.py
```

---

## 📈 Success Metrics

Task completa quando:
- ✅ ~7.6k LOC gerados
- ✅ Todos testes passando (>80% coverage)
- ✅ GrammaticalBERT ≥ vanilla BERT (GLUE)
- ✅ Hallucination -20% (Glass framework)
- ✅ Documentação completa (README + RFC)
- ✅ Código production-ready (types, tests, docs)

---

## 🎯 Final Notes

**Este é um projeto completo de automação Claude Code Headless.**

**Você tem:**
- ✅ Scripts prontos para uso
- ✅ Documentação completa
- ✅ Troubleshooting guide
- ✅ Workflows testados
- ✅ Advanced techniques

**Próximo passo:**
```bash
./test_setup.sh && ./run_headless_tmux.sh
```

**Então:** Vai dormir. 😴

**Acordar:** 7.6k LOC prontos. ✅

---

**LFG.** 🚀🔥💎

---

*Claude Code Headless - Complete Documentation*
*October 2025*
