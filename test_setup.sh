#!/bin/bash

# 🧪 Test Claude Code Headless Setup
# Verifica se tudo está pronto para rodar

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   🧪 Claude Code Headless - Setup Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

ERRORS=0

# Test 1: Claude Code instalado
echo -n "Verificando Claude Code... "
if command -v claude &> /dev/null; then
    VERSION=$(claude --version 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✅ Instalado${NC} ($VERSION)"
else
    echo -e "${RED}❌ Não encontrado${NC}"
    echo "   Instale: npm install -g @anthropic/claude-code"
    ERRORS=$((ERRORS + 1))
fi

# Test 2: tmux instalado
echo -n "Verificando tmux... "
if command -v tmux &> /dev/null; then
    VERSION=$(tmux -V)
    echo -e "${GREEN}✅ Instalado${NC} ($VERSION)"
else
    echo -e "${RED}❌ Não encontrado${NC}"
    echo "   Instale: brew install tmux (Mac) ou apt install tmux (Linux)"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: Scripts executáveis
echo -n "Verificando permissões dos scripts... "
if [[ -x "./run_headless.sh" && -x "./run_headless_tmux.sh" && -x "./monitor.sh" && -x "./stop.sh" ]]; then
    echo -e "${GREEN}✅ OK${NC}"
else
    echo -e "${YELLOW}⚠️  Ajustando...${NC}"
    chmod +x *.sh 2>/dev/null
    if [[ -x "./run_headless.sh" ]]; then
        echo -e "${GREEN}✅ Corrigido${NC}"
    else
        echo -e "${RED}❌ Falhou${NC}"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Test 4: TASK.md existe
echo -n "Verificando TASK.md... "
if [[ -f "TASK.md" ]]; then
    LINES=$(wc -l < TASK.md | xargs)
    echo -e "${GREEN}✅ Existe${NC} ($LINES linhas)"
else
    echo -e "${RED}❌ Não encontrado${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Test 5: Docs disponíveis
echo -n "Verificando docs de referência... "
if [[ -d "docs/referentials" ]]; then
    PDF_COUNT=$(find docs/referentials -name "*.pdf" 2>/dev/null | wc -l | xargs)
    echo -e "${GREEN}✅ OK${NC} ($PDF_COUNT PDFs)"
else
    echo -e "${YELLOW}⚠️  docs/referentials não encontrado${NC}"
fi

# Test 6: Python + PyTorch (opcional, para rodar código depois)
echo -n "Verificando Python... "
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version)
    echo -e "${GREEN}✅ Instalado${NC} ($VERSION)"

    # Check PyTorch
    echo -n "Verificando PyTorch... "
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo -e "${GREEN}✅ Instalado${NC} (v$TORCH_VERSION)"
    else
        echo -e "${YELLOW}⚠️  Não instalado${NC}"
        echo "   (Opcional, mas necessário para rodar o código depois)"
        echo "   Instale: pip install torch transformers datasets"
    fi
else
    echo -e "${YELLOW}⚠️  Python não encontrado${NC}"
    echo "   (Opcional, mas necessário para rodar o código depois)"
fi

# Test 7: Sessões tmux existentes
echo -n "Verificando sessões tmux ativas... "
if command -v tmux &> /dev/null; then
    SESSIONS=$(tmux ls 2>/dev/null | grep "claude_grammatical" | wc -l | xargs)
    if [[ $SESSIONS -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  $SESSIONS sessão(ões) ativa(s)${NC}"
        tmux ls 2>/dev/null | grep "claude_grammatical"
    else
        echo -e "${GREEN}✅ Nenhuma${NC}"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [[ $ERRORS -eq 0 ]]; then
    echo -e "${GREEN}🎉 Setup completo! Tudo pronto para rodar.${NC}"
    echo ""
    echo -e "${YELLOW}Próximo passo:${NC}"
    echo "  ./run_headless_tmux.sh    # Background mode (recomendado)"
    echo "  ./run_headless.sh         # Foreground mode"
    echo ""
    echo -e "${YELLOW}Monitoramento:${NC}"
    echo "  ./monitor.sh              # Ver progresso"
    echo "  watch -n 10 ./monitor.sh  # Auto-refresh"
else
    echo -e "${RED}❌ Setup incompleto. Corrija os erros acima.${NC}"
    echo ""
    echo -e "${YELLOW}Erros encontrados: $ERRORS${NC}"
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exit $ERRORS
