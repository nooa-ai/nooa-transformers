#!/bin/bash

# 🛑 Stop Claude Code Headless Session

SESSION_FILE=".current_session"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Parando Claude Code...${NC}"
echo ""

if [ -f "$SESSION_FILE" ]; then
    SESSION_NAME=$(cat "$SESSION_FILE")

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "${GREEN}✅ Sessão $SESSION_NAME encerrada${NC}"
    else
        echo -e "${YELLOW}⚠️  Sessão $SESSION_NAME já estava encerrada${NC}"
    fi

    rm "$SESSION_FILE"
else
    echo -e "${YELLOW}⚠️  Nenhuma sessão ativa encontrada${NC}"
fi

echo ""
echo -e "${GREEN}📊 Resultados salvos em:${NC}"
echo "  ./grammatical_transformers/  # Código gerado"
echo "  ./claude_headless.log        # Logs completos"
echo ""
echo -e "${GREEN}🔍 Ver resultados:${NC}"
echo "  ls -la grammatical_transformers/"
echo "  cat RESULTS.md                # Se foi gerado"
echo "  cat RFC.md                    # Se foi gerado"
echo ""
