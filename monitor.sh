#!/bin/bash

# 🔍 Claude Code Progress Monitor

LOG_FILE="claude_headless.log"
OUTPUT_DIR="grammatical_transformers"
SESSION_FILE=".current_session"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   🤖 Claude Code Progress Monitor${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Status da sessão tmux
if [ -f "$SESSION_FILE" ]; then
    SESSION_NAME=$(cat "$SESSION_FILE")
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${GREEN}✅ Claude Code está rodando${NC} (sessão: $SESSION_NAME)"
    else
        echo -e "${RED}❌ Sessão finalizada${NC} (última: $SESSION_NAME)"
    fi
else
    echo -e "${YELLOW}⚠️  Nenhuma sessão encontrada${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📁 Arquivos Criados${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "$OUTPUT_DIR" ]; then
    FILE_COUNT=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l | xargs)
    echo "Total de arquivos: $FILE_COUNT"
    echo ""

    # Conta por extensão
    echo "Por tipo:"
    find "$OUTPUT_DIR" -type f -name "*.py" 2>/dev/null | wc -l | xargs echo "  .py files:"
    find "$OUTPUT_DIR" -type f -name "*.md" 2>/dev/null | wc -l | xargs echo "  .md files:"
    find "$OUTPUT_DIR" -type f -name "*.json" 2>/dev/null | wc -l | xargs echo "  .json files:"
    find "$OUTPUT_DIR" -type f -name "*.yml" -o -name "*.yaml" 2>/dev/null | wc -l | xargs echo "  .yml files:"
else
    echo -e "${YELLOW}⚠️  Diretório $OUTPUT_DIR ainda não criado${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📝 Linhas de Código${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "$OUTPUT_DIR" ]; then
    PY_LINES=$(find "$OUTPUT_DIR" -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    if [ -n "$PY_LINES" ]; then
        echo "Python: $PY_LINES linhas"
    else
        echo "Python: 0 linhas"
    fi

    # Total geral
    TOTAL_LINES=$(find "$OUTPUT_DIR" -type f -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    if [ -n "$TOTAL_LINES" ]; then
        echo "Total: $TOTAL_LINES linhas"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📂 Estrutura de Diretórios${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "$OUTPUT_DIR" ]; then
    if command -v tree &> /dev/null; then
        tree "$OUTPUT_DIR" -L 3 --charset ascii
    else
        find "$OUTPUT_DIR" -type f | head -20
        echo ""
        echo "(Instale 'tree' para visualização melhor: brew install tree)"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📋 Últimas 15 linhas do log${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
else
    echo -e "${YELLOW}⚠️  Log ainda não criado${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⚡ Comandos Úteis${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  ./monitor.sh              # Rodar este script novamente"
echo "  tail -f $LOG_FILE         # Ver logs em tempo real"
echo "  ./stop.sh                 # Parar execução"

if [ -f "$SESSION_FILE" ]; then
    echo "  tmux attach -t $(cat $SESSION_FILE)  # Conectar na sessão"
fi

echo ""
echo -e "${GREEN}💡 Tip: Execute 'watch -n 10 ./monitor.sh' para atualização automática${NC}"
echo ""
