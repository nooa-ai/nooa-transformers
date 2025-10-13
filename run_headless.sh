#!/bin/bash

# 🔥 Claude Code Headless Runner (REAL VERSION)
# Based on: https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless

set -e

# Configuração
TASK_FILE="TASK.md"
OUTPUT_DIR="grammatical_transformers"
LOG_FILE="claude_headless.log"
SESSION_NAME="claude_grammatical_$(date +%Y%m%d_%H%M%S)"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Claude Code Headless Mode - GrammaticalTransformers${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verifica se Claude Code está instalado
if ! command -v claude &> /dev/null; then
    echo -e "${RED}❌ Claude Code não encontrado!${NC}"
    echo "Instale com: npm install -g @anthropic/claude-code"
    exit 1
fi

# Limpa output anterior (opcional)
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Diretório $OUTPUT_DIR já existe. Removendo...${NC}"
    rm -rf "$OUTPUT_DIR"
fi

# Cria diretório de output
mkdir -p "$OUTPUT_DIR"

# Lê o TASK.md
if [ ! -f "$TASK_FILE" ]; then
    echo -e "${RED}❌ Arquivo $TASK_FILE não encontrado!${NC}"
    exit 1
fi

TASK_CONTENT=$(cat "$TASK_FILE")

echo -e "${GREEN}✅ Task carregada de $TASK_FILE${NC}"
echo -e "${GREEN}✅ Output em: $OUTPUT_DIR${NC}"
echo -e "${GREEN}✅ Logs em: $LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}⚡ Iniciando Claude Code em modo YOLO (auto-approve)...${NC}"
echo ""

# OPÇÃO 1: Modo YOLO COMPLETO (Auto-approve tudo)
# ⚠️  PERIGO: Usa apenas em ambiente isolado!

# claude -p "$TASK_CONTENT" \
#   --dangerously-skip-permissions \
#   --output-format stream-json \
#   2>&1 | tee -a "$LOG_FILE"

# OPÇÃO 2: Modo Headless com aprovações (mais seguro)
# Remove --dangerously-skip-permissions para pedir confirmações

claude -p "$TASK_CONTENT" \
  --output-format stream-json \
  --verbose \
  2>&1 | tee -a "$LOG_FILE"

# OPÇÃO 3: Usar tmux para sessão persistente
# Descomente para rodar em background com tmux

# echo -e "${GREEN}📦 Iniciando sessão tmux: $SESSION_NAME${NC}"
# tmux new-session -d -s "$SESSION_NAME" "
#   claude -p \"$TASK_CONTENT\" \
#     --dangerously-skip-permissions \
#     --output-format stream-json \
#     2>&1 | tee -a $LOG_FILE
# "

# echo ""
# echo -e "${GREEN}✅ Claude Code rodando em background!${NC}"
# echo -e "${YELLOW}Comandos úteis:${NC}"
# echo "  tmux attach -t $SESSION_NAME   # Conectar na sessão"
# echo "  tmux kill-session -t $SESSION_NAME  # Parar execução"
# echo "  tail -f $LOG_FILE              # Ver logs em tempo real"
# echo "  ./monitor.sh                   # Monitorar progresso"

echo ""
echo -e "${GREEN}🎉 Execução completa!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Resultados em: ./$OUTPUT_DIR${NC}"
echo -e "${GREEN}Logs completos: ./$LOG_FILE${NC}"
