#!/bin/bash

# 🔥 Claude Code Headless Runner - TMUX BACKGROUND VERSION
# Roda Claude Code em background e permite monitoramento

set -e

TASK_FILE="TASK.md"
LOG_FILE="claude_headless.log"
SESSION_NAME="claude_grammatical_$(date +%Y%m%d_%H%M%S)"

# Cores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Claude Code Headless - Background Mode${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verifica dependências
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code não encontrado!"
    exit 1
fi

if ! command -v tmux &> /dev/null; then
    echo "❌ tmux não encontrado! Instale: brew install tmux (Mac) ou apt install tmux (Linux)"
    exit 1
fi

# Lê task
if [ ! -f "$TASK_FILE" ]; then
    echo "❌ $TASK_FILE não encontrado!"
    exit 1
fi

TASK_CONTENT=$(cat "$TASK_FILE")

# Cria sessão tmux
echo -e "${GREEN}📦 Criando sessão tmux: $SESSION_NAME${NC}"

tmux new-session -d -s "$SESSION_NAME" bash -c "
  echo '🤖 Claude Code iniciado em: \$(date)' | tee $LOG_FILE
  echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' | tee -a $LOG_FILE
  echo '' | tee -a $LOG_FILE

  claude -p \"$TASK_CONTENT\" \
    --dangerously-skip-permissions \
    --output-format stream-json \
    --verbose \
    2>&1 | tee -a $LOG_FILE

  echo '' | tee -a $LOG_FILE
  echo '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━' | tee -a $LOG_FILE
  echo '✅ Claude Code finalizado em: \$(date)' | tee -a $LOG_FILE

  # Mantém sessão aberta
  read -p 'Pressione ENTER para fechar...'
"

echo ""
echo -e "${GREEN}✅ Claude Code rodando em background!${NC}"
echo ""
echo -e "${YELLOW}📊 Comandos úteis:${NC}"
echo "  tmux attach -t $SESSION_NAME     # Ver em tempo real"
echo "  tmux kill-session -t $SESSION_NAME    # Parar execução"
echo "  tail -f $LOG_FILE                # Ver logs"
echo "  ./monitor.sh                     # Monitorar progresso"
echo "  ./stop.sh                        # Script de parada"
echo ""
echo -e "${GREEN}💤 Pode ir dormir! Claude tá trabalhando...${NC}"
echo ""

# Salva info da sessão
echo "$SESSION_NAME" > .current_session
