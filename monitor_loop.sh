#!/bin/bash

# 🔄 Monitor Loop - Alternativa ao watch (para macOS)
# Auto-refresh a cada N segundos

INTERVAL=${1:-10}  # Default: 10 segundos

echo "🔄 Monitor auto-refresh iniciado (a cada ${INTERVAL}s)"
echo "Pressione Ctrl+C para parar"
echo ""

while true; do
  clear
  ./monitor.sh
  sleep $INTERVAL
done
