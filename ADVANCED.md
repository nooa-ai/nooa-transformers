# 🚀 Advanced Claude Code Headless Usage

Além do básico, aqui estão técnicas avançadas para otimizar o fluxo headless.

---

## 🎯 Multi-Task Parallel Execution

### Executar múltiplas tasks em paralelo:

```bash
# Terminal 1: Main implementation
claude -p "$(cat TASK.md)" \
  --dangerously-skip-permissions \
  --output-format stream-json \
  2>&1 | tee main.log &

# Terminal 2: Benchmarks isolados
claude -p "Implement only the benchmarks from TASK.md. Focus on GLUE and hallucination tests." \
  --output-format stream-json \
  2>&1 | tee benchmarks.log &

# Terminal 3: Documentation
claude -p "Write comprehensive README and RFC based on the code in grammatical_transformers/" \
  --output-format stream-json \
  2>&1 | tee docs.log &

# Aguardar todos
wait
echo "✅ All tasks complete!"
```

---

## 🔄 Resume & Continue Sessions

### Continuar última sessão:
```bash
claude --continue
```

### Continuar sessão específica:
```bash
# Listar sessões
claude --list-sessions

# Resumir sessão por ID
claude --resume <session-id>
```

### Script para auto-resume se falhar:
```bash
#!/bin/bash
# auto_resume.sh

while true; do
  claude -p "$(cat TASK.md)" \
    --dangerously-skip-permissions \
    --output-format stream-json \
    2>&1 | tee -a claude.log

  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Completed successfully"
    break
  else
    echo "⚠️  Error detected. Resuming in 10s..."
    sleep 10
    claude --continue
  fi
done
```

---

## 📊 Structured Output Processing

### Parse JSON output:
```bash
claude -p "List all Python files in grammatical_transformers/" \
  --output-format stream-json | \
  jq -r '.files[] | select(.extension == ".py") | .path'
```

### Extract metrics:
```bash
claude -p "Run benchmarks and output results as JSON" \
  --output-format stream-json | \
  jq '.results.accuracy'
```

### Save structured results:
```bash
claude -p "$(cat TASK.md)" \
  --output-format stream-json > output.json

# Process later
jq '.summary' output.json
jq '.files_created[]' output.json
```

---

## 🔐 Security & Isolation

### Docker container for YOLO mode:
```dockerfile
# Dockerfile
FROM node:20-slim

RUN npm install -g @anthropic/claude-code

WORKDIR /workspace

# Copy only necessary files
COPY TASK.md ./
COPY docs/ ./docs/

# Run in isolated environment
CMD ["claude", "-p", "$(cat TASK.md)", \
     "--dangerously-skip-permissions", \
     "--output-format", "stream-json"]
```

```bash
# Build & run
docker build -t claude-headless .
docker run -v $(pwd)/output:/workspace/output claude-headless
```

### VM isolation:
```bash
# Create isolated VM
multipass launch --name claude-vm --cpus 4 --mem 8G --disk 50G

# Copy files
multipass transfer TASK.md claude-vm:/home/ubuntu/
multipass transfer docs claude-vm:/home/ubuntu/docs

# SSH and run
multipass exec claude-vm -- bash -c "
  npm install -g @anthropic/claude-code
  cd /home/ubuntu
  claude -p \"\$(cat TASK.md)\" --dangerously-skip-permissions
"

# Retrieve results
multipass transfer claude-vm:/home/ubuntu/grammatical_transformers ./output
```

---

## ⚡ Performance Optimization

### Limit allowed tools:
```bash
claude -p "$(cat TASK.md)" \
  --allowedTools "read,write,bash,grep" \
  --output-format stream-json
```

### Set timeout:
```bash
timeout 8h claude -p "$(cat TASK.md)" \
  --dangerously-skip-permissions
```

### Resource monitoring:
```bash
# Monitor CPU/Memory while running
watch -n 5 'ps aux | grep claude | grep -v grep'

# Log resource usage
while true; do
  ps aux | grep claude | grep -v grep >> resources.log
  sleep 60
done &
```

---

## 🔧 Custom Task Templates

### Template system:
```bash
# templates/implement.md
Implement {{MODULE}} following these rules:
- Architecture: Clean Architecture
- Tests: pytest with >80% coverage
- Docs: Google docstring format
- Type hints: Full mypy compliance

Context available:
{{CONTEXT_FILES}}
```

```bash
# generate_task.sh
#!/bin/bash

MODULE=$1
CONTEXT_FILES=$(find docs -name "*.md" -o -name "*.pdf")

envsubst < templates/implement.md > TASK_${MODULE}.md

claude -p "$(cat TASK_${MODULE}.md)" \
  --dangerously-skip-permissions \
  --output-format stream-json
```

Usage:
```bash
./generate_task.sh ChomskyParser
./generate_task.sh GrammaticalAttention
```

---

## 📈 Incremental Development

### Phase-by-phase execution:
```bash
# Phase 1: Study
claude -p "Execute only Phase 1 (Study) from TASK.md" \
  --output-format stream-json > phase1.json

# Phase 2: Design (after reviewing Phase 1)
claude -p "Execute Phase 2 (Design) using insights from phase1.json" \
  --output-format stream-json > phase2.json

# Phase 3: Implement
claude -p "Execute Phase 3 (Implementation) using design from phase2.json" \
  --dangerously-skip-permissions \
  --output-format stream-json > phase3.json

# And so on...
```

### Checkpoint-based approach:
```bash
#!/bin/bash
# incremental_execute.sh

PHASES=("Study" "Design" "Implement" "Benchmark" "Document")

for i in "${!PHASES[@]}"; do
  PHASE="${PHASES[$i]}"

  echo "📍 Starting Phase $((i+1)): $PHASE"

  claude -p "Execute only Phase $((i+1)) ($PHASE) from TASK.md" \
    --output-format stream-json \
    2>&1 | tee "phase_${i}_${PHASE}.log"

  # Checkpoint
  tar -czf "checkpoint_phase_${i}.tar.gz" grammatical_transformers/

  echo "✅ Phase $((i+1)) complete. Checkpoint saved."
  echo ""

  # Optional: Review before continuing
  read -p "Continue to next phase? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Paused. Resume with: ./incremental_execute.sh --start-from $((i+1))"
    exit 0
  fi
done

echo "🎉 All phases complete!"
```

---

## 🤖 Claude Agent SDK Integration

### Build custom agent using Claude Agent SDK:
```typescript
// claude-agent.ts
import { ClaudeAgent } from '@anthropic/claude-agent-sdk';

const agent = new ClaudeAgent({
  model: 'claude-sonnet-4-5',
  tools: ['bash', 'read', 'write', 'grep'],
  systemPrompt: `
    You are a specialist in implementing Chomsky's grammar in ML systems.
    Your task is to create GrammaticalTransformers.
    Base your work on:
    - docs/referentials/ChomskyMinimalistProgram.pdf
    - docs/grammar/UNIVERSAL_GRAMMAR_PROOF.md
  `
});

async function main() {
  const task = await fs.readFile('TASK.md', 'utf-8');

  const result = await agent.execute({
    prompt: task,
    outputFormat: 'json',
    autoApprove: true, // YOLO mode
    checkpoints: {
      enabled: true,
      interval: 3600000 // 1h
    }
  });

  console.log('✅ Task complete!');
  console.log('Files created:', result.filesCreated.length);
  console.log('LOC:', result.linesOfCode);

  await fs.writeFile('RESULTS.json', JSON.stringify(result, null, 2));
}

main();
```

Run:
```bash
npm install @anthropic/claude-agent-sdk
npx tsx claude-agent.ts
```

---

## 🧪 Testing & Validation

### Dry-run mode:
```bash
# Simulate without writing files
claude -p "$(cat TASK.md)" \
  --dry-run \
  --output-format stream-json | \
  jq '.plannedActions'
```

### Validate output:
```bash
# After execution, validate
claude -p "Review the code in grammatical_transformers/ and report:
1. Test coverage
2. Type hint completeness
3. Documentation quality
4. Architecture compliance" \
  --output-format stream-json > validation.json

# Parse results
jq '.validation' validation.json
```

### Auto-fix issues:
```bash
# Run validation, then auto-fix
claude -p "Fix all issues found in validation.json" \
  --dangerously-skip-permissions
```

---

## 📡 Remote Execution

### SSH to remote server:
```bash
# On remote server
ssh user@server << 'EOF'
  cd /path/to/project
  claude -p "$(cat TASK.md)" \
    --dangerously-skip-permissions \
    --output-format stream-json \
    2>&1 | tee remote.log
EOF
```

### With tmux (persistent):
```bash
ssh user@server << 'EOF'
  tmux new-session -d -s claude_remote "
    cd /path/to/project
    claude -p \"\$(cat TASK.md)\" \
      --dangerously-skip-permissions \
      --output-format stream-json \
      2>&1 | tee remote.log
  "
  echo "✅ Started on remote. Attach with: tmux attach -t claude_remote"
EOF
```

---

## 🔄 CI/CD Integration

### GitHub Actions:
```yaml
# .github/workflows/claude-code.yml
name: Claude Code Automation

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      task:
        description: 'Task to execute'
        required: true

jobs:
  claude-execute:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install Claude Code
        run: npm install -g @anthropic/claude-code

      - name: Execute Task
        run: |
          claude -p "${{ github.event.inputs.task || '$(cat TASK.md)' }}" \
            --dangerously-skip-permissions \
            --output-format stream-json \
            2>&1 | tee output.log

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: claude-output
          path: |
            grammatical_transformers/
            output.log
```

### GitLab CI:
```yaml
# .gitlab-ci.yml
claude-automation:
  image: node:20-slim
  script:
    - npm install -g @anthropic/claude-code
    - |
      claude -p "$(cat TASK.md)" \
        --dangerously-skip-permissions \
        --output-format stream-json \
        2>&1 | tee output.log
  artifacts:
    paths:
      - grammatical_transformers/
      - output.log
    expire_in: 1 week
```

---

## 📊 Metrics & Observability

### Track progress:
```bash
# progress_tracker.sh
#!/bin/bash

while true; do
  TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

  # Files count
  FILES=$(find grammatical_transformers -type f 2>/dev/null | wc -l | xargs)

  # LOC
  LOC=$(find grammatical_transformers -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

  # Log
  echo "$TIMESTAMP,$FILES,$LOC" >> metrics.csv

  sleep 300 # Every 5 min
done
```

```bash
# Start tracking
./progress_tracker.sh &

# Visualize later
gnuplot << 'EOF'
set datafile separator ","
set xdata time
set timefmt "%Y-%m-%d_%H:%M:%S"
set format x "%H:%M"
set terminal png size 800,600
set output 'progress.png'
plot 'metrics.csv' using 1:3 with lines title 'LOC over time'
EOF
```

---

## 🎯 Advanced Use Cases

### 1. Code Migration
```bash
claude -p "Migrate all code from JavaScript to TypeScript while maintaining functionality" \
  --dangerously-skip-permissions
```

### 2. Architecture Refactoring
```bash
claude -p "Refactor codebase to Clean Architecture with:
- Domain layer
- Data layer
- Infrastructure layer
- Presentation layer
Maintain 100% test coverage." \
  --dangerously-skip-permissions
```

### 3. Documentation Generation
```bash
claude -p "Generate comprehensive documentation for grammatical_transformers/:
- API reference
- Architecture diagram
- Usage examples
- Contributing guide" \
  --output-format stream-json
```

### 4. Security Audit
```bash
claude -p "Perform security audit on codebase:
- Check for SQL injection
- Validate input sanitization
- Review authentication flows
- Check dependency vulnerabilities
Output: SECURITY_AUDIT.md" \
  --output-format stream-json
```

---

## 💡 Pro Tips

1. **Always checkpoint before YOLO mode**
   ```bash
   tar -czf pre_yolo_backup.tar.gz .
   ./run_headless_tmux.sh
   ```

2. **Use `.claudeignore` to exclude files**
   ```bash
   echo "node_modules/" > .claudeignore
   echo "*.log" >> .claudeignore
   echo "*.pyc" >> .claudeignore
   ```

3. **Rate limiting & quotas**
   ```bash
   # Add delays between retries
   for i in {1..5}; do
     claude -p "Phase $i" --output-format stream-json
     sleep 600 # 10 min between phases
   done
   ```

4. **Combine with other AI tools**
   ```bash
   # Claude does implementation
   ./run_headless.sh

   # GPT-4 does review
   gh copilot suggest -t shell "Review code in grammatical_transformers/"
   ```

---

## 🚀 Next-Level Automation

### Meta-agent that spawns Claude instances:
```python
# meta_agent.py
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

tasks = [
    "Implement ChomskyParser",
    "Implement GrammaticalAttention",
    "Implement SymmetryLoss",
    "Write tests for all modules",
    "Write benchmarks",
    "Write documentation"
]

def execute_task(task):
    result = subprocess.run([
        'claude', '-p', task,
        '--output-format', 'stream-json'
    ], capture_output=True, text=True)

    return json.loads(result.stdout)

# Run all tasks in parallel
with ThreadPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(execute_task, tasks))

print(f"✅ Completed {len(results)} tasks")
```

---

**Agora você tem o arsenal completo.** 🔥

**Headless. Parallel. Automated. Isolated.**

**Let Claude cook while you sleep.** 😴🤖

---

*Advanced Claude Code Techniques - October 2025*
