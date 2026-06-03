#!/bin/zsh
# Per-session startup for Qwen3-VL research project.
# Run with:  source start.sh   (NOT  ./start.sh — must be sourced to activate venv)

PROJECT_ROOT="$(cd "$(dirname "${(%):-%x}")" && pwd)"
VENV="$PROJECT_ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
    echo "[start.sh] ERROR: .venv not found. Run the one-time setup first (see setup.md)."
    return 1
fi

source "$VENV/bin/activate"
echo "[start.sh] venv active  : $VIRTUAL_ENV"
echo "[start.sh] Python       : $(python --version)"
echo "[start.sh] Torch        : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "[start.sh] MPS          : $(python -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null || echo 'unknown')"

# Check model weights
MODEL_DIR="$PROJECT_ROOT/models/Qwen3-VL-2B-Instruct"
if [[ -d "$MODEL_DIR" ]]; then
    echo "[start.sh] Model weights : $MODEL_DIR"
else
    echo "[start.sh] Model weights : NOT FOUND — run the download step in setup.md before running evaluate.py"
fi

echo ""
echo "Ready. Working directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
