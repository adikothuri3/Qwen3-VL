# Setup Guide — Qwen3-VL Research

## Every Session (30 seconds)

Open Terminal, navigate to the project, source the startup script:

```bash
cd /Users/sandhyakothuri/Qwen3-VL
source start.sh
```

That's it. The script activates the venv, prints your Python/torch/MPS status, and confirms model weights are present.

---

## One-Time Setup (already done — reference only)

### Environment created
- Python 3.13 venv at `.venv/`
- PyTorch 2.12 with MPS (Apple M1 GPU) support
- All packages from `requirements.txt` installed
- `local_transformers` wiring verified — model class loads from local source

### Verify wiring at any time
```bash
source start.sh
python -c "
import sys; sys.path.insert(0, '.')
import local_transformers; sys.modules['transformers'] = local_transformers
from local_transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
import inspect; print(inspect.getfile(Qwen3VLForConditionalGeneration))
"
```
Should print: `.../Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`

---

## Model Download (blocked — need ~5 GB free disk space first)

Your disk is currently 99% full. Free at least 5 GB, then run:

```bash
source start.sh

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-VL-2B-Instruct',
    local_dir='models/Qwen3-VL-2B-Instruct',
    ignore_patterns=['*.pt', '*.bin'],
)
print('Done.')
"
```

The model downloads to `models/Qwen3-VL-2B-Instruct/` (~4.5 GB, safetensors format).
`models/` is in `.gitignore` — weights will never be committed.

If you see a token prompt, get a read token from huggingface.co/settings/tokens and paste it.

---

## Running the Evaluation Script

```bash
source start.sh

# CPU (slow but always works on M1)
python src/evaluate.py \
    --model-id models/Qwen3-VL-2B-Instruct \
    --device cpu \
    --dtype float32 \
    --max-new-tokens 32 \
    --num-samples 1 \
    --no-torch-profiler

# MPS — Apple M1 GPU (faster)
python src/evaluate.py \
    --model-id models/Qwen3-VL-2B-Instruct \
    --device mps \
    --dtype float32 \
    --max-new-tokens 64 \
    --num-samples 1
```

First line of output will always be:
```
[local_transformers] Model loaded from: .../local_transformers/models/qwen3_vl/modeling_qwen3_vl.py
```
This confirms your local model edits are active.

---

## How Model Edits Work

1. Edit any file in `local_transformers/models/qwen3_vl/`
2. Run `python src/evaluate.py` — your changes are live immediately
3. No reinstall, no rebuild needed

The key file for DeepStack research: `local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`

---

## Project File Map

```
src/
  evaluate.py              ← main entry point (profiling + evaluation)
  deepstack/               ← Phase 1+ research code goes here
  experiments/             ← one script per paper experiment
local_transformers/
  models/qwen3_vl/
    modeling_qwen3_vl.py   ← THE model source (edit this for research)
    processing_qwen3_vl.py ← processor
models/
  Qwen3-VL-2B-Instruct/   ← downloaded weights (not committed)
paper.md                   ← full research context (read at session start)
log.md                     ← dated changelog
CLAUDE.md                  ← Claude operating instructions
```

---

## Troubleshooting

**`source start.sh` says venv not found**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**ImportError on local_transformers**
The `dependency_versions_table.py` has already been patched to remove stale upper-bound version constraints. If it breaks again after a pip upgrade, check that file for any `<1.0` or similar pinned upper bounds.

**MPS not available**
Run on CPU with `--device cpu --dtype float32`. MPS requires macOS 12.3+ and PyTorch 1.12+. Both should be satisfied on your machine.

**Out of disk space during model download**
Free at least 5 GB first. Run `du -sh ~/` to find large directories.
