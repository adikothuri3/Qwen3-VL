# Qwen3-VL Research Project — Claude Operating Instructions

## Startup Protocol

**On every session start, before doing anything else:**
Read `paper.md` in full. This is the source of truth for the research direction, methodology, and current state of the project. Do not assume context from memory alone — always verify against paper.md.

---

## Project Context

This is a research project studying **DeepStack-Aware Visual Token Budgeting** for efficient multimodal inference with Qwen3-VL. The core idea: DeepStack injects visual features at different LLM depths — each depth-group should get its own token budget based on its compression sensitivity, rather than uniform or global pruning.

The goal is a publishable research paper showing that per-group DeepStack budgeting outperforms uniform pruning at the same retained-token count, with real latency/memory improvements.

Full research context, methodology, and experiments are in `paper.md`.

---

## Working Protocol

1. **Always plan first.** Before writing any code or making any file edits, produce a clear written plan: what you're going to change, which files, why, and what the expected outcome is.
2. **Present the plan for review.** The user must explicitly approve the plan before any implementation begins.
3. **Only then implement.** Once approved, implement exactly what was agreed. If scope changes mid-implementation, stop and re-plan.
4. **Log the change.** After any moderate change (new feature, new experiment, structural refactor, result recorded), add an entry to `log.md`.

---

## Logging Rule

Every moderate change gets an entry in `log.md`. Format:

```
[YYYY-MM-DD] | [COMPONENT] | [DESCRIPTION] | [EFFECTIVE?]
```

"Moderate change" includes: new experiment code, new profiling hooks, new pruning method, new baseline, benchmark run completed, architecture modification, file restructure. It does not include typo fixes or minor formatting.

The user will use these log entries as references when writing the paper.

---

## Colab GPU Workflow

Heavy runs (model inference, profiling, benchmarks) execute on a Colab GPU via `colab_run.ipynb`, not locally. Local machine has no CUDA GPU. The data flow:

```
local repo --git push--> GitHub --git pull--> Colab   (code edits reach the GPU)
Colab runs evaluate.py -> results/<timestamp>/
Colab --git push--> GitHub --git pull--> local repo   (results come back)
```

- **Code** (edits to `local_transformers/`, new experiments) round-trips through GitHub. To test a change on GPU: commit + push locally, then in Colab run the notebook (Cell 2 pulls the latest).
- **Model weights** (~4.5 GB) live in Google Drive only (`/content/drive/MyDrive/Qwen3-VL-models/`), downloaded once and reused. Never committed.
- **Results** are committed + pushed by the notebook's last cell; pull locally to retrieve them.
- **Auth**: the GitHub fine-grained PAT is read as `GITHUB_TOKEN`, resolved in this order: (1) Colab Secrets (🔑 icon) — works **only** when run from the colab.research.google.com web UI; (2) a `GITHUB_TOKEN` environment variable — the fallback for any other kernel (VS Code / local Jupyter), where Colab Secrets time out. The token is never written into the notebook (the notebook is committed to GitHub).
- **Trace files** (`trace_*.json`) are gitignored — they can be hundreds of MB and stay on Colab/Drive only.

To run an evaluation, use the **Colab web UI** (recommended): open `colab_run.ipynb` on colab.research.google.com → Runtime → Change runtime type → GPU (T4) → **Run all**. The 🔑-panel `GITHUB_TOKEN` secret (with *Notebook access* toggled on) resolves natively there. Only Cell 0 (config: model path, eval args) is meant to be edited. Running from a VS Code/local kernel also works but requires `export GITHUB_TOKEN=…` before launching the kernel, since Colab Secrets are unreachable outside the web UI.

---

## Key File Map

| File/Folder | Purpose |
|---|---|
| `paper.md` | Full research context — read on startup |
| `log.md` | Dated changelog — write after every moderate change |
| `colab_run.ipynb` | One-click Colab GPU runner (clone → install → eval → push results) |
| `src/evaluate.py` | Core profiling and evaluation script |
| `local_transformers/models/qwen3_vl/modeling_qwen3_vl.py` | Qwen3-VL model source (editable locally) |
| `local_transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py` | MOE variant model source |
| `qwen-vl-finetune/qwenvl/train/train_qwen.py` | Fine-tuning entry point |
| `evaluation/mmmu/run_mmmu.py` | MMMU benchmark evaluation |
| `results/` | Timestamped benchmark outputs |
| `docs/` | Project timeline and setup guides |

---

## Code Conventions

- **Linting**: ruff (configured in `pyproject.toml`), line length 120
- **Type checking**: mypy with strict settings; all new code must have type hints
- **Comments**: only when the WHY is non-obvious — no narration of what the code does
- **New scripts**: place in `src/` for evaluation/profiling, `qwen-vl-finetune/` for training
- **Results**: always write to a timestamped subdirectory under `results/`
- **No global side effects**: model loading, CUDA setup, and file I/O must be explicit and controllable via arguments
