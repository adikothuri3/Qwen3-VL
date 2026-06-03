# Change Log

Running log of all moderate changes made during the research project. Each entry is dated and describes what changed, why, and whether it was effective. Intended as a reference for writing the paper's methods and results sections.

---

## Format

```
[YYYY-MM-DD] | [COMPONENT] | [DESCRIPTION] | [EFFECTIVE?]
```

- **COMPONENT**: which part of the codebase or research (e.g., `profiling`, `pruning`, `budgeting`, `evaluation`, `model`, `experiment`)
- **DESCRIPTION**: what was changed and why
- **EFFECTIVE?**: yes / no / partial / pending — with a brief note on outcome if known

---

## Log

[2026-06-01] | [project-setup] | Created CLAUDE.md (operating instructions), paper.md (full research context), and log.md (this file). Established project structure for DeepStack-Aware Visual Token Budgeting research. | yes — foundational setup

[2026-06-01] | [codebase-cleanup] | Deleted src/__pycache__, scripts/setup_branch_protection.py, docs/compile.bat. Created empty package scaffolding: src/deepstack/ (probe, ablation, prune, budget) and src/experiments/ (exp_sensitivity, exp_budgeting, exp_scoring, exp_latency, exp_pareto). Added root requirements.txt and results/README.md documenting 9 baseline runs. | yes — clean starting point for Phase 1

[2026-06-01] | [codebase-cleanup-2] | Deleted all 9 old results/ run directories, docs/ folder, .cursorrules, CODEOWNERS. Full clean slate. | yes

[2026-06-01] | [environment] | Created Python 3.13 venv (.venv/). Installed PyTorch 2.12 (MPS/M1 support), all requirements. Patched local_transformers/dependency_versions_table.py to remove stale huggingface-hub<1.0 upper bound (installed version is 1.17.0). Verified local_transformers wiring — Qwen3VLForConditionalGeneration resolves to local source. Downloaded Qwen3-VL-2B-Instruct (4.0 GB, safetensors). Fixed torch_dtype→dtype deprecation in evaluate.py. Created start.sh and setup.md. Smoke test (CPU, float32, 32 tokens) running — local source confirmed in output. | yes — full environment ready

[2026-06-02] | [colab] | Rebuilt colab_run.ipynb as a one-click GPU runner: Cell 0 config (model path, eval args, reads GITHUB_TOKEN from Colab Secrets), then mount Drive → clone/pull repo → install deps → download model to Drive (cached) → verify GPU + local_transformers wiring → run evaluate.py → commit & push results/ back to GitHub. Token-bearing git URLs are never printed (sanitized subprocess wrapper). Documented the GitHub<->Colab<->local data flow in CLAUDE.md (new "Colab GPU Workflow" section). Model weights stay in Drive, code + results round-trip via GitHub, traces gitignored. | yes — single Run-all executes a full eval and returns results locally on git pull

[2026-06-01] | [evaluate.py] | Renamed Qwen3VLTesting.py → evaluate.py. Fixed two bugs: (1) local_transformers wiring was broken — Qwen3VLForConditionalGeneration was not registered in local_transformers/__init__.py, so the sys.modules redirect was silently falling through to installed transformers; fixed by importing directly from local_transformers.models.qwen3_vl.modeling_qwen3_vl. (2) device_map branch never called from_pretrained, leaving self.model as None. Added startup print showing exact source file of model class for verification. | yes — local model edits now reflected at runtime

---
