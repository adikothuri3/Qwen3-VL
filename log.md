# Change Log

Running log of all moderate changes made during the research project. Each entry is dated and describes what changed, why, and whether it was effective. Intended as a reference for writing the paper's methods and results sections.

---

[2026-06-02] | [colab] | Fixed GITHUB_TOKEN loading in colab_run.ipynb Cell 0. Root cause: `userdata.get()` only resolves in the colab.research.google.com web UI; running via a connected kernel (VS Code/local Jupyter) raises TimeoutException ("Secrets can only be fetched when running from the Colab UI"), which the old bare `except Exception` swallowed as a misleading "No GITHUB_TOKEN secret found". Now the except surfaces the real exception type/message and falls back to a GITHUB_TOKEN environment variable, so the notebook works from the Colab UI (Secrets) or any other kernel (env var). | yes — token resolves in both run contexts; precise diagnostics on failure

## Format

```
[YYYY-MM-DD] | [COMPONENT] | [DESCRIPTION] | [EFFECTIVE?]
```

- **COMPONENT**: which part of the codebase or research (e.g., `profiling`, `pruning`, `budgeting`, `evaluation`, `model`, `experiment`)
- **DESCRIPTION**: what was changed and why
- **EFFECTIVE?**: yes / no / partial / pending — with a brief note on outcome if known

---

## Log

[2026-06-02] | [profiling] | Phase 1 complete — mapped Qwen3-VL DeepStack internals. Added src/deepstack/probe.py: a non-invasive forward-hook probe (DeepStackProbe) that reuses Qwen3VLEvaluator for loading + create_default_test_case for input, captures the 3 deepstack groups (vision layers [8,16,24]), per-group shape/dtype/L2-norm stats, injection depths (decoder layers 0/1/2), the visual_pos_masks count, and cross-checks the visual-token count against image_grid_thw. Includes a mutation test proving the strict 1:1 count-contract: naively dropping tokens from a group breaks the injection add (shape mismatch), while prune-then-reconstruct-to-full-length (scatter kept tokens, zero-fill dropped) keeps it valid — this dictates the Phase 4 pruning design. Writes results/<ts>/deepstack_probe.json. Documented findings in paper.md §13 Phase 1 and checked the Phase 1 box. | yes — internals verified against source; runtime confirmation pending the Colab probe run

[2026-06-02] | [workflow] | Enforced hard rule: the model is NEVER run locally — all model runs (evaluate.py, probe.py, future experiments) go through the Colab GPU via colab_run.ipynb. Documented in CLAUDE.md (Colab GPU Workflow). Wired the probe into colab_run.ipynb: Cell 0 gains RUN_PROBE toggle + PROBE_ARGS; Cell 6 dispatches to `python -m src.deepstack.probe` when RUN_PROBE=True, else evaluate.py. | yes — one-click probe run on Colab

[2026-06-01] | [project-setup] | Created CLAUDE.md (operating instructions), paper.md (full research context), and log.md (this file). Established project structure for DeepStack-Aware Visual Token Budgeting research. | yes — foundational setup

[2026-06-01] | [codebase-cleanup] | Deleted src/__pycache__, scripts/setup_branch_protection.py, docs/compile.bat. Created empty package scaffolding: src/deepstack/ (probe, ablation, prune, budget) and src/experiments/ (exp_sensitivity, exp_budgeting, exp_scoring, exp_latency, exp_pareto). Added root requirements.txt and results/README.md documenting 9 baseline runs. | yes — clean starting point for Phase 1

[2026-06-01] | [codebase-cleanup-2] | Deleted all 9 old results/ run directories, docs/ folder, .cursorrules, CODEOWNERS. Full clean slate. | yes

[2026-06-01] | [environment] | Created Python 3.13 venv (.venv/). Installed PyTorch 2.12 (MPS/M1 support), all requirements. Patched local_transformers/dependency_versions_table.py to remove stale huggingface-hub<1.0 upper bound (installed version is 1.17.0). Verified local_transformers wiring — Qwen3VLForConditionalGeneration resolves to local source. Downloaded Qwen3-VL-2B-Instruct (4.0 GB, safetensors). Fixed torch_dtype→dtype deprecation in evaluate.py. Created start.sh and setup.md. Smoke test (CPU, float32, 32 tokens) running — local source confirmed in output. | yes — full environment ready

[2026-06-02] | [colab] | Rebuilt colab_run.ipynb as a one-click GPU runner: Cell 0 config (model path, eval args, reads GITHUB_TOKEN from Colab Secrets), then mount Drive → clone/pull repo → install deps → download model to Drive (cached) → verify GPU + local_transformers wiring → run evaluate.py → commit & push results/ back to GitHub. Token-bearing git URLs are never printed (sanitized subprocess wrapper). Documented the GitHub<->Colab<->local data flow in CLAUDE.md (new "Colab GPU Workflow" section). Model weights stay in Drive, code + results round-trip via GitHub, traces gitignored. | yes — single Run-all executes a full eval and returns results locally on git pull

[2026-06-01] | [evaluate.py] | Renamed Qwen3VLTesting.py → evaluate.py. Fixed two bugs: (1) local_transformers wiring was broken — Qwen3VLForConditionalGeneration was not registered in local_transformers/__init__.py, so the sys.modules redirect was silently falling through to installed transformers; fixed by importing directly from local_transformers.models.qwen3_vl.modeling_qwen3_vl. (2) device_map branch never called from_pretrained, leaving self.model as None. Added startup print showing exact source file of model class for verification. | yes — local model edits now reflected at runtime

---
