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

[2026-06-03] | [profiling] | Phase 2 started — added src/deepstack/instrument.py: a non-invasive DeepStackInstrumentor (context manager) that attaches forward hooks to measure, across a small calibration set, the five quantities Phase 2 calls for: (1) per-group token count, (2) per-group feature-norm distribution (stats + percentiles + 30-bin histogram, pooled over tokens×samples), (3) opt-in per-group visual-token attention saliency (--capture-attention, forces eager attention, hooks only Qwen3VLTextAttention injection layers via module.layer_idx), (4) extraction latency (pre/post hooks on each deepstack PatchMerger) and injection latency (bracketing decoder layers 0..N — injection i runs between layer i return and layer i+1 call), (5) per-group GPU memory deltas + peak. Group count / vision layers / out_hidden_size are read from the loaded model config at runtime, so it self-corrects to the actual [5,11,17]/2048 found in Phase 1 rather than the class defaults. Reuses Qwen3VLEvaluator/create_default_test_case and the probe's hook/context-manager pattern; runs one prefill forward per sample over a built-in CALIBRATION_IMAGES list (natural/OCR/chart/counting). Writes results/<ts>/deepstack_instrument.json. CLI mirrors probe.py (+ --num-samples, --capture-attention). Wired into colab_run.ipynb (RUN_INSTRUMENT toggle + INSTRUMENT_ARGS/INSTRUMENT_CAPTURE_ATTENTION in Cell 0, dispatch branch in Cell 6). Local checks pass (py_compile, ruff, mypy, notebook JSON). | pending — awaiting Colab GPU run to populate the distributions

[2026-06-03] | [profiling] | Phase 1 probe RAN on Colab T4 (results/20260603_050848/deepstack_probe.json). Key runtime findings on Qwen3-VL-2B-Instruct: (1) actual config taps vision layers [5,11,17] — NOT the class default [8,16,24]; always read deepstack_visual_indexes from the loaded model. (2) out_hidden_size=2048 = text hidden dim (groups pre-projected into LLM space). (3) demo image grid [1,86,128] → 2752 tokens/group, 3 groups; count_match PASS all, grid cross-check MATCH. (4) mutation test confirmed the 1:1 count-contract: naive 25% drop raised ValueError (2752 vs 2064), reconstruct-to-full PASSED. (5) Per-group token L2-norm grows with depth (μ 15.1→17.9→23.2); group 0 (layer 5) is heavy-tailed (max 141.7 ≈ 9× mean) → activation-magnitude scoring needs a diversity term there. Corrected layer/dim numbers in paper.md §13 Phase 1 and added a Phase 1 Runtime Results subsection with implications for Phases 2/4/5. | yes — Phase 1 fully closed, runtime-confirmed

[2026-06-02] | [profiling] | Phase 1 complete — mapped Qwen3-VL DeepStack internals. Added src/deepstack/probe.py: a non-invasive forward-hook probe (DeepStackProbe) that reuses Qwen3VLEvaluator for loading + create_default_test_case for input, captures the 3 deepstack groups (vision layers [8,16,24]), per-group shape/dtype/L2-norm stats, injection depths (decoder layers 0/1/2), the visual_pos_masks count, and cross-checks the visual-token count against image_grid_thw. Includes a mutation test proving the strict 1:1 count-contract: naively dropping tokens from a group breaks the injection add (shape mismatch), while prune-then-reconstruct-to-full-length (scatter kept tokens, zero-fill dropped) keeps it valid — this dictates the Phase 4 pruning design. Writes results/<ts>/deepstack_probe.json. Documented findings in paper.md §13 Phase 1 and checked the Phase 1 box. | yes — internals verified against source; runtime confirmation pending the Colab probe run

[2026-06-02] | [workflow] | Enforced hard rule: the model is NEVER run locally — all model runs (evaluate.py, probe.py, future experiments) go through the Colab GPU via colab_run.ipynb. Documented in CLAUDE.md (Colab GPU Workflow). Wired the probe into colab_run.ipynb: Cell 0 gains RUN_PROBE toggle + PROBE_ARGS; Cell 6 dispatches to `python -m src.deepstack.probe` when RUN_PROBE=True, else evaluate.py. | yes — one-click probe run on Colab

[2026-06-01] | [project-setup] | Created CLAUDE.md (operating instructions), paper.md (full research context), and log.md (this file). Established project structure for DeepStack-Aware Visual Token Budgeting research. | yes — foundational setup

[2026-06-01] | [codebase-cleanup] | Deleted src/__pycache__, scripts/setup_branch_protection.py, docs/compile.bat. Created empty package scaffolding: src/deepstack/ (probe, ablation, prune, budget) and src/experiments/ (exp_sensitivity, exp_budgeting, exp_scoring, exp_latency, exp_pareto). Added root requirements.txt and results/README.md documenting 9 baseline runs. | yes — clean starting point for Phase 1

[2026-06-01] | [codebase-cleanup-2] | Deleted all 9 old results/ run directories, docs/ folder, .cursorrules, CODEOWNERS. Full clean slate. | yes

[2026-06-01] | [environment] | Created Python 3.13 venv (.venv/). Installed PyTorch 2.12 (MPS/M1 support), all requirements. Patched local_transformers/dependency_versions_table.py to remove stale huggingface-hub<1.0 upper bound (installed version is 1.17.0). Verified local_transformers wiring — Qwen3VLForConditionalGeneration resolves to local source. Downloaded Qwen3-VL-2B-Instruct (4.0 GB, safetensors). Fixed torch_dtype→dtype deprecation in evaluate.py. Created start.sh and setup.md. Smoke test (CPU, float32, 32 tokens) running — local source confirmed in output. | yes — full environment ready

[2026-06-02] | [colab] | Rebuilt colab_run.ipynb as a one-click GPU runner: Cell 0 config (model path, eval args, reads GITHUB_TOKEN from Colab Secrets), then mount Drive → clone/pull repo → install deps → download model to Drive (cached) → verify GPU + local_transformers wiring → run evaluate.py → commit & push results/ back to GitHub. Token-bearing git URLs are never printed (sanitized subprocess wrapper). Documented the GitHub<->Colab<->local data flow in CLAUDE.md (new "Colab GPU Workflow" section). Model weights stay in Drive, code + results round-trip via GitHub, traces gitignored. | yes — single Run-all executes a full eval and returns results locally on git pull

[2026-06-01] | [evaluate.py] | Renamed Qwen3VLTesting.py → evaluate.py. Fixed two bugs: (1) local_transformers wiring was broken — Qwen3VLForConditionalGeneration was not registered in local_transformers/__init__.py, so the sys.modules redirect was silently falling through to installed transformers; fixed by importing directly from local_transformers.models.qwen3_vl.modeling_qwen3_vl. (2) device_map branch never called from_pretrained, leaving self.model as None. Added startup print showing exact source file of model class for verification. | yes — local model edits now reflected at runtime

---
