# DeepStack Phase 3 — Group Sensitivity (Go/No-Go)

- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Groups:** 3 — ViT layers [5, 11, 17], injected at the first decoder layers.
- **Tasks / samples:** {'general_vqa': 100, 'textvqa': 100, 'docvqa': 100, 'counting': 100}
- **Conditions:** ['full', 'drop_g0', 'drop_g1', 'drop_g2', 'keep_g0', 'keep_g1', 'keep_g2', 'drop_all']  (drop_all == No-DeepStack baseline)

## What this experiment answers
Phase 2 showed groups differ in *feature structure* (the dispersion lens). It did **not** show they differ in *accuracy sensitivity*. This experiment removes each group and measures the per-task accuracy drop. **Different drops across groups = per-depth budgeting is justified** (paper.md §4, §16).

## Figures
- `sensitivity_heatmap.png` — rows = tasks, columns = ablations, color = accuracy drop vs full. A redder cell means that group is more load-bearing for that task. Read each row: if `drop_g0`/`drop_g1`/`drop_g2` differ within a row, that task relies on the groups unequally.
- `kl_by_condition.png` — a dense, label-free cross-check: how much each ablation shifts the first-token distribution away from full DeepStack.

## Verdict from this run
**GO signal.** Within-task spread across single-group drops reaches 0.053 accuracy — groups are *not* interchangeable. Most-load-bearing group per task: general_vqa→drop_g0, textvqa→drop_g2, docvqa→drop_g2, counting→drop_g0 — and it differs across tasks (task-dependent sensitivity).

Per-task accuracy under each condition:

| task | full | drop_g0 | drop_g1 | drop_g2 | keep_g0 | keep_g1 | keep_g2 | drop_all |
|---|---|---|---|---|---|---|---|---|
| general_vqa | 0.833 | 0.820 | 0.840 | 0.833 | 0.830 | 0.833 | 0.800 | 0.837 |
| textvqa | 0.840 | 0.853 | 0.843 | 0.800 | 0.800 | 0.817 | 0.823 | 0.807 |
| docvqa | 0.890 | 0.900 | 0.900 | 0.898 | 0.866 | 0.900 | 0.898 | 0.837 |
| counting | 0.820 | 0.820 | 0.830 | 0.820 | 0.840 | 0.830 | 0.830 | 0.780 |
