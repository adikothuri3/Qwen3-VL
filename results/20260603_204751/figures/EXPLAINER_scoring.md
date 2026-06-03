# DeepStack Phase 4 — Within-Group Scoring Comparison

- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Groups:** 3 — ViT layers [5, 11, 17].
- **Budget mode:** `uniform_all_groups` — the *same* keep-ratio is applied to every DeepStack group, so this isolates scorer quality from budget allocation (Phase 5).
- **Scorers:** ['random', 'spatial_uniform', 'activation_magnitude', 'diversity', 'hybrid']
- **Keep-ratios:** [1.0, 0.75, 0.5, 0.25]  (1.0 = full / no pruning baseline)
- **Tasks / samples:** {'general_vqa': 100, 'textvqa': 100, 'docvqa': 100, 'counting': 100}

## What this experiment answers
Given a fixed retained-token budget, *which tokens should we keep?* Every scorer keeps the exact same number of tokens per group, so any accuracy difference is purely about token *selection*. `random` is the control — a useful scorer must beat it; `activation_magnitude`, `diversity`, and `hybrid` are the vision-side signals motivated by Phase 2 (decoder attention was an informative null there).

## Figures
- `scoring_accuracy_curves.png` — per task, accuracy as the keep-ratio drops from 1.0. A scorer whose line stays high as we move right (more pruning) preserves the right tokens.
- `scoring_bar_at_50pct.png` — a single-glance comparison at a mid-compression keep-ratio.

## Read at keep-ratio 0.50

- **general_vqa**: best = `random` (0.843; +0.000 vs random; +0.010 vs full).
- **textvqa**: best = `spatial_uniform` (0.827; +0.013 vs random; -0.013 vs full).
- **docvqa**: best = `activation_magnitude` (0.901; +0.003 vs random; +0.011 vs full).
- **counting**: best = `diversity` (0.850; +0.030 vs random; +0.030 vs full).

The best scorer is task-dependent: general_vqa→`random`, textvqa→`spatial_uniform`, docvqa→`activation_magnitude`, counting→`diversity`. Consider this when fixing the within-group scorer for Phase 5.

## Full accuracy table

| task | full | random@0.25 | spatial_uniform@0.25 | activation_magnitude@0.25 | diversity@0.25 | hybrid@0.25 | random@0.50 | spatial_uniform@0.50 | activation_magnitude@0.50 | diversity@0.50 | hybrid@0.50 | random@0.75 | spatial_uniform@0.75 | activation_magnitude@0.75 | diversity@0.75 | hybrid@0.75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| general_vqa | 0.833 | 0.840 | 0.837 | 0.823 | 0.840 | 0.823 | 0.843 | 0.827 | 0.833 | 0.823 | 0.840 | 0.813 | 0.830 | 0.807 | 0.817 | 0.833 |
| textvqa | 0.840 | 0.833 | 0.820 | 0.833 | 0.813 | 0.830 | 0.813 | 0.827 | 0.820 | 0.813 | 0.810 | 0.823 | 0.833 | 0.840 | 0.843 | 0.823 |
| docvqa | 0.890 | 0.850 | 0.865 | 0.897 | 0.862 | 0.887 | 0.898 | 0.879 | 0.901 | 0.889 | 0.900 | 0.901 | 0.898 | 0.892 | 0.910 | 0.890 |
| counting | 0.820 | 0.840 | 0.810 | 0.850 | 0.830 | 0.830 | 0.820 | 0.830 | 0.820 | 0.850 | 0.840 | 0.800 | 0.830 | 0.830 | 0.840 | 0.820 |

