# DeepStack Phase 4 — Within-Group Scoring Comparison

- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Groups:** 3 — ViT layers [5, 11, 17].
- **Budget mode:** `uniform_all_groups` — the *same* keep-ratio is applied to every DeepStack group, so this isolates scorer quality from budget allocation (Phase 5).
- **Scorers:** ['random', 'activation_magnitude', 'hybrid', 'vision_attention']
- **Keep-ratios:** [1.0, 0.5, 0.25]  (1.0 = full / no pruning baseline)
- **Tasks / samples:** {'general_vqa': 300, 'textvqa': 300, 'docvqa': 300, 'counting': 300}

## What this experiment answers
Given a fixed retained-token budget, *which tokens should we keep?* Every scorer keeps the exact same number of tokens per group, so any accuracy difference is purely about token *selection*. `random` is the control — a useful scorer must beat it. `activation_magnitude`, `diversity`, and `hybrid` are vision-side feature signals; `vision_attention` is the literature's strong vision-encoder attention-received signal (VisPruner / FasterVLM, CLS-free recipe). Decoder attention is deliberately excluded — Phase 2 showed it is a null.

**Rank by accuracy, not KL.** The first-token KL reported alongside is a secondary signal and is *structurally biased* toward `activation_magnitude` (KL rewards a small output shift, and magnitude keeps the largest additive vectors, minimizing that shift almost by construction). At this sample size the verdict is read off accuracy; KL is a cross-check only.

## Figures
- `scoring_accuracy_curves.png` — per task, accuracy as the keep-ratio drops from 1.0. A scorer whose line stays high as we move right (more pruning) preserves the right tokens.
- `scoring_bar_at_50pct.png` — a single-glance comparison at a mid-compression keep-ratio.

## Read at keep-ratio 0.50

- **general_vqa**: best = `activation_magnitude` (0.831; +0.009 vs random; +0.013 vs full).
- **textvqa**: best = `vision_attention` (0.822; +0.018 vs random; -0.002 vs full).
- **docvqa**: best = `hybrid` (0.892; +0.009 vs random; -0.000 vs full).
- **counting**: best = `hybrid` (0.833; +0.023 vs random; +0.020 vs full).

The best scorer is task-dependent: general_vqa→`activation_magnitude`, textvqa→`vision_attention`, docvqa→`hybrid`, counting→`hybrid`. Consider this when fixing the within-group scorer for Phase 5.

## Full accuracy table

| task | full | random@0.25 | activation_magnitude@0.25 | hybrid@0.25 | vision_attention@0.25 | random@0.50 | activation_magnitude@0.50 | hybrid@0.50 | vision_attention@0.50 |
|---|---|---|---|---|---|---|---|---|---|
| general_vqa | 0.818 | 0.822 | 0.824 | 0.820 | 0.820 | 0.822 | 0.831 | 0.826 | 0.822 |
| textvqa | 0.824 | 0.808 | 0.814 | 0.812 | 0.793 | 0.804 | 0.812 | 0.816 | 0.822 |
| docvqa | 0.892 | 0.832 | 0.871 | 0.872 | 0.861 | 0.883 | 0.885 | 0.892 | 0.874 |
| counting | 0.813 | 0.820 | 0.840 | 0.823 | 0.823 | 0.810 | 0.830 | 0.833 | 0.823 |

