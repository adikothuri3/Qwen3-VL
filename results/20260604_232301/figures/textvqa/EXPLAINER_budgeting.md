# DeepStack Phase 5 (Stage A) — Per-Group Budget + Scorer Search

- **Task:** `textvqa` (General VQA / VQAv2).
- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Groups:** 3 — ViT layers [5, 11, 17].
- **Scorers swept:** ['random', 'activation_magnitude', 'hybrid', 'vision_attention']
- **Samples:** 100  •  baseline acc = 0.840

## What this measures
Each DeepStack group is pruned **independently** (the other two held full) from 100% → 0% in 0.05 steps, for every scorer. This is **zeroing** (the refinement at a token is set to 0; the base token still occupies the sequence), so there is no latency change here — methods are compared at an equal count of retained refinement rows. Real sequence-shortening + latency is Stage B.

## Figures
- `budgeting_sensitivity_curves.png` — per-group accuracy as each group is pruned alone. Flat lines = that group tolerates pruning; a steep drop = a fragile group.
- `budgeting_kl_curves.png` — the same in first-token KL (dense, but biased toward magnitude).
- `budgeting_scorer_bars.png` — which scorer best preserves accuracy at an aggressive keep-ratio.

## Best within-group scorer (at keep-ratio 0.25): `vision_attention`

## Per-group prunability read
- **Group 0** at keep 0.25 (vision_attention): acc 0.843 (+0.003 vs full).
- **Group 1** at keep 0.25 (vision_attention): acc 0.843 (+0.003 vs full).
- **Group 2** at keep 0.25 (vision_attention): acc 0.827 (-0.013 vs full).

## Joint budget validation (held-out, equal retained-token count)

- **Best scorer used:** `hybrid`  •  held-out n = 300  •  baseline acc = 0.822

| condition | type | budget | acc | 95% CI | retained |
|---|---|---|---|---|---|
| uniform@0.50 | uniform | [0.5, 0.5, 0.5] | 0.820 | [0.779, 0.859] | 1104 |
| pergroup@0.50#0 | pergroup | [0.0, 0.8, 0.7] | 0.801 | [0.760, 0.842] | 1104 |
| pergroup@0.50#1 | pergroup | [0.0, 0.85, 0.65] | 0.791 | [0.748, 0.833] | 1104 |
| pergroup@0.50#2 | pergroup | [0.0, 0.9, 0.6] | 0.800 | [0.757, 0.842] | 1104 |
| global@0.50 | global | None | 0.799 | [0.754, 0.841] | 1104 |
| uniform@0.30 | uniform | [0.3, 0.3, 0.3] | 0.819 | [0.778, 0.860] | 662 |
| pergroup@0.30#0 | pergroup | [0.0, 0.0, 0.9] | 0.791 | [0.750, 0.831] | 662 |
| pergroup@0.30#1 | pergroup | [0.0, 0.05, 0.85] | 0.801 | [0.760, 0.841] | 662 |
| pergroup@0.30#2 | pergroup | [0.05, 0.0, 0.85] | 0.797 | [0.753, 0.838] | 662 |
| global@0.30 | global | None | 0.801 | [0.760, 0.842] | 662 |
| uniform@0.20 | uniform | [0.2, 0.2, 0.2] | 0.802 | [0.760, 0.847] | 442 |
| pergroup@0.20#0 | pergroup | [0.0, 0.0, 0.6] | 0.789 | [0.747, 0.830] | 442 |
| pergroup@0.20#1 | pergroup | [0.1, 0.0, 0.5] | 0.806 | [0.763, 0.848] | 442 |
| pergroup@0.20#2 | pergroup | [0.1, 0.05, 0.45] | 0.797 | [0.753, 0.839] | 442 |
| global@0.20 | global | None | 0.806 | [0.761, 0.846] | 442 |
| uniform@0.15 | uniform | [0.15, 0.15, 0.15] | 0.793 | [0.751, 0.836] | 331 |
| pergroup@0.15#0 | pergroup | [0.0, 0.0, 0.45] | 0.788 | [0.746, 0.829] | 331 |
| pergroup@0.15#1 | pergroup | [0.1, 0.0, 0.35] | 0.801 | [0.759, 0.843] | 331 |
| pergroup@0.15#2 | pergroup | [0.1, 0.2, 0.15] | 0.791 | [0.748, 0.833] | 331 |
| global@0.15 | global | None | 0.792 | [0.749, 0.833] | 331 |

**How to read it:** compare `pergroup@T` vs `uniform@T` vs `global@T` *within each target T* (they share a retained-token count). If the per-group CI overlaps uniform's, per-group budgeting gives no benefit on this task — which, for General VQA, is the expected and honest outcome (Phase 3/4b found it compression-insensitive); the headline then is the large compressibility headroom and the scorer choice, with the per-group win reserved for the OCR/text tasks.

