# DeepStack Phase 5 (Stage A) — Per-Group Budget + Scorer Search

- **Task:** `general_vqa` (General VQA / VQAv2).
- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Groups:** 3 — ViT layers [5, 11, 17].
- **Scorers swept:** ['random', 'activation_magnitude', 'hybrid', 'vision_attention']
- **Samples:** 100  •  baseline acc = 0.833

## What this measures
Each DeepStack group is pruned **independently** (the other two held full) from 100% → 0% in 0.05 steps, for every scorer. This is **zeroing** (the refinement at a token is set to 0; the base token still occupies the sequence), so there is no latency change here — methods are compared at an equal count of retained refinement rows. Real sequence-shortening + latency is Stage B.

## Figures
- `budgeting_sensitivity_curves.png` — per-group accuracy as each group is pruned alone. Flat lines = that group tolerates pruning; a steep drop = a fragile group.
- `budgeting_kl_curves.png` — the same in first-token KL (dense, but biased toward magnitude).
- `budgeting_scorer_bars.png` — which scorer best preserves accuracy at an aggressive keep-ratio.

## Best within-group scorer (at keep-ratio 0.25): `random`

## Per-group prunability read
- **Group 0** at keep 0.25 (random): acc 0.813 (-0.020 vs full).
- **Group 1** at keep 0.25 (random): acc 0.840 (+0.007 vs full).
- **Group 2** at keep 0.25 (random): acc 0.833 (+0.000 vs full).

## Joint budget validation (held-out, equal retained-token count)

- **Best scorer used:** `hybrid`  •  held-out n = 300  •  baseline acc = 0.818

| condition | type | budget | acc | 95% CI | retained |
|---|---|---|---|---|---|
| uniform@0.50 | uniform | [0.5, 0.5, 0.5] | 0.821 | [0.781, 0.861] | 412 |
| pergroup@0.50#0 | pergroup | [1.0, 0.3, 0.2] | 0.814 | [0.774, 0.852] | 413 |
| pergroup@0.50#1 | pergroup | [1.0, 0.35, 0.15] | 0.811 | [0.771, 0.850] | 412 |
| pergroup@0.50#2 | pergroup | [0.85, 0.5, 0.15] | 0.808 | [0.769, 0.847] | 413 |
| global@0.50 | global | None | 0.826 | [0.788, 0.862] | 413 |
| uniform@0.30 | uniform | [0.3, 0.3, 0.3] | 0.822 | [0.783, 0.859] | 248 |
| pergroup@0.30#0 | pergroup | [0.85, 0.0, 0.05] | 0.813 | [0.773, 0.850] | 248 |
| pergroup@0.30#1 | pergroup | [0.85, 0.05, 0.0] | 0.809 | [0.769, 0.846] | 248 |
| pergroup@0.30#2 | pergroup | [0.9, 0.0, 0.0] | 0.813 | [0.773, 0.850] | 248 |
| global@0.30 | global | None | 0.822 | [0.784, 0.860] | 248 |
| uniform@0.20 | uniform | [0.2, 0.2, 0.2] | 0.821 | [0.782, 0.858] | 165 |
| pergroup@0.20#0 | pergroup | [0.0, 0.4, 0.2] | 0.819 | [0.780, 0.857] | 165 |
| pergroup@0.20#1 | pergroup | [0.0, 0.45, 0.15] | 0.816 | [0.776, 0.854] | 165 |
| pergroup@0.20#2 | pergroup | [0.0, 0.5, 0.1] | 0.818 | [0.778, 0.856] | 165 |
| global@0.20 | global | None | 0.819 | [0.780, 0.856] | 165 |
| uniform@0.15 | uniform | [0.15, 0.15, 0.15] | 0.816 | [0.776, 0.853] | 124 |
| pergroup@0.15#0 | pergroup | [0.0, 0.35, 0.1] | 0.818 | [0.778, 0.856] | 124 |
| pergroup@0.15#1 | pergroup | [0.0, 0.25, 0.2] | 0.819 | [0.780, 0.857] | 124 |
| pergroup@0.15#2 | pergroup | [0.0, 0.45, 0.0] | 0.811 | [0.771, 0.850] | 124 |
| global@0.15 | global | None | 0.820 | [0.781, 0.856] | 124 |

**How to read it:** compare `pergroup@T` vs `uniform@T` vs `global@T` *within each target T* (they share a retained-token count). If the per-group CI overlaps uniform's, per-group budgeting gives no benefit on this task — which, for General VQA, is the expected and honest outcome (Phase 3/4b found it compression-insensitive); the headline then is the large compressibility headroom and the scorer choice, with the per-group win reserved for the OCR/text tasks.

