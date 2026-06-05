# DeepStack Phase 5 (Stage A) — Per-Group Budget + Scorer Search

- **Task:** `docvqa` (General VQA / VQAv2).
- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Groups:** 3 — ViT layers [5, 11, 17].
- **Scorers swept:** ['random', 'activation_magnitude', 'hybrid', 'vision_attention']
- **Samples:** 100  •  baseline acc = 0.890

## What this measures
Each DeepStack group is pruned **independently** (the other two held full) from 100% → 0% in 0.05 steps, for every scorer. This is **zeroing** (the refinement at a token is set to 0; the base token still occupies the sequence), so there is no latency change here — methods are compared at an equal count of retained refinement rows. Real sequence-shortening + latency is Stage B.

## Figures
- `budgeting_sensitivity_curves.png` — per-group accuracy as each group is pruned alone. Flat lines = that group tolerates pruning; a steep drop = a fragile group.
- `budgeting_kl_curves.png` — the same in first-token KL (dense, but biased toward magnitude).
- `budgeting_scorer_bars.png` — which scorer best preserves accuracy at an aggressive keep-ratio.

## Best within-group scorer (at keep-ratio 0.25): `hybrid`

## Per-group prunability read
- **Group 0** at keep 0.25 (hybrid): acc 0.900 (+0.010 vs full).
- **Group 1** at keep 0.25 (hybrid): acc 0.920 (+0.030 vs full).
- **Group 2** at keep 0.25 (hybrid): acc 0.908 (+0.018 vs full).

## Joint budget validation (held-out, equal retained-token count)

- **Best scorer used:** `hybrid`  •  held-out n = 300  •  baseline acc = 0.884

| condition | type | budget | acc | 95% CI | retained |
|---|---|---|---|---|---|
| uniform@0.50 | uniform | [0.5, 0.5, 0.5] | 0.887 | [0.852, 0.917] | 1127 |
| pergroup@0.50#0 | pergroup | [0.8, 0.65, 0.05] | 0.867 | [0.829, 0.899] | 1127 |
| pergroup@0.50#1 | pergroup | [0.55, 0.1, 0.85] | 0.885 | [0.850, 0.915] | 1127 |
| pergroup@0.50#2 | pergroup | [0.6, 0.1, 0.8] | 0.885 | [0.850, 0.915] | 1127 |
| global@0.50 | global | None | 0.873 | [0.834, 0.905] | 1127 |
| uniform@0.30 | uniform | [0.3, 0.3, 0.3] | 0.867 | [0.830, 0.899] | 676 |
| pergroup@0.30#0 | pergroup | [0.55, 0.25, 0.1] | 0.859 | [0.821, 0.892] | 676 |
| pergroup@0.30#1 | pergroup | [0.55, 0.3, 0.05] | 0.862 | [0.825, 0.895] | 676 |
| pergroup@0.30#2 | pergroup | [0.6, 0.25, 0.05] | 0.860 | [0.822, 0.892] | 676 |
| global@0.30 | global | None | 0.857 | [0.818, 0.893] | 676 |
| uniform@0.20 | uniform | [0.2, 0.2, 0.2] | 0.858 | [0.820, 0.892] | 451 |
| pergroup@0.20#0 | pergroup | [0.55, 0.0, 0.05] | 0.856 | [0.818, 0.889] | 451 |
| pergroup@0.20#1 | pergroup | [0.0, 0.55, 0.05] | 0.851 | [0.813, 0.886] | 451 |
| pergroup@0.20#2 | pergroup | [0.55, 0.05, 0.0] | 0.850 | [0.810, 0.884] | 451 |
| global@0.20 | global | None | 0.853 | [0.815, 0.887] | 451 |
| uniform@0.15 | uniform | [0.15, 0.15, 0.15] | 0.860 | [0.823, 0.892] | 338 |
| pergroup@0.15#0 | pergroup | [0.25, 0.1, 0.1] | 0.850 | [0.811, 0.884] | 338 |
| pergroup@0.15#1 | pergroup | [0.3, 0.1, 0.05] | 0.848 | [0.809, 0.883] | 338 |
| pergroup@0.15#2 | pergroup | [0.05, 0.3, 0.1] | 0.858 | [0.821, 0.892] | 338 |
| global@0.15 | global | None | 0.858 | [0.821, 0.893] | 338 |

**How to read it:** compare `pergroup@T` vs `uniform@T` vs `global@T` *within each target T* (they share a retained-token count). If the per-group CI overlaps uniform's, per-group budgeting gives no benefit on this task — which, for General VQA, is the expected and honest outcome (Phase 3/4b found it compression-insensitive); the headline then is the large compressibility headroom and the scorer choice, with the per-group win reserved for the OCR/text tasks.

