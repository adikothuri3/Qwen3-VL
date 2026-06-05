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

