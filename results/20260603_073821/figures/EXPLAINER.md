# DeepStack Phase 2 — What the Figures Mean

- **Model:** `/content/Qwen3-VL/local_transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- **Samples:** 4  (sources: ['url', 'url', 'synthetic', 'synthetic'])
- **Groups:** 3 — visual features tapped at ViT layers [5, 11, 17] and injected into decoder layers [0, 1, 2].
- **Visual tokens per image:** [672, 300, 256, 256]

Every group carries the **same number** of tokens for a given image — DeepStack injects the same token positions at each depth. So the question is not *how many* tokens a group has, but *how valuable its tokens are*. The figures below answer that.

## 1. Token-strength distribution (`01_norm_distribution.png`)
Each visual token is a 2048-number vector; its **strength** is the vector's length (L2 norm) — loosely, how much signal it injects. The curves show how strength is spread within each group. A curve pushed to the **right** = stronger tokens. A tall spike on the **left** = lots of weak, likely-redundant tokens.

## 2. Average strength per group (`02_norm_summary.png`)
The same thing, summarized: mean ± spread, with median and the p10/p90 range. If strength **rises with depth**, deeper groups pack more information per token.

## 3. Prunability (`03_prunability.png`)
The percentage of each group's tokens that fall **below a low-strength cutoff**. Higher bar = more of that group is low-value = safer to prune. This is the most direct hint at where a token budget can be cut with least damage.

## 4. DeepStack overhead (`04_latency.png`)
Time to build (extraction) and add (injection) each group's tokens. These are tiny, which matters: it means real speedups come from **reducing the token count the decoder must process**, not from touching the DeepStack machinery itself.

## Takeaway from this run
- Weakest group on average: **Group 0** (ViT L5, mean strength 14.7) → most prunable.
- Strongest group on average: **Group 2** (ViT L17, mean strength 25.3) → most fragile, keep more of it.

This non-uniformity across groups is exactly the signal the per-group budgeting method needs (paper.md §4 hypothesis).
