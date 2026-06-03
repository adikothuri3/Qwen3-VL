# Benchmark Results

All runs use 1 sample, 128 max_new_tokens, base Qwen3-VL model (no pruning).
These are the **pre-optimization baseline** runs — motivation data for the paper (Figure 2).

| Run | Date | Total (ms) | Generation (ms) | Preprocess (ms) | Tok/s | Notes |
|-----|------|-----------|----------------|-----------------|-------|-------|
| 20251231_103002 | 2025-12-31 | 12,527 | 9,927 | 2,589 | 12.9 | Initial baseline |
| 20251231_103119 | 2025-12-31 | 11,916 | 9,656 | 2,260 | 13.3 | Repeat run |
| 20251231_104050 | 2025-12-31 | 13,117 | 10,478 | 2,636 | 12.2 | Repeat run |
| 20251231_124114 | 2025-12-31 | 12,932 | 10,442 | 2,490 | 12.3 | Repeat run |
| 20251231_130330 | 2025-12-31 | 12,915 | 10,494 | 2,422 | 12.2 | Repeat run |
| 20260127_114046 | 2026-01-27 | 15,778 | 13,336 | 2,440 | 9.6  | Slower — possible background load |
| 20260216_133731 | 2026-02-16 | 14,292 | 11,411 | 2,881 | 11.2 | After code refactor |
| 20260216_134806 | 2026-02-16 | 12,484 | 10,103 | 2,379 | 12.7 | Repeat run |
| 20260216_135055 | 2026-02-16 | 11,416 | 9,103  | 2,313 | 14.1 | **Best baseline — also has profile_data.json and profile_chart.png** |

## Key takeaway for the paper

Generation consistently dominates: **~75–85% of total latency is generation time**.
Preprocessing (vision encoder + input prep) is only ~15–25%.
This confirms that reducing visual tokens matters for the **decoder side**, not just encoding.

The most recent run (`20260216_135055`) is the reference baseline — use its
`profile_data.json` for component-level timing (text decoder: 10,585 ms, vision encoder: 863 ms).
