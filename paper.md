# DeepStack-Aware Visual Token Budgeting for Efficient Multimodal Inference

https://colab.research.google.com/github/adikothuri3/Qwen3-VL/blob/main/colab_run.ipynb


*Living research document — updated as the project evolves.*

---

## 1. Core Idea

Modern multimodal models are slow partly because images/videos produce many visual tokens. These tokens increase memory, prefill cost, and decoder-side workload. Profiling of Qwen3-VL shows that generation dominates total inference time: the text decoder takes 10,585.5 ms out of 12,084.8 ms total generation time, while the vision encoder took only 863.7 ms. Visual tokens matter not just because encoding is expensive, but because they inflate the decoder's workload.

**DeepStack** changes the usual VLM setup. Instead of feeding all visual tokens into the first LLM layer, DeepStack stacks visual tokens into groups and feeds each group into aligned transformer layers from bottom to top. The original DeepStack paper reports strong performance even with only one-fifth of the context length compared with full-context baselines.

**Research question:**
> Can we reduce the visual-token mass of DeepStack models by assigning different token budgets to different DeepStack visual feature groups, while preserving multimodal accuracy?

In simpler terms: instead of keeping the same amount of visual information at every DeepStack depth, keep only the useful visual tokens for each DeepStack group.

---

## 2. Why This Matters

Multimodal inference is becoming token-heavy. Qwen3-VL supports long interleaved text, image, and video contexts up to 256K tokens. A single image can generate hundreds or thousands of visual tokens, causing high prefill cost and decoding memory overhead (CoViPAL). Visual-token compression is already a major research area because of this.

This work sits in the intersection of two real problems:
1. Visual tokens are a dominant cost in multimodal inference.
2. DeepStack's depth-structured injection has not been specifically targeted for compression.

---

## 3. Novelty

The broad area of visual-token pruning is crowded. What is already explored:

- Visual-token pruning (general)
- Attention-guided pruning
- Layer-wise visual-token pruning
- Intermediate ViT-layer compression (LaCo: >15% throughput improvement)
- KV-cache compression for VLMs (VL-Cache: retains 10% KV cache, up to 2.33× latency speedup)
- Pruning + quantization combined
- Progressive pruning during decoding (ST3: ~2× faster inference, ~30% KV-cache memory vs LLaVA)

**This paper's novelty:** *DeepStack-aware source-to-injection token budgeting.*

Existing methods treat visual tokens as a flat sequence, compress intermediate ViT layers generically, or compress decoder KV-cache. This work targets the specific visual feature streams that DeepStack extracts from ViT layers and injects into different LLM depths. Each DeepStack group gets its own token budget based on its role, sensitivity, and usefulness.

**Contribution statement:**
> We study whether DeepStack's depth-aligned visual feature groups have non-uniform compression sensitivity and propose a DeepStack-aware token budgeting method that assigns separate pruning budgets to each visual feature group before injection.

This is distinct from generic ViT-layer pruning: the question is not only "which ViT tokens are useful?" but "which tokens are useful for the specific DeepStack group that will be injected at a specific LLM depth?"

---

## 4. Hypothesis

> Different DeepStack groups have different compression tolerance depending on depth, task, and visual complexity.

**Operationalizing the hypothesis — the dispersion lens (central framing, added 2026-06-03).**
We make the hypothesis concrete and measurable through **within-group feature dispersion**: how
*unequal* a group's visual tokens are relative to that group's own scale. A high-dispersion group
holds many near-redundant tokens plus a few dominant ones → more of it can be pruned at the same
accuracy; a low-dispersion (uniform) group has little dead weight → it is more fragile. The paper is
*built around* this lens: dispersion is the signal that both **motivates** per-group budgeting and
**parameterizes the budget-allocation rule** (allocate larger pruning budgets to higher-dispersion
groups). We measure dispersion scale-free with the **coefficient of variation (CV = std/mean)** of
the per-token L2 norm — deliberately *not* the absolute norm, because absolute activation norm grows
with transformer depth as a residual-stream artifact and is therefore confounded.

Phase 2 result (see §13 Phase 2 Findings): the dispersion gap is real, monotonic, and replicated —
G0 (ViT L5) CV ≈ 0.61, G1 (L11) ≈ 0.45, G2 (L17) ≈ 0.42. This **motivates** the method but does not
**prove** differential accuracy sensitivity.

Supporting evidence: HiPrune finds that middle vision-encoder layers tend to capture object-centric features, while deeper layers encode more global contextual representations. LaCo's success with intermediate vision-layer compression also supports that visual-token behavior differs across layers.

This is the go/no-go experiment: if different groups show the same sensitivity, per-depth budgeting loses its justification. **Crucial caveat:** dispersion is a *proxy* — the hypothesis is confirmed or rejected only by the Phase 3 ablation / Experiment 2–3 (prune each group, measure per-task accuracy drop), not by the Phase 2 feature statistics alone.

---

## 5. Research Questions

| # | Question |
|---|---|
| RQ1 | Are DeepStack visual feature groups equally important? |
| RQ2 | Does DeepStack-aware token budgeting beat uniform pruning? |
| RQ3 | Does DeepStack-aware budgeting beat global attention pruning? |
| RQ4 | Does it produce real speed/memory improvements? |
| RQ5 | Which tasks are most sensitive to DeepStack compression? |

---

## 6. Method

### Level 1: Per-DeepStack-group budget allocation

Each DeepStack visual feature group gets a separate token budget.

| Strategy | Description |
|---|---|
| Uniform | Same token budget for every group |
| Manual per-depth | Fixed different budgets by group |
| Sensitivity-calibrated | Budget based on ablation sensitivity |
| Attention/saliency-calibrated | Budget based on visual-token usefulness scores |
| Task-aware (extension) | Different budget by task type |

**Primary publishable method:** Sensitivity-calibrated DeepStack token budgeting — run a calibration set, estimate which DeepStack groups are more fragile, give them larger budgets.

### Level 2: Within-group token selection

After each group gets a budget, choose which tokens to keep inside that group.

| Scoring method | Rationale |
|---|---|
| Random | Control |
| Spatial uniform | Preserves spatial coverage |
| Attention-based | Common baseline |
| Vision saliency | Avoids relying solely on decoder attention |
| Activation magnitude | Captures strong feature activity |
| Similarity/diversity | Avoids keeping duplicate tokens |
| Hybrid score | Likely strongest |

**Note:** VisPruner reports that text-visual attention in the language model is not an ideal indicator for visual-token pruning. A better approach uses visual cues plus duplicate removal.

**Proposed hybrid score:**
```
importance = visual_saliency + activation_magnitude + diversity_bonus
```

Then keep top-k tokens inside each DeepStack group.

---

## 7. What Is Being Optimized

This method reduces:
- Number of visual tokens injected
- Memory moved during feature fusion
- Visual feature storage
- Prefill computation
- Downstream hidden-state visual mass
- Possibly KV/cache pressure

This method does **not** primarily optimize the mechanical injection operation itself (`.clone()`, `masked_scatter`, Python overhead).

---

## 8. Model

**Primary model:** Qwen3-VL (DeepStack-style multi-level visual feature fusion)
- Start with smallest practical variant: Qwen3-VL-2B or Qwen3-VL-4B
- Dense and MoE variants available

**Optional second model:** another DeepStack-style model, or a non-DeepStack VLM as contrast.

Do not overpromise cross-model generality unless multiple architectures are actually tested.

---

## 9. Baselines

| Baseline | Description | Purpose |
|---|---|---|
| Full DeepStack | Original model, no compression | Accuracy, latency, memory reference |
| No DeepStack | Disabled DeepStack groups (if possible) | Shows how much DeepStack matters |
| Uniform pruning | Same ratio for every group (25%, 50%, 75%) | Tests whether per-group budgeting helps |
| Global top-k | Score all visual tokens globally, keep top-k | Tests whether DeepStack structure matters |
| Random per-group | Random selection at same budget | Tests whether importance scoring matters |
| Spatial uniform | Evenly distributed tokens across patches | Controls for spatial coverage |
| Attention-only | Attention scores only | Tests against common baseline |
| ViT-layer pruning | Prune inside source ViT layers, no DeepStack-aware injection | Closest competitor (LaCo/HiPrune style) |
| Reproduced prior method | CoViPAL / LaCo / HiPrune / ST3 style | One strong close competitor is better than five shallow ones |

---

## 10. Experiments

### Experiment 1: Profiling Baseline
**Goal:** Show where time and memory go before any optimization.

**Measure:**
- Preprocessing time
- Vision encoder time
- DeepStack feature extraction time
- DeepStack feature injection/fusion time
- Prefill time
- Decode time per generated token
- Total generation time
- Peak GPU memory
- Token count per modality
- Visual tokens per DeepStack group

**Existing data point:** preprocessing 1,935.9 ms, generation 12,084.8 ms, text decoder 10,585.5 ms, vision encoder 863.7 ms.

**Expected output:** Stacked bar chart of inference time by component. This becomes the motivation figure.

---

### Experiment 2: DeepStack Group Sensitivity Ablation
**Goal:** Prove that different DeepStack groups matter differently.

**Conditions:**
- Full DeepStack
- Remove group 1 only
- Remove group 2 only
- Remove group 3 only
- Keep group 1 only
- Keep group 2 only
- Keep group 3 only
- Remove all DeepStack groups

Measure quality on multiple tasks for each condition.

**Go/no-go signal:** If removing/compressing different groups causes different performance drops, the per-depth idea is valid. If all groups behave the same, the method needs rethinking.

---

### Experiment 3: Uniform vs Per-Group Pruning
**Goal:** Show DeepStack-aware budgets matter.

**Important:** Compare at the **same total retained-token count**.

Example:
- Uniform 50% = keep 50% of all visual tokens, same per group
- DeepStack-aware 50% average = keep 50% total but allocate differently per group

This proves gains come from *where* tokens are allocated, not from keeping more.

---

### Experiment 4: Within-Group Scoring Comparison
**Goal:** Show token selection method is better than random or attention-only.

Compare inside each DeepStack group:
- Random
- Spatial uniform
- Attention-only
- Vision saliency
- Activation magnitude
- Diversity-aware
- Hybrid score

---

### Experiment 5: Latency and Memory Benchmark
**Goal:** Prove real efficiency — not just token count reduction.

**Measure:**
- Wall-clock total latency
- Time-to-first-token
- Prefill latency
- Decode latency per token
- Generated tokens/sec
- Peak GPU memory
- KV-cache memory (if affected)
- Visual token count
- GPU utilization

**Fixed conditions:** same prompt, same `max_new_tokens`, same precision, same device, same batch size, warmup runs, multiple trials, report mean and standard deviation.

**Warning:** Do not only report FLOPs. Dynamic token selection can create overhead or poor hardware utilization that erases theoretical gains.

---

### Experiment 6: Quality Evaluation by Task Type
**Goal:** Show method does not only work on easy tasks.

| Task type | Reason needed |
|---|---|
| Captioning | Likely tolerant to compression |
| General VQA | Standard multimodal ability |
| TextVQA / OCR | Fragile, needs fine visual detail |
| DocVQA | Document text/detail sensitivity |
| Chart/table QA | Structured visual reasoning |
| Spatial reasoning | Location/relationship sensitivity |
| Counting | Token pruning can remove repeated objects |
| Multi-image QA | Visual context complexity |
| Video QA (optional) | High visual-token load |

DeepStack originally showed strong gains on TextVQA, DocVQA, and InfoVQA — test these especially.

---

### Experiment 7: Pareto Frontier
**Goal:** Show speed/accuracy tradeoff.

- x-axis: latency or retained token ratio
- y-axis: accuracy

Compare: full model, uniform pruning, global pruning, attention-only, DeepStack-aware budgeting.

Target: method sits above all others (same latency, better accuracy; or same accuracy, lower latency).

---

### Experiment 8: Failure-Case Analysis
**Goal:** Be honest; strengthen credibility.

Inspect examples where compression fails:
- Small text missed
- Wrong count
- Wrong spatial relation
- Hallucinated object
- Chart/table value error
- Multi-object confusion

---

## 11. Metrics

### Efficiency
- Total inference latency
- Time-to-first-token
- Prefill latency
- Decode latency/token
- Generated tokens/sec
- Peak GPU memory
- Visual tokens retained
- Average retained token ratio per DeepStack group
- FLOPs estimate (optional)

### Quality
- VQA accuracy
- Exact match
- Relaxed accuracy (OCR/numeric answers)
- CIDEr/BLEU/SPICE for captioning (if used)
- GPT/judge evaluation: optional, not primary
- Human spot-check for failure cases

### Compression
- Group 1 retained ratio
- Group 2 retained ratio
- Group 3 retained ratio
- Average retained ratio
- Total token reduction

---

## 12. Datasets

| Category | Dataset |
|---|---|
| General VQA | VQAv2, GQA, MME, MMMU subset |
| OCR / Document | TextVQA, DocVQA, InfoVQA, OCRBench |
| Spatial / Counting | TallyQA, GQA spatial subsets, synthetic |
| Chart / Table | ChartQA, AI2D, TableVQA |
| Captioning | COCO Caption subset, NoCaps subset |

Use small subsets first (500–2,000 examples). A strong controlled study on a small clean set is sufficient for a student research project.

---

## 13. Implementation Plan

### Phase 1: Understand Qwen3-VL DeepStack Internals — ✅ MAPPED

Verified directly against `local_transformers/models/qwen3_vl/` (dense model) and confirmed at
runtime by the non-invasive probe `src/deepstack/probe.py` (writes
`results/<ts>/deepstack_probe.json`). Findings:

- **ViT feature extraction** — `Qwen3VLVisionModel.forward` runs the 27 vision blocks; at each
  layer in `deepstack_visual_indexes` it passes that block's hidden state through a dedicated
  `Qwen3VLVisionPatchMerger` and appends the result to `deepstack_feature_lists`. The forward
  returns `(final_hidden_states, deepstack_feature_lists)`.
  *(modeling_qwen3_vl.py:737-753)*
- **Groups** — **3 groups**. The default in `Qwen3VLVisionConfig` is `[8, 16, 24]`
  (configuration_qwen3_vl.py:42), **but the actual Qwen3-VL-2B-Instruct checkpoint config overrides
  this to `deepstack_visual_indexes = [5, 11, 17]`** (confirmed at runtime by the probe — always read
  it from the loaded model, never assume the class default). One `PatchMerger` per group
  (modeling_qwen3_vl.py:590-599).
- **Where `deepstack_visual_embeds` are built** — in `Qwen3VLModel.forward` the per-group features
  are aligned to the visual placeholder positions, producing `visual_pos_masks` (bool, `(B, seq)`)
  and `deepstack_visual_embeds` (list of `(num_visual_positions, out_hidden_size)` tensors;
  **measured `out_hidden_size = 2048`** on the 2B model = the text decoder hidden size, so each group
  is already projected into the LLM space before injection). *(modeling_qwen3_vl.py:1153-1175)*
- **Injection** — in `Qwen3VLTextModel.forward`, after decoder layer `i` for
  `i in range(len(deepstack_visual_embeds))` (i.e. **decoder layers 0, 1, 2**), `_deepstack_process`
  **adds** group `i` onto the visual rows of the hidden state:
  `hidden_states[visual_pos_masks] += deepstack_visual_embeds[i]`. So vision layer 5→dec layer 0,
  11→1, 17→2. *(modeling_qwen3_vl.py:849-867, 876-883)*
- **Token shape per group** — `(num_visual_positions, 2048)`; `num_visual_positions` equals
  `Σ t·(h//spatial_merge_size)·(w//spatial_merge_size)` over `image_grid_thw`
  (`spatial_merge_size=2`). Probe cross-check passed: the demo image `grid_thw=[1, 86, 128]` →
  `1·43·64 = 2752` tokens per group, matching the measured count exactly.
- **Does pruning break positional encoding / the model?** The injection is a **strict 1:1
  count-contract**: `deepstack_visual_embeds[i].shape[0]` MUST equal `visual_pos_masks.sum()`
  (fixed by the prompt's image-placeholder tokens). MRoPE position IDs are computed once
  (modeling_qwen3_vl.py:846) and are **not** touched by injection. Therefore **naively dropping
  tokens from a group breaks the injection add (shape mismatch)** — confirmed by the probe's
  mutation test. The viable pruning pattern for Phase 4 is **prune-then-reconstruct-to-full-length**:
  select a subset, scatter kept tokens back into a full-length zero-filled tensor so the count
  (and thus position alignment) stays valid. The probe demonstrates this reconstruction passes.

#### Phase 1 Runtime Results (probe run `results/20260603_050848/`, Qwen3-VL-2B-Instruct, T4, fp16)

Single demo image, `grid_thw = [1, 86, 128]` → **2752 visual tokens per group**, 3 groups, all
`count_match = PASS`, grid cross-check `MATCH`, mutation test as predicted (naive 25% drop →
`ValueError: 2752 vs 2064`; reconstruct-to-full → PASS).

**Per-group token L2-norm distribution** (the first real research signal — distributions differ by
depth, an early prior for the non-uniform-sensitivity hypothesis, though sensitivity itself is
Experiment 2):

| Group | ViT layer | Inject @ dec | mean | std | min | max |
|---|---|---|---|---|---|---|
| 0 | 5 (shallow) | 0 | 15.1 | 9.7 | 8.97 | **141.7** |
| 1 | 11 (mid) | 1 | 17.9 | 8.8 | 6.72 | 68.4 |
| 2 | 17 (deep) | 2 | 23.2 | 10.8 | 8.51 | 62.8 |

Observations and what they imply for later phases:
1. **Norm grows with depth** (15 → 18 → 23). Deeper groups inject larger-magnitude features, so a
   fixed additive perturbation from pruning is *relatively* smaller deep, larger shallow — a hint
   that the budget schedule should not be uniform (relevant to Phase 5 budgeting).
2. **Group 0 is heavy-tailed** (max 141.7 ≈ 9× its mean; std ≈ mean). A few "massive-activation" /
   sink tokens dominate the shallow group. **Implication for Phase 4 scoring:** pure
   activation-magnitude scoring will be hijacked by these outliers → the **diversity term in the
   hybrid score matters most for group 0**; spatial-uniform and diversity baselines are important
   controls there.
3. **Visual mass is large:** 2752 tokens × 3 groups = 8256 added visual injections on top of the
   2752 base-embed tokens — confirms the motivation that DeepStack inflates decoder-side visual mass
   and that per-group pruning has real headroom.
4. **`out_hidden_size = 2048` = text hidden size** → groups are pre-projected into LLM space, so
   pruning/scoring operates directly on injectable 2048-d vectors (no extra projection needed in the
   prune module).

**Direct consequences for the build (carry into later phases):**
- Phase 3/4/5 code must read `deepstack_visual_indexes` from the loaded config (`[5, 11, 17]` here),
  never hard-code `[8,16,24]`. Budgets/ablation index by group position 0/1/2, not by ViT layer.
- Phase 4 `prune.py` must implement the **reconstruct-to-full-length (scatter + zero-fill)** contract
  proven here; a naive top-k that returns fewer rows will crash injection.
- The probe already emits per-group norm stats — Phase 2 extends the same hook scaffold with
  attention/saliency capture and latency/memory around extraction+injection rather than starting fresh.

### Phase 2: Add Measurement Hooks — ✅ DONE (run + figures)
Log:
- Token count per DeepStack group
- Feature norm distributions
- Attention/saliency scores
- Latency around feature extraction/injection
- Memory usage per group

Implemented in `src/deepstack/instrument.py` as a non-invasive `DeepStackInstrumentor` context
manager (extends the Phase 1 hook scaffold; edits no model source). It runs one prefill forward per
calibration sample (`CALIBRATION_IMAGES`: natural / OCR-text / chart / counting) and aggregates
**per-group distributions** over the set:

- **Token count** — from the `deepstack_feature_lists` returned by `Qwen3VLVisionModel.forward`;
  cross-checked against `visual_pos_masks.sum()` (`count_matches_mask`).
- **Feature-norm distribution** — per-token L2 norms pooled over tokens×samples → mean/std/min/max,
  p10/p50/p90, 30-bin histogram (so the depth-increasing, group-0 heavy-tailed norm structure seen in
  Phase 1 is captured as a full distribution, not a point estimate).
- **Attention saliency (opt-in, `--capture-attention`)** — forces eager attention, hooks
  `Qwen3VLTextAttention` at the injection layers (`module.layer_idx`), takes the attention mass
  *received* by each visual key (mean over batch/heads/queries) → per-group saliency distribution.
  Off by default (O(seq²); T4-heavy).
- **Latency** — extraction timed by pre/post hooks on each deepstack `PatchMerger`; injection timed by
  bracketing decoder layers (injection *i* runs between layer *i*'s return and layer *i+1*'s call), all
  with CUDA sync.
- **Memory** — `memory_allocated()` deltas around extraction and injection, plus per-extraction peak
  via `max_memory_allocated()`.

Group count, vision layers, and `out_hidden_size` are read from the loaded config at runtime, so it
self-corrects to the actual `[5,11,17]`/`2048` rather than the class defaults. CLI mirrors the probe
(`--model-id/--device/--dtype/--output-dir/--num-samples/--capture-attention`); writes
`results/<ts>/deepstack_instrument.json`. Wired into `colab_run.ipynb` via the `RUN_INSTRUMENT`
toggle. Figures + a plain-English EXPLAINER are rendered by `src/deepstack/visualize.py` into
`results/<ts>/figures/`.

#### Phase 2 Findings (8 real images, attention on — `results/20260603_080941`)

**This is the central, paper-defining result: groups are non-uniform in feature dispersion, and that
is the dispersion lens of §4.**

1. **Same token count, different content.** All 3 groups carry identical token counts per image
   (DeepStack injects the same placeholder positions at every depth), so the question is token
   *value*, not token *count*.
2. **Dispersion gradient (the headline).** Per-token L2-norm **CV (std/mean)** = **0.61 / 0.45 / 0.42**
   for G0 / G1 / G2; max÷median skew = **8.9× / 4.3× / 2.9×**. Shallow G0 is highly unequal (a crowd of
   redundant low-norm tokens + a few outliers) → most prunable; deep G2 is uniform/dense → most
   fragile. Replicated on the earlier 4-image run. Illustrated by the "% of tokens below the overall
   median strength" figure: **68% / 51% / 22%**.
3. **Absolute norm grows with depth (14.6 → 17.0 → 23.6) — treated as a confound, not a result.**
   Residual-stream norms grow with depth regardless of importance; the paper leads with the scale-free
   **CV**, not absolute norm.
4. **Attention is an informative NULL.** Mean attention *received* per visual token is ~0.001–0.0016
   and near-identical across groups — it sits at the ~1/sequence-length uniform floor, averaged over
   all heads/queries at the earliest (positional) decoder layers, over identical positions. This
   reproduces **VisPruner**'s finding that LM text-visual attention is a poor pruning indicator →
   **the within-group scorer (§6 Level 2) should rely on vision-side signals (norm, vision saliency,
   diversity), not decoder attention.**
5. **DeepStack overhead is negligible (~5 ms total extract+inject).** So the efficiency win must come
   from *reducing the token count the decoder processes*, not from the injection op — consistent with
   §1's profiling that the decoder dominates.

**Status of the hypothesis:** Phase 2 shows non-uniformity in *feature structure* (supported,
replicated, scale-robust) — enough to justify building the per-group budgeting method. It does **not**
yet show non-uniformity in *accuracy/compression tolerance*; that is the job of Phase 3.

### Phase 3: Implement Ablation Switches — ✅ DONE (run `results/20260603_184741`)

This is the **go/no-go** experiment (Experiment 2): it is the experiment that
*confirms or rejects* per-group accuracy sensitivity, which Phase 2's feature-structure
result only *motivates* (see §4 caveat).

**Ablation mechanism — `src/deepstack/ablation.py` (`DeepStackAblator`).** Non-invasive,
edits no model source. Injection is a pure additive op
(`hidden_states[visual_pos_masks] += deepstack_visual_embeds[i]`) and the embeds reach the
text model as a forward kwarg, so a forward-pre-hook that **zeros group `i`'s embeds tensor**
reproduces "remove group `i`" exactly — the 1:1 count contract and MRoPE positions stay valid,
only the additive contribution becomes 0. Injection only fires on the prefill forward, so the
hook naturally affects only prefill. `standard_conditions(num_groups)` builds the **8 conditions**:
`full`, `drop_g0/1/2`, `keep_g0/1/2`, `drop_all` (`drop_all` == the §9 "No DeepStack" baseline).
Group count is read from the loaded config's `deepstack_visual_indexes` (never hard-coded).

**Experiment runner — `src/experiments/exp_sensitivity.py`.** For each of 4 task types and each of
the 8 conditions, over 100 labeled samples/task (400 samples total, 3 200 condition-evaluations), it
measures:
- **Labeled accuracy** (the headline Figure-3 signal): VQA soft-accuracy (General VQA, TextVQA),
  ANLS (DocVQA), integer exact-match (Counting). Scorers are pure-Python (canonical VQA
  normalization; in-file Levenshtein for ANLS).
- **First-token KL** (dense, reference-free): `KL(P_full ‖ P_cond)` of the first generated-token
  distribution vs full DeepStack.

**Visualization — `src/deepstack/visualize_sensitivity.py`**: `sensitivity_heatmap.png` (Figure 3),
`kl_by_condition.png`, `EXPLAINER_sensitivity.md`.

#### Phase 3 Findings (100 samples/task, Qwen3-VL-2B-Instruct, T4, fp16 — `results/20260603_184741`)

**Full accuracy by condition:**

| Task | full | drop_g0 | drop_g1 | drop_g2 | keep_g0 | keep_g1 | keep_g2 | drop_all |
|---|---|---|---|---|---|---|---|---|
| general_vqa | .833 | .820 | .840 | .833 | .830 | .833 | .800 | .837 |
| textvqa | .840 | .853 | .843 | .800 | .800 | .817 | .823 | .807 |
| docvqa | .890 | .900 | .900 | .898 | .866 | .900 | .898 | .837 |
| counting | .820 | .820 | .830 | .820 | .840 | .830 | .830 | .780 |

**First-token KL(P_full ‖ P_cond) pooled over all tasks:**
drop_g0=0.038 | drop_g1=0.020 | drop_g2=0.028 | keep_g0=0.089 | keep_g1=0.112 | keep_g2=0.068 | drop_all=0.193

**Finding 1 — G2 (ViT L17, deep) is task-specialized, not universally dominant.**
TextVQA shows the sharpest signal in the entire experiment: `drop_g2` costs **4.0% accuracy** —
the largest single-group accuracy drop across all tasks and conditions. `keep_g0` produces the
identical 4.0% drop, meaning shallow features alone are as bad as having no deep group at all.
This is a direct, clean read: **G2 carries information that G0 and G1 together cannot replicate
for text-reading tasks.** For general_vqa and counting, G2 is interchangeable with other groups
(drop_g2 ≈ 0%).

**Finding 2 — G0 (ViT L5, shallow) is the most expendable group and mildly harmful for detail tasks.**
For both TextVQA (`drop_g0` = +1.3% *improvement*) and DocVQA (`drop_g0` = +1.0% improvement),
dropping G0 *improves* performance. This is consistent with Phase 2's finding that G0 is highly
dispersed (CV=0.61) with many weak/noisy tokens. For tasks requiring fine-grained character
recognition, those noisy tokens appear to slightly interfere with the more informative signals from
G1 and G2. **G0 is the best candidate for aggressive pruning across all tasks.**

**Finding 3 — The aggregate deepstack contribution is real; individual groups are largely
interchangeable except G2 on OCR tasks.**
`drop_all` hurts consistently across all four tasks: −0.3% (general_vqa, barely), −3.3% (textvqa),
−5.3% (docvqa), −4.0% (counting). But dropping any single group costs ≤1.3% for three of four
tasks. The groups provide partially overlapping coverage — any two can compensate for the third —
except that G0+G1 cannot compensate for G2 on OCR.

**Finding 4 — General VQA and counting are nearly immune to individual group ablations.**
For counting, all three single-group drops give 0% or negative (improvement) changes. Only
`drop_all` registers a real 4% decline. These tasks rely primarily on coarse spatial features
that the base image embeddings capture; the deepstack injections provide refinement, not critical
information.

**Finding 5 — The KL signal reveals G1 as the "bridging" group.**
`keep_g1` (only G1 active) produces the *highest* KL at 0.112, higher than `keep_g0` (0.089).
Yet `drop_g1` produces the *lowest* single-drop KL at 0.020. This asymmetry means: G1 carries
information that is most distinct from what G0+G2 together provide (high KL when isolated), but G1's
individual contribution is maximally redundant when the other two groups are present (lowest KL when
removed). **G1 is the bridging group** — it interacts across depth scales in a way neither G0 nor G2
does alone, but its contribution is absorbed by the ensemble.

**Finding 6 — Several conditions beat "full" accuracy** (drop_g0 for TextVQA, all three single-drop
conditions for DocVQA). These negative drops are most likely noise at n=100 — ANLS on document text
and VQA soft-acc both carry ±1–2% variance at this sample size. A subset may reflect genuine marginal
noise from G0's weak tokens (consistent with Finding 2), but should not be over-interpreted. The
DocVQA single-drop conditions are best treated as indistinguishable from zero; the real DocVQA signals
are `keep_g0` (−2.3%) and `drop_all` (−5.3%).

**Go/No-Go verdict: WEAK GO — task-dependent sensitivity confirmed, universal sensitivity not confirmed.**

The hypothesis that "groups have non-uniform compression sensitivity" is **supported for OCR/text
tasks**: the G2 signal on TextVQA (4.0% drop) is real, directional, and consistent across both the
accuracy metric and the KL cross-check. It is **not supported as a universal property across all task
types** — for counting and general VQA the groups are nearly interchangeable individually.

The strongest honest version of this paper's contribution is: *DeepStack groups show
**task-dependent** sensitivity. The deep group (G2, ViT L17) is disproportionately load-bearing
for OCR/text tasks while being dispensable for spatial/counting tasks, and the shallow group (G0,
ViT L5) is slightly harmful for detail-sensitive tasks due to its high noise fraction. This
justifies **task-aware differential token budgeting** rather than uniform pruning.*

**Critical caveat:** this run used 100 samples/task. Several conditions beat full accuracy (a
statistical impossibility in expectation), indicating ±1–2% variance at this scale. Results should
be confirmed at larger scale (500+ samples) before being cited as definitive numbers. The directional
findings (G2 matters for OCR; G0 is prunable; drop_all hurts all tasks) are robust enough to build on.

**Implications for Phase 4/5 design** (connects to earlier efficiency conversation):

Since zeroing groups does not reduce sequence length (the ablation is diagnostic only), real
efficiency must come from reducing visual token count at the input level. Phase 3's calibration
table directly parameterizes this:

| Task class | Group sensitivity | Resolution tolerance | Pruning strategy |
|---|---|---|---|
| TextVQA / DocVQA | G2 critical, G0 harmful | LOW — needs full resolution | Prune G0 aggressively; protect G2 |
| General VQA | All groups dispensable individually | MEDIUM | Moderate uniform reduction |
| Counting | All groups individually redundant | HIGH — tolerant | Aggressive reduction |

G0 being expendable (and mildly harmful for text tasks) is the most immediately actionable finding:
pruning G0 most aggressively is safe across all tasks and may marginally improve OCR performance.

### Phase 4: Implement Pruning — ✅ DONE (run `results/20260603_204751`)

**The hard constraint (from Phase 1).** DeepStack injection is a strict 1:1 additive op
(`hidden_states[visual_pos_masks] += deepstack_visual_embeds[i]`), so a group's embeds tensor MUST
keep exactly `visual_pos_masks.sum()` rows or the add crashes (proven by the Phase 1 probe mutation
test). Therefore "pruning" is implemented as **reconstruct-to-full-length**: score all N tokens,
choose k to keep, and **zero the (N−k) pruned rows** while preserving the `(N, D)` shape. A zeroed
row contributes 0 to the additive injection — exactly as if its feature were never injected — while
the count contract and MRoPE positions stay valid.

**Pruning module — `src/deepstack/prune.py`.** Five within-group scorers (paper.md §6 Level 2),
each returning **exactly `k = round(keep_ratio·N)` unique kept indices** so methods are compared at
an identical retained-token count (the fairness requirement of Experiment 3/4):

1. **`random`** — uniform random keep (seeded). The control; a useful scorer must beat it.
2. **`spatial_uniform`** — even subsample on the merged token grid (true 2D row/column lattice when a
   single image's `grid_thw` is known; flat raster stride otherwise / multi-image). Tests whether
   spatial coverage alone explains any gains.
3. **`activation_magnitude`** — keep highest per-token L2 norm. Primary vision-side signal (Phase 2
   showed decoder attention is an informative null). *Known weakness:* Phase 1/2 found G0 has outlier
   "sink" tokens (max norm ≈ 9× mean) that hijack pure magnitude — motivating the diversity term.
4. **`diversity`** — farthest-point sampling (FPS) in a fixed 32-d random projection of the
   L2-normalized features; seeded by hidden size so it is deterministic. Keeps a feature-space-spread
   subset, avoiding near-duplicates.
5. **`hybrid`** — magnitude-seeded greedy selection maximizing `α·norm_z + β·min_dist_to_selected`
   (α=β=0.5). The paper's proposed `saliency + magnitude + diversity_bonus`; same O(N·k) cost as FPS,
   balances "keep strong" and "keep diverse".

`DeepStackPruner` is a non-invasive context manager (mirrors `DeepStackAblator`, edits no model
source): a pre-hook on `Qwen3VLTextModel.forward` prunes each configured group's embeds before
injection; a pre-hook on `Qwen3VLModel.forward` captures `image_grid_thw` for the spatial scorer.
Greedy selection keeps all ops on-device (no per-step host sync) so it stays cheap on the GPU.
Local verification (CPU, no model): the full-length contract holds for every scorer at
keep_ratio ∈ {1.0, 0.75, 0.50, 0.25}, all return exact-k unique indices, magnitude retains the sink
tokens, and the no-grid / multi-image fallbacks return the right count.

**Experiment runner — `src/experiments/exp_scoring.py` (Experiment 4).** For each task × scorer ×
keep-ratio it applies a **uniform budget across all 3 groups** (same keep-ratio per group) — this
isolates *which tokens to keep* from *how to split the budget across groups* (the latter is Phase 5).
It reuses the Phase 3 task registry and scorers (VQA soft-accuracy / ANLS / integer exact-match) and
measures labeled accuracy plus first-token KL vs the no-pruning baseline. Conditions: 1 baseline +
5 methods × 3 pruning ratios = 16; default 100 samples/task → 6,400 generations. Writes
`results/<ts>/scoring.json`.

**Visualization — `src/deepstack/visualize_scoring.py`**: `scoring_accuracy_curves.png` (per-task
accuracy vs keep-ratio, one line per scorer), `scoring_bar_at_50pct.png` (grouped bars at keep-ratio
0.50), and a data-driven `EXPLAINER_scoring.md`. Wired into `colab_run.ipynb` via the `RUN_SCORING`
toggle.

#### Phase 4 Findings (100 samples/task, Qwen3-VL-2B-Instruct, T4, fp16 — `results/20260603_204751`)

> **⚠️ Correction (superseded in part by Phase 4b).** The original verdict below over-claimed
> "activation_magnitude wins." It does not cleanly: at the discriminating ratio (0.25) magnitude and
> hybrid are a **dead heat**, and the KL signal that favored magnitude is **structurally biased**
> toward it (see the caveat below). Two things are solid and stand: (a) the **controls lose** —
> random, spatial_uniform, and diversity are all clearly worse; (b) the **large compressibility
> headroom** (50–75% droppable within noise). What is *not* settled is the leader, and the literature's
> strong **vision-encoder attention** signal was never tested here. Phase 4b adds `vision_attention`
> and re-runs at 300 samples to settle it on accuracy. Read this section as the run-1 record, then
> Phase 4b for the resolved verdict.

**Read the KL, not the raw accuracy.** At n=100 the labeled-accuracy metric is dominated by noise:
~40% of the (task × scorer × ratio) cells show a *negative* drop (pruning "improving" accuracy),
which is impossible in expectation — the same ±1–2% variance Phase 3 flagged. The auto-generated
`EXPLAINER_scoring.md` ranks scorers by this noisy accuracy and therefore picks different "winners"
per task (random/spatial/diversity); that ranking is an artifact of the noise. The reliable signal is
the **first-token KL(P_full ‖ P_pruned)**, which is dense and rises smoothly and monotonically as the
keep-ratio drops (0.75 → 0.50 → 0.25) for every scorer, confirming the prune hook genuinely bites.

**Caveat on the KL signal (stated up front):** KL measures how far pruning shifts the output
distribution, and `activation_magnitude` keeps the largest additive features, so it *mechanically*
minimizes the L2 perturbation to the hidden state → low KL. So magnitude winning on KL is partly
structural. What rescues the finding from circularity is that **accuracy independently agrees on the
one task clean enough to read it** — DocVQA at keep-ratio 0.25 (highest baseline, lowest-variance
ANLS): random +3.9% (worst in the experiment), spatial +2.5%, diversity +2.7%, hybrid +0.2%,
**magnitude −0.8% (best)**. Accuracy is not circular (it scores task correctness, not distribution
distance), and it confirms magnitude/hybrid ≫ random/spatial/diversity.

**KL win-count across all 12 (task × pruning-ratio) cells — lower KL = better token selection:**

| scorer | cells won | role |
|---|---|---|
| activation_magnitude | **6** | best overall; dominant on detail tasks (DocVQA, counting) |
| hybrid | **5** | tied; best on text/soft tasks (TextVQA, general VQA) |
| random | 1 | only at 0.75, where all scorers are within noise |
| diversity | **0** | never wins; sometimes *worse* than random (counting@0.25) |
| spatial_uniform | **0** | never wins; often the worst (e.g. general_vqa@0.50 KL 0.048 ≈ 3× magnitude's 0.015) |

Representative KL at the aggressive keep-ratio 0.25 (random / spatial / magnitude / diversity / hybrid):
general_vqa 0.036 / 0.043 / 0.030 / 0.044 / **0.028**; textvqa 0.114 / 0.105 / 0.086 / 0.098 / **0.066**;
docvqa 0.173 / 0.173 / **0.120** / 0.163 / 0.121; counting 0.069 / 0.072 / **0.049** / 0.087 / 0.067.

**Finding 1 — Activation magnitude wins; the sink tokens were the signal, not the problem.**
Phase 1/2 worried that G0's outlier massive-activation tokens (max norm ≈ 9× mean) would *hijack*
magnitude scoring. The data shows the opposite: keeping high-norm tokens is exactly right. Massive-
activation tokens are functionally critical in transformers (attention sinks / global-info stores);
magnitude correctly identifies them as load-bearing rather than being fooled by them.

**Finding 2 — Diversity (FPS) fails — a genuine negative result.** The paper's prior (echoing
VisPruner's duplicate-removal intuition, and the Phase 1 note that "the diversity term matters most
for group 0") predicted diversity would help, especially for the redundant shallow group. It does
not: pure farthest-point sampling discards the high-norm tokens that matter and is the *worst* scorer
on counting@0.25. **The VisPruner-style diversity intuition does not transfer to DeepStack additive
groups.** This supersedes the Phase 1 speculation.

**Finding 3 — Spatial coverage is not what matters; feature content is.** `spatial_uniform` is
consistently near-worst. Keeping an even spatial grid is much worse than keeping high-activation
tokens — killing the "you only need spatial coverage" alternative explanation. The gains are about
*which features*, not *where*.

**Finding 4 — DeepStack's additive refinement is highly compressible (large headroom).** At
keep-ratio 0.50, every scorer holds accuracy within noise on every task; at 0.25, magnitude *gains*
on DocVQA (−0.8%) and counting (−3%). KL grows to 0.12–0.17 at 0.25 while accuracy barely moves —
aggressive pruning perturbs the model's *confidence* but rarely flips the *answer*. Strong evidence
that the DeepStack token mass is redundant and has real pruning headroom (the paper's core premise).

**Reconciliation with Phase 3.** Consistent and mutually reinforcing: TextVQA (Phase 3's fragile
OCR task, G2-critical) is exactly where scorer choice matters most and where hybrid pulls ahead at
aggressive pruning; DocVQA is scorer-sensitive (bad scorers cost +3.9%, magnitude costs nothing);
counting/general VQA are robust (flattest curves, smallest KL).

**Two caveats to carry forward.**
1. *Noise:* accuracy at n=100 can only rank scorers at the extremes; the KL ranking is solid but
   structurally favors magnitude. Both signals point the same way (the strongest available form of
   the claim), but a citable headline number ("magnitude beats random by X%") needs 500+ samples.
2. *No latency win here, by design:* the reconstruct-to-full-length prune zeroes a token's additive
   refinement but does **not** shorten the sequence — the token position still exists. Phase 4 is the
   diagnostic that establishes *which tokens to keep*; it yields no speedup on its own. Real
   efficiency requires an actual sequence-shortening mechanism, and §10 Experiment 5's warning
   applies (token cuts don't automatically become wall-clock gains — measure them).

**Decision (original, now deferred to Phase 4b).** Run 1 suggested fixing the scorer to
`activation_magnitude` with `hybrid` as an ablation and dropping `diversity`/`spatial_uniform`. This is
deferred: magnitude vs hybrid is unsettled at n=100 and the KL ranking is biased, and the literature's
vision-encoder attention signal was untested. Phase 4b resolves the scorer choice on accuracy at 300
samples before Phase 5 commits to one.

#### Phase 4b: Vision-encoder attention scorer + accuracy-led re-run — ✅ DONE (run `results/20260604_001256`)

**Why.** The run-1 leader was unsettled and the strongest training-free signal in the literature was
missing. Per **VisPruner (ICCV 2025, arXiv:2412.01818)** and **FasterVLM**, the **vision-encoder**
attention (not LM/decoder attention) is the SOTA importance signal: it "decisively outperforms"
text-visual/decoder attention and random selection (VisPruner ablation: +5% TextVQA, +1.5% POPE over
random; FasterVLM prunes 95% of tokens keeping ~90% performance). Decoder attention is correctly
excluded (Phase 2's null; VisPruner's "attention shift/dispersion").

**CLS-free adaptation (important).** VisPruner's canonical signal is **[CLS]→patch** attention, but
**Qwen3-VL's ViT has no [CLS] token and no register tokens** (verified: `patch_embed → pos_embed →
blocks → merger`, pure patch tokens). VisPruner gives the exact recipe for this case ("for models
without a [CLS] token, e.g. SigLIP, average the rows of A = the average attention each patch token
receives"). Our `vision_attention` scorer implements exactly that: **per-patch attention-received** at
the DeepStack source ViT layers (5/11/17). This is literature-sanctioned, not improvised.

**Implementation.** `src/deepstack/saliency.py` (`VisionAttentionCapturer`): forces vision-only eager
attention and wraps the module-global `eager_attention_forward` to capture the weights the model
already computes (no second matmul), identifying the 3 source-layer attention modules by object
identity. The pre-merge→post-merge map is exact (`PatchMerger.view(-1, hidden·merge²)` ⇒ 4 contiguous
patches per merged token ⇒ `received.reshape(N,4).mean(1)`; guarded by `received.numel()==4N`). Saliency
is image-only and captured once during the eager `full` baseline forward, then reused across keep-ratios
(no extra forward, no ordering problem). `prune.py` adds `vision_attention` as a score-based selector
(top-k by saliency; same exact-k + reconstruct-to-full-length contract). `exp_scoring.py` defaults to
methods `{random, activation_magnitude, hybrid, vision_attention}` (drops the proven losers from the
default set; both remain available via `--methods`), keep-ratios `{1.0, 0.50, 0.25}`, 300 samples/task
(~10,800 generations), with per-task checkpointing of `scoring.json`. On capture failure/OOM,
`vision_attention` is marked N/A for that sample and the run continues.

**Verdict is read off accuracy** (KL stays magnitude-biased). One plausible outcome to anticipate: in a
CLS-free encoder, attention-received often correlates with activation magnitude (sink tokens have both
high norm and high attention), so `vision_attention ≈ activation_magnitude` is a real possibility — and
would itself explain run 1's magnitude/hybrid ambiguity.

#### Phase 4b Findings (300 samples/task, Qwen3-VL-2B-Instruct, T4, fp16 — `results/20260604_001256`)

Methods: random (control), activation_magnitude, hybrid, vision_attention. Keep-ratios 1.0/0.50/0.25.
**Ranked by accuracy** (KL remains structurally magnitude-biased — see the run-1 caveat).

**Accuracy drop vs full (positive = worse; bold = best per cell):**

| task (full) | ratio | random | magnitude | hybrid | vision_attn |
|---|---|---|---|---|---|
| general_vqa (.818) | 0.50 | −.004 | **−.013** | −.008 | −.004 |
|  | 0.25 | −.004 | −.007 | −.002 | −.002 |
| textvqa (.824) | 0.50 | +.020 | +.012 | +.009 | **+.002** |
|  | 0.25 | +.017 | **+.010** | +.012 | +.031 |
| docvqa (.892) | 0.50 | +.010 | +.007 | **+.0003** | +.018 |
|  | 0.25 | +.060 | +.021 | **+.020** | +.031 |
| counting (.813) | 0.50 | +.003 | −.017 | **−.020** | −.010 |
|  | 0.25 | −.007 | **−.027** | −.010 | −.010 |

**Finding 1 — random clearly loses (confirms run 1 at scale).** Worst/near-worst on KL almost
everywhere; on the discriminating task DocVQA@0.25 it collapses (**+6.0%** accuracy drop, KL 0.248 —
the largest degradation in the experiment). Token selection genuinely matters.

**Finding 2 (headline) — vision_attention does NOT win; it is mixed-to-disappointing.** This is the
key, somewhat surprising result and the answer to "does the literature's strong signal transfer?"
- Competitive only at **mild** pruning on text-reading tasks: textvqa@0.50 it is the *best* scorer
  (−0.2% drop), docvqa@0.50 it is 2nd on KL.
- **Degrades toward random at aggressive (0.25) pruning and on spatial/general tasks:** textvqa@0.25
  it is the *worst* (+3.1% vs magnitude's +1.0%); general_vqa@0.25 and counting@0.25 its KL (0.040,
  0.092) sits at the random floor (0.042, 0.099).
- On the cleanest test **DocVQA@0.25** the ordering is **hybrid ≈ magnitude > vision_attention >
  random** (+2.0% / +2.1% / +3.1% / +6.0%) — vision_attention is beaten by *both* feature-based
  scorers. The pre-registered "vision_attention ≈ magnitude" guess is **rejected**: they pick
  different tokens and magnitude's choice is better at high compression.

  *Attributed causes (honest, not over-claimed):* (i) we read attention at the **intermediate**
  DeepStack source layers (5/11/17) — the correct place for DeepStack-aware pruning — not the *final*
  encoder layer VisPruner/FasterVLM use, where attention is more object-focused; (ii) Qwen3-VL's ViT
  has **no CLS anchor**, so "attention-received" is exposed to the **attention-sink** problem (a few
  patches dominate; the rest is diffuse), discarding task-relevant detail under aggressive pruning.
  We therefore claim vision-encoder attention does **not transfer to DeepStack intermediate
  source-layer selection**, not that it is universally useless.

**Finding 3 — hybrid is the most robust scorer; magnitude is the near-tied simple fallback.** hybrid
wins DocVQA (the discriminating task) on **both** accuracy and KL at both ratios, ties magnitude on
the insensitive tasks, and **never has a losing cell** — unlike vision_attention (collapses at 0.25)
and random (collapses on DocVQA). magnitude is essentially tied and parameter-free. The run-1
magnitude/hybrid dead heat resolves at 300 samples to **hybrid ≥ magnitude**; the diversity term in
hybrid is what removes magnitude's occasional slips.

**Finding 4 — large compressibility headroom, confirmed at scale.** keep-ratio 0.50 is within ~2% of
full for *every* method on *every* task; even 0.25 holds within ~2% on the hardest task (DocVQA) with
hybrid/magnitude. The DeepStack additive refinement is highly redundant — the paper's core premise.

**Finding 5 — task taxonomy is firm.** DocVQA/TextVQA are compression-sensitive (scorer choice
matters, real positive drops). general_vqa/counting are compression-*insensitive* (negative drops
persist at n=300 → genuine insensitivity, not just noise — the coarse base embeddings already suffice).

**Sharpened novelty.** Across Phases 2–4b we have now ruled out *both* attention families with
evidence: decoder attention (Phase 2 null) and vision-encoder attention (Phase 4b, mixed/near-random
at aggressive pruning). The right token-selection signal for DeepStack source-to-injection groups is
**feature-based (magnitude + diversity)**. This is a stronger, better-defended contribution than
"prune DeepStack tokens," and it distinguishes the work from "apply VisPruner to DeepStack."

**Caveats.** KL stays magnitude-biased (ranked on accuracy). vision_attention's weakness is partly
confounded by the intermediate-layer read vs VisPruner's final-layer use. Still diagnostic only —
zeroing, not sequence-shortening, so no latency/memory win yet (Experiment 5 remains the gate).

### Phase 5: Implement Budgeting — Stage A IMPLEMENTED (run pending), task-specialized to General VQA

**Scope decision (2026-06-04).** Phase 5 is split into two stages and **specialized to a single task,
General VQA (VQAv2)** — the most general, widely-used, scalable target. **Stage A** (this phase) finds,
*for General VQA*, the optimal **per-group keep-budget** `(r0,r1,r2)` and the optimal **within-group
scorer**, by **zeroing-based** pruning. **Stage B** (deferred, separate plan) realizes the chosen
optimum as **real sequence-shortening** and benchmarks actual latency/memory vs. the baseline and other
methods.

**Why zeroing for Stage A, and the zeroing-vs-real-pruning distinction (they are NOT the same).**
Confirmed against the model source: base visual tokens are scattered into the sequence
(`masked_scatter`, modeling_qwen3_vl.py:1143) and occupy their positions through every layer; DeepStack
only *adds* per-group refinements onto those positions at decoder layers 0/1/2
(modeling_qwen3_vl.py:881). So:
- **Zeroing** a group's refinement at a token removes that depth's additive contribution while the base
  token still occupies the sequence → sequence length, attention FLOPs, KV-cache, and latency are
  **unchanged**. It is purely diagnostic, but it is the **only** way to vary the three groups
  *independently* (they share positions), making it the correct tool for the per-group accuracy search.
- **Real pruning** drops the base token from the sequence → shorter sequence → real latency/KV-cache
  win, but with *different* logits (attention renormalizes over fewer keys; MRoPE positions shift). It
  cannot be done per-group independently. This is Stage B's job, applied to Stage A's optimum.
Stage A therefore compares methods at an **equal retained-token count** = equal count of non-zeroed
refinement rows (§10 Experiment 3's fairness unit), not equal sequence length.

**Within-group scorer (from Phase 4b): `hybrid` is the prior favorite** (magnitude+diversity; most
robust, never collapses), with `activation_magnitude` the near-tied parameter-free alternative and
`vision_attention` the literature ablation. Stage A does **not** assume the winner — it sweeps **all
four scorers** (`random`, `activation_magnitude`, `hybrid`, `vision_attention`) and picks the best on
General VQA empirically.

**Stage A experiment (implemented; awaiting Colab run).** Two modes in
`src/experiments/exp_budgeting.py`, driven by `src/deepstack/budget.py` and the Phase-4
`DeepStackPruner` (now with `set_keep_indices` for the global-top-k baseline):
- **`sweep`** — prune **each group independently** (others full) from 100%→0% in **5% steps**, for
  **every scorer**, on a General VQA calibration set (default 100). Records VQA soft-accuracy +
  first-token KL vs the no-pruning baseline → per-group sensitivity curves + the best scorer. Writes
  `budgeting_sweep.json`. (The `r=0` endpoint of each group's curve is a built-in anchor: it must
  reproduce Phase 3's `drop_g{i}`.)
- **`validate`** — water-fill the measured curves into candidate **joint** budgets `(r0,r1,r2)` at
  target average keep-ratios {0.7, 0.5, 0.3} (best scorer), and run them head-to-head vs **uniform**
  `(T,T,T)` and a flat **global top-k** baseline on a **disjoint held-out split** (default 300), all at
  an **equal retained-token count**, with bootstrap 95% CIs. Writes `budgeting_validation.json` — the
  Stage-A deliverable (best scorer + best per-group budget + the per-group-vs-uniform-vs-global verdict).

**Honest expectation for General VQA.** Phase 3/4b found it **compression-insensitive** (groups
individually interchangeable; negative drops persisted at n=300). So the per-group-beats-uniform signal
may be weak here — a tie with uniform is a plausible, *reportable* outcome ("General VQA tolerates
aggressive uniform pruning; a simpler policy suffices"), and the large compressibility headroom + the
empirical scorer choice remain solid, scalable results. The per-group contrast is strongest on
OCR/text (DocVQA/TextVQA) and can be added later.

The earlier fixed-schedule list (uniform / decreasing / increasing / sensitivity-calibrated guesses)
is superseded by the data-driven water-filling search above.

### Phase 6: Benchmark
Run all methods with fixed settings.

### Phase 7: Analyze
Produce:
- Latency table
- Accuracy table
- Per-task breakdown
- Pareto curve
- Per-group sensitivity heatmap
- Failure examples

---

## 14. Expected Paper Figures

| Figure | Content |
|---|---|
| Figure 1 | Architecture diagram: ViT → DeepStack groups → LLM injection depths, with pruning module shown before injection |
| Figure 2 | Baseline profiling: stacked bar of preprocessing / vision encoder / DeepStack / prefill / decoding |
| Figure 3 | DeepStack group sensitivity heatmap: rows = tasks, columns = compress group 1/2/3, color = accuracy drop |
| Figure 4 | Pareto curve: accuracy vs latency/token ratio |
| Figure 5 | Token retention by group: how the method allocates different budgets |
| Figure 6 | Failure cases: visual examples where aggressive pruning fails |

---

## 15. Success Criteria

| Criterion | Definition |
|---|---|
| Non-uniform sensitivity | Different groups tolerate different pruning levels |
| Better than uniform pruning | At the same retained-token budget, higher accuracy |
| Better than global pruning | Respecting DeepStack groups matters |
| Real efficiency gain | Measurable latency or memory improvement |
| Robustness on hard tasks | OCR/document/spatial performance does not collapse |

**Good result:** 30–50% DeepStack visual-token reduction with less than 1–2% average accuracy drop and measurable latency/memory improvement.

**Great result:** DeepStack-aware budgeting beats uniform pruning across OCR, VQA, and spatial tasks at the same token budget.

---

## 16. Failure Modes

| Failure mode | Response |
|---|---|
| Token reduction does not improve latency (bottleneck stays in decoder) | Reframe around memory / quality-preserving compression, or pivot |
| All DeepStack groups have similar sensitivity | Use global pruning or move to another idea |
| Uniform pruning performs just as well | Method is overcomplicated; novelty collapses |
| OCR/spatial tasks collapse | Add task-sensitive budgets or preserve high-detail groups |
| Dynamic pruning overhead erases gains | Use static calibration budgets or hardware-friendly pruning |

---

## 17. Paper Structure

**Abstract:** Problem, method, results.

**1. Introduction**
- VLMs are expensive because visual tokens inflate inference.
- DeepStack is a new architecture that injects visual features across depths.
- Existing compression methods treat visual tokens too generically.
- Contribution: DeepStack-aware token budgeting.

**2. Related Work**
- Visual token pruning
- Layer-wise visual compression
- KV-cache compression for VLMs
- DeepStack and multi-depth visual fusion
- Note that CoViPAL, LaCo, ST3, HiPrune, QAPruner, and VL-Cache are closest but do not specifically optimize DeepStack source-to-injection feature groups.

**3. Motivation and Profiling**
- Qwen3-VL profiling data: decoder dominates, visual tokens inflate decoder cost.

**4. Method**
- DeepStack groups
- Per-group budgeting
- Within-group scoring
- Pruning/merging process

**5. Experiments**
- Models, datasets, baselines, metrics, implementation details

**6. Results**
- Accuracy/latency, group sensitivity, ablation, Pareto curves

**7. Analysis**
- Which tasks need which groups
- Failure cases
- Hardware implications

**8. Limitations**
- DeepStack-specific method
- Tested primarily on Qwen3-VL
- Dynamic pruning overhead
- May not transfer to non-DeepStack VLMs
- Exact speedup depends on implementation

**9. Conclusion**
- DeepStack-aware token budgeting is a useful direction for efficient multimodal inference.

---

## 18. One-Sentence Summary

> This paper studies whether DeepStack-based vision-language models can run faster by reducing the number of visual tokens in each depth-specific DeepStack feature group, instead of pruning visual tokens globally or uniformly.

---

## 19. Title Options

| Title | Notes |
|---|---|
| DeepStack-Aware Visual Token Budgeting for Efficient Multimodal Inference | Primary — clean and descriptive |
| Source-to-Injection Aware Visual Token Compression for DeepStack Vision-Language Models | More technical |
| Reducing DeepStack Visual Token Mass via Depth-Aware Budget Allocation | More paper-like |
| Depth-Specific Visual Feature Compression for DeepStack Multimodal Transformers | Most precise |

---

## 20. Related Work Reference List

*(To be expanded as reading continues)*

- **CoViPAL**: Layer-wise contextualized visual-token pruning before LVLM processing.
- **LaCo**: Compresses visual tokens within intermediate layers of the vision encoder; reports >15% inference throughput improvement.
- **ST3**: Progressively prunes visual tokens across decoder layers and generation steps; ~2× faster inference, ~30% KV-cache memory vs LLaVA.
- **VL-Cache**: Modality-aware KV-cache compression; retains 10% of KV cache with comparable accuracy, up to 2.33× end-to-end latency speedup, 7.08× decoding speedup.
- **HiPrune**: Middle ViT layers capture object-centric features; deeper layers encode global context.
- **VisPruner**: Text-visual attention in the LM is not an ideal indicator for visual-token pruning; recommends visual cues + duplicate removal.
- **QAPruner**: (to be read/cited)
- **DeepStack original paper**: Visual features from intermediate layers fed into aligned LLM depths; strong performance at 1/5 context length.

---

## Current Status

- [x] Comprehensive research review complete
- [x] CLAUDE.md and paper.md created
- [x] Phase 1: DeepStack internals mapped in `local_transformers/` (probe: `src/deepstack/probe.py`; see §13 Phase 1)
- [x] Phase 2: Measurement hooks implemented + run + figures (`src/deepstack/instrument.py`, `visualize.py`; results/20260603_080941). Headline: non-uniform within-group dispersion (CV 0.61/0.45/0.42) = the dispersion lens; attention is an informative null. See §13 Phase 2 Findings
- [x] Phase 3: Ablation switches + sensitivity experiment run (`results/20260603_184741`). Verdict: WEAK GO — task-dependent sensitivity confirmed. G2 (deep) is critical for OCR/text (TextVQA drop_g2=−4%); G0 (shallow) is expendable and mildly harmful for detail tasks; groups are interchangeable for counting/general VQA. See §13 Phase 3 Findings
- [x] Phase 4: Pruning implemented + run 1 (`results/20260603_204751`). 5 scorers + reconstruct-to-full-length contract. Solid results: **controls (random/spatial/diversity) lose**; **DeepStack refinement is highly compressible** (50–75% droppable within noise). NOT settled: magnitude vs hybrid (dead heat; KL is biased toward magnitude; accuracy is noise at n=100). See §13 Phase 4 Findings + correction banner
- [x] Phase 4b: Vision-encoder attention scorer (`vision_attention`, CLS-free VisPruner/FasterVLM signal) + accuracy-led 300-sample re-run (`results/20260604_001256`). Verdict: **`vision_attention` does NOT win** — competitive only at mild pruning on text tasks, near-random at aggressive pruning, beaten by hybrid/magnitude on DocVQA@0.25 (tested negative result; attributed to intermediate-layer read + CLS-free attention-sink). **`hybrid` is the most robust scorer** (magnitude near-tied); random loses; headroom confirmed (50% ~free). Both attention families now ruled out → selection signal is feature-based. Phase 5 scorer = hybrid. See §13 Phase 4b Findings
- [~] Phase 5: Budgeting — **Stage A implemented** (code only; Colab run pending). Specialized to General VQA; `src/deepstack/budget.py` + `src/experiments/exp_budgeting.py` (`sweep`/`validate`) + `src/deepstack/visualize_budgeting.py`. Independent per-group 5% sweep × all 4 scorers (zeroing), then water-filled joint budgets vs uniform vs global-topk at equal retained-token count. Stage B (real sequence-shortening + latency) deferred. See §13 Phase 5
- [ ] Phase 6: Full benchmark run
- [ ] Phase 7: Analysis and figures
