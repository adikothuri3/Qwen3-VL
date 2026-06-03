# DeepStack-Aware Visual Token Budgeting for Efficient Multimodal Inference

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

Supporting evidence: HiPrune finds that middle vision-encoder layers tend to capture object-centric features, while deeper layers encode more global contextual representations. LaCo's success with intermediate vision-layer compression also supports that visual-token behavior differs across layers.

This is the go/no-go experiment: if different groups show the same sensitivity, per-depth budgeting loses its justification.

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

### Phase 1: Understand Qwen3-VL DeepStack Internals
Identify:
- Where ViT layer features are extracted
- Which layers become DeepStack groups (currently: layers 8, 16, 24 of the 27-layer vision encoder)
- Where `deepstack_visual_embeds` are stored
- Where they are injected into decoder hidden states
- Token shape per group
- Whether token pruning before injection breaks positional encoding

### Phase 2: Add Measurement Hooks
Log:
- Token count per DeepStack group
- Feature norm distributions
- Attention/saliency scores
- Latency around feature extraction/injection
- Memory usage per group

### Phase 3: Implement Ablation Switches
Add flags:
```
--disable_deepstack_group 0
--disable_deepstack_group 1
--disable_deepstack_group 2
--keep_deepstack_group 0
```

### Phase 4: Implement Pruning (start simple)
Order:
1. Random pruning per group
2. Spatial uniform pruning per group
3. Attention/saliency pruning per group
4. Hybrid scoring

### Phase 5: Implement Budgeting
Fixed budget schedules to test:
```
[100%, 100%, 100%]  # baseline
[75%,  75%,  75%]   # uniform
[50%,  50%,  50%]   # uniform aggressive
[75%,  50%,  25%]   # decreasing
[25%,  50%,  75%]   # increasing
[90%,  50%,  30%]   # sensitivity-calibrated guess
```
Then: sensitivity-calibrated budgets from ablation data.

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
- [ ] Phase 1: DeepStack internals mapped in `local_transformers/`
- [ ] Phase 2: Measurement hooks implemented
- [ ] Phase 3: Ablation switches implemented
- [ ] Phase 4: Pruning implemented
- [ ] Phase 5: Budgeting implemented
- [ ] Phase 6: Full benchmark run
- [ ] Phase 7: Analysis and figures
