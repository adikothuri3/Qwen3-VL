# How Compressible Is DeepStack? Feature-Based Token Pruning and the Limits of Depth-Aware Budgeting in Vision-Language Models

**[Author Name]**¹  **[Co-author(s)]**¹
¹[Affiliation]
Correspondence: [email]

*Draft prepared for [Venue / Course]. Code and all raw results: [repository URL].*

---

## Abstract

Vision-language models (VLMs) are increasingly bottlenecked by the sheer number of visual tokens their
decoders must process. **DeepStack** is a recent architecture that, instead of feeding all visual tokens
into the first decoder layer, extracts visual features from several intermediate vision-encoder layers
and *injects* them additively at correspondingly different decoder depths. This depth-structured design
raises a natural efficiency question: **because different depths carry different visual information, can
we assign each depth its own token budget and compress more aggressively than uniform pruning?**

We study this question empirically on Qwen3-VL-2B-Instruct across three VQA tasks (VQAv2, TextVQA,
DocVQA), using a controlled methodology that compares pruning strategies at an **equal retained-token
count**. We report three findings. **(1)** DeepStack's injected visual refinement is highly redundant: a
simple **feature-based uniform** prune removes **50–85% of the injected refinement at ≤2% accuracy
change** (held-out, n=300, with bootstrap confidence intervals). **(2)** *Which* tokens to keep matters
and is **feature-based, not attention-based** — both decoder-side attention and vision-encoder attention
(the signal used by VisPruner/FasterVLM) are no better than, and sometimes worse than, simple feature
magnitude/diversity at aggressive pruning. **(3)** *How* the budget is split across depth-groups does
**not** matter — **depth-aware per-group budgeting does not beat uniform allocation**, and neither does a
global cross-group top-k. We trace (3) to **mutual redundancy** across depth-groups: keeping a fraction
of every group already captures the overlapping information, so concentrating cuts in one group only
discards unique coverage. The practical recommendation is therefore simple: *prune DeepStack tokens
uniformly with a magnitude/diversity scorer; do not invest in attention scores or per-depth budgets.*
All compression here is measured diagnostically (by zeroing injected refinements); converting it to
wall-clock latency via real sequence-shortening is identified as the natural follow-up.

---

## 1. Introduction

Multimodal inference is becoming token-heavy. A single high-resolution image can expand into hundreds or
thousands of visual tokens, and modern VLMs support very long interleaved contexts. Profiling
Qwen3-VL-2B (Section 3) shows that generation is dominated by the **text decoder** (10,585.5 ms of
12,084.8 ms total generation time in a representative single-image run), while the vision encoder accounts
for only 863.7 ms. Visual tokens are expensive not primarily because they are *encoded*, but because they
*inflate the decoder's workload* at every layer. Reducing visual-token mass is therefore a central lever
for efficient multimodal inference, and visual-token pruning is an active research area.

**DeepStack** changes the standard VLM setup. Rather than injecting all visual tokens at the first decoder
layer, it extracts visual features from several intermediate vision-encoder layers and adds each group of
features onto the visual token positions at a different, aligned decoder depth. In Qwen3-VL-2B, three
groups are taken from ViT layers 5, 11, and 17 and injected at decoder layers 0, 1, and 2 respectively.
This produces a tempting hypothesis for compression: since shallow, middle, and deep features differ in
content, perhaps each depth-group tolerates a *different* amount of pruning, and a **depth-aware per-group
token budget** could beat uniform pruning at the same total token count.

We set out to build exactly such a method, and we tested the hypothesis rigorously. **The hypothesis did
not hold.** Our contributions, stated honestly including the negative results, are:

1. **A compressibility characterization of DeepStack.** Under a feature-based *uniform* prune, DeepStack's
   injected refinement is highly redundant — 50–85% of it can be removed within ~2% accuracy on held-out
   data, and entire single depth-groups are removable at ≤1% on two of three tasks.
2. **A token-selection result: attention fails, feature signals win.** We test decoder attention, the
   vision-encoder attention signal popularized by VisPruner/FasterVLM, and feature-based scorers. Both
   attention families are no better than feature magnitude/diversity, and vision-encoder attention decays
   to random at aggressive pruning. This is a negative result for transplanting attention-based pruning to
   DeepStack's intermediate source layers.
3. **A negative result for depth-aware budgeting.** At equal retained-token count and on a held-out split,
   per-group budgets do not beat uniform allocation on any task, and neither does global cross-group
   top-k. We explain this via mutual redundancy across depth-groups, and distill a simple, robust recipe.

We view the two negatives as useful: they map *where the intuitive idea breaks and why*, and they yield a
clear, deployable recommendation rather than a fragile method.

---

## 2. Related Work

**Visual-token pruning.** A large body of work prunes visual tokens to accelerate VLM inference.
**VisPruner** argues that text–visual attention *inside the language model* is not a reliable importance
signal and instead uses **vision-encoder** attention plus duplicate removal; **FasterVLM** similarly
relies on the vision encoder's `[CLS]`→patch attention and reports retaining ~90% performance while
pruning ~95% of tokens. Our token-selection experiments adopt the vision-encoder attention signal as a
strong baseline (adapted to Qwen3-VL's CLS-free ViT by averaging per-patch attention received) and find
it does **not** transfer to DeepStack's *intermediate* source layers.

**Intermediate-layer and progressive compression.** **LaCo** compresses visual tokens within intermediate
vision-encoder layers; **HiPrune** observes that middle ViT layers capture object-centric features while
deeper layers encode more global context; **ST3** progressively prunes across decoder layers and
generation steps. These motivate the premise that visual-token behavior differs across layers — the very
premise we test for *exploitability* at the allocation level.

**KV-cache compression.** **VL-Cache** and related methods compress the decoder KV-cache modality-aware.
This is complementary: we operate on the injected visual features before/at injection, not on the cache.

**DeepStack.** The original DeepStack work introduces depth-aligned injection of intermediate visual
features and reports strong performance at a fraction of the context length. To our knowledge, prior
pruning work treats visual tokens as a flat sequence or compresses generic ViT layers; **none specifically
studies whether DeepStack's source-to-injection depth structure can be exploited for token budgeting** —
the gap this paper addresses (and, ultimately, closes with a negative result plus a positive
compressibility characterization).

*(Full bibliographic citations: [TBD — VisPruner ICCV 2025, arXiv:2412.01818; FasterVLM; LaCo; HiPrune;
ST3; VL-Cache; CoViPAL; DeepStack original]. See References.)*

---

## 3. Background: DeepStack in Qwen3-VL

We verified DeepStack's mechanism directly against the Qwen3-VL-2B-Instruct model source and confirmed it
at runtime with a non-invasive probe.

**Mechanism.** The vision encoder runs its blocks once; at each layer in `deepstack_visual_indexes` it
passes that block's hidden state through a dedicated patch-merger and appends the result to a per-group
feature list. In the 2B checkpoint there are **three groups, taken from ViT layers [5, 11, 17]** (we read
this from the loaded config rather than assuming the class default of [8,16,24]). Each group is projected
to the decoder hidden size (out_hidden_size = 2048) and, inside the text model, **added** onto the visual
token positions after decoder layers 0, 1, and 2 respectively:
`hidden_states[visual_pos_masks] += deepstack_visual_embeds[i]`.

**The 1:1 count contract.** Injection is a strict additive operation: each group's feature tensor must
have exactly as many rows as there are visual placeholder positions, which are fixed by the prompt. Naively
dropping rows breaks the add (a shape mismatch, confirmed by a mutation test) and would also disturb the
multimodal rotary position encodings. We therefore implement pruning as **reconstruct-to-full-length**:
score all N tokens, keep k, and **zero the (N−k) pruned rows** while preserving the (N, 2048) shape. A
zeroed row contributes nothing to the additive injection, exactly as if its feature were never injected,
while counts and positions stay valid.

**Zeroing vs. real pruning (an important distinction we keep throughout).** Base visual tokens are
scattered into the sequence and occupy their positions through every layer; DeepStack only *adds*
per-group refinements at depths 0/1/2. **Zeroing** a group's refinement removes that depth's additive
contribution but leaves the token in the sequence — so sequence length, attention FLOPs, and KV-cache are
**unchanged**. Zeroing is purely diagnostic, but it is the only way to vary the three groups
*independently* (they share token positions), which is exactly what an importance study requires.
**Real pruning** would drop the base token entirely (shorter sequence, real latency savings, but
renormalized attention and shifted positions, and not separable per group). **All accuracy results in
this paper use zeroing**; we compare methods at an *equal count of retained (non-zeroed) refinement rows*.
Converting the measured redundancy into wall-clock latency via real sequence-shortening is future work
(Section 7).

**Why token count, not the injection op, is the lever.** The extraction+injection operations themselves
cost ≈5 ms total — negligible against the ~10.6 s decoder time. Any efficiency win must come from reducing
the number of tokens the decoder processes, consistent with the profiling above.

---

## 4. Experimental Setup

**Model.** Qwen3-VL-2B-Instruct, fp16, single GPU. We use a local copy of the model source so importance
hooks and pruning operate on the true computation path.

**Tasks and metrics.** Three VQA tasks spanning a difficulty gradient: **VQAv2** (general VQA, soft VQA
accuracy), **TextVQA** (scene-text reading, soft VQA accuracy), **DocVQA** (document understanding, ANLS).
Phase 3 also includes a counting task. Alongside task accuracy we report **first-token KL** —
`KL(P_full ‖ P_condition)` of the first generated token's distribution — as a dense, label-free measure of
how far a pruning condition shifts the model's output. KL is sensitive but is structurally biased toward
magnitude-based selection (it rewards keeping high-norm features that minimize the hidden-state
perturbation), so **we treat accuracy as the arbiter and use KL only as corroboration.**

**Within-group token scorers.** `random` (control), `spatial_uniform` (even grid subsample),
`activation_magnitude` (top-k by per-token L2 norm), `diversity` (farthest-point sampling),
`hybrid` (magnitude-seeded greedy maximizing a magnitude+diversity objective), and `vision_attention`
(per-patch vision-encoder attention received, the CLS-free VisPruner/FasterVLM adaptation, read at the
DeepStack source layers 5/11/17). All return exactly k = round(keep_ratio·N) unique indices so methods are
compared at identical retained-token counts.

**Allocation strategies (the budgeting question).** Given a target *average* keep-ratio T, we compare:
**Uniform** `(T,T,T)`; **Per-group** budgets `(r0,r1,r2)` chosen by a separable "water-filling" search over
the measured per-group sensitivity curves; and a flat **Global top-k** that scores all groups' tokens
together and keeps the global top-k (ignoring group boundaries). **All three are compared at an equal
retained-token count** — the fairness unit that ensures any difference reflects *where* tokens are kept,
not *how many*.

**Calibration vs. held-out.** Per-group sensitivity curves and budgets are estimated on a calibration
split (n=100/task). The decisive head-to-head comparison is run on a **disjoint held-out split**
(n=300/task), with **bootstrap 95% confidence intervals** on accuracy. We pin the within-group scorer to
`hybrid` for the head-to-head (justified in Section 5.3); we explicitly disable accuracy-based auto-
selection of the scorer because at n=100 accuracy cannot rank scorers reliably (Section 5.3).

---

## 5. Results

The experiments form a single arc: *where* redundancy lives (5.1) → whether single groups are dispensable
(5.2) → *which* tokens to keep (5.3) → how far we can prune each group in isolation (5.4) → and finally
whether depth-aware allocation beats uniform on held-out data (5.5).

### 5.1 Where the redundancy lives: per-group feature dispersion

We first measure the per-token feature distribution within each group over a set of real images.
All three groups carry the **same token count** (injection uses the same placeholder positions at every
depth), so the question is token *value*, not token *count*. Using the scale-free coefficient of variation
(CV = std/mean) of per-token L2-norm — deliberately not absolute norm, which grows with depth as a
residual-stream artifact — we find a clear **dispersion gradient**:

| Group (ViT layer) | CV of token norm | max ÷ median skew | % tokens below overall-median norm |
|---|---|---|---|
| G0 (L5, shallow) | **0.61** | 8.9× | 68% |
| G1 (L11, mid)    | 0.45 | 4.3× | 51% |
| G2 (L17, deep)   | **0.42** | 2.9× | 22% |

The shallow group is highly unequal (a crowd of low-norm tokens plus a few dominant outliers) and the
representation becomes denser/more uniform with depth. Under the intuition "dispersed ⇒ much removable
redundancy; dense ⇒ fragile," this *motivated* a non-uniform, depth-aware budget. **(This motivation is
later overturned in 5.5.)** Separately, **decoder attention is an informative null**: the attention mass
received per visual token is ≈0.001–0.0016 and near-identical across groups — essentially the 1/sequence-
length floor — reproducing VisPruner's observation that language-model attention is a poor pruning signal.

*Figure 1: per-group token-norm distributions (`results/20260603_080941/figures/01_norm_distribution.png`).*

### 5.2 Are single depth-groups dispensable? (ablation, in isolation)

We zero one group at a time (others full) and measure accuracy and first-token KL across tasks (n=100).

| Task | full | drop G0 | drop G1 | drop G2 | keep G0 only | keep G1 only | keep G2 only | drop all |
|---|---|---|---|---|---|---|---|---|
| VQAv2 (general) | .833 | .820 | .840 | .833 | .830 | .833 | .800 | .837 |
| TextVQA | .840 | **.853** | .843 | **.800** | .800 | .817 | .823 | .807 |
| DocVQA | .890 | .900 | .900 | .898 | .866 | .900 | .898 | .837 |
| Counting | .820 | .820 | .830 | .820 | .840 | .830 | .830 | .780 |

*Pooled first-token KL: drop_g0 .038, drop_g1 .020, drop_g2 .028; keep_g0 .089, keep_g1 .112, keep_g2 .068;
drop_all .193.*

Three reads. **(i)** The deep group is OCR/text-specialized: on TextVQA, dropping G2 costs **−4.0%** (the
largest single-group effect anywhere), while G0 and G1 are interchangeable there. **(ii)** The shallow
group is the most expendable and even mildly *harmful* on detail tasks (dropping G0 *improves* TextVQA by
+1.3% and DocVQA by +1.0%), consistent with G0's high dispersion. **(iii)** The aggregate matters but
individual groups are largely redundant: `drop_all` hurts every task (−3% to −5%), yet dropping any single
group costs ≤1.3% on three of four tasks. These isolation effects again *suggested* exploitable per-group
structure.

*Figure 2: per-group × task sensitivity heatmap (`results/20260603_184741/figures/sensitivity_heatmap.png`).*

### 5.3 Which tokens to keep: attention fails, feature signals win

Holding the budget uniform, we compare scorers at keep-ratios 0.50 and 0.25 (n=300/task). Values are
accuracy change vs. the full model (positive = worse):

| Task (full) | keep | random | magnitude | hybrid | vision_attn |
|---|---|---|---|---|---|
| VQAv2 (.818) | 0.50 | −.004 | −.013 | −.008 | −.004 |
|  | 0.25 | −.004 | −.007 | **−.002** | −.002 |
| TextVQA (.824) | 0.50 | +.020 | +.012 | +.009 | **+.002** |
|  | 0.25 | +.017 | **+.010** | +.012 | +.031 |
| DocVQA (.892) | 0.50 | +.010 | +.007 | **+.0003** | +.018 |
|  | 0.25 | +.060 | +.021 | **+.020** | +.031 |

**Random clearly loses** (DocVQA@0.25 collapses by +6.0%) — token selection genuinely matters.
**Vision-encoder attention does not win**: it is competitive only at mild pruning on text tasks and decays
toward random at aggressive pruning; on the cleanest discriminating cell (DocVQA@0.25) the ordering is
**hybrid ≈ magnitude > vision_attention > random**. We attribute this to (a) reading attention at
*intermediate* DeepStack source layers (5/11/17) rather than the final encoder layer used by
VisPruner/FasterVLM, and (b) Qwen3-VL's ViT being **CLS-free**, exposing attention-received to the
attention-sink problem. **`hybrid` is the most robust scorer and never has a losing cell; `activation_
magnitude` is near-tied and parameter-free.** Notably, an accuracy-ranked auto-selector picks a *different*
"best" scorer per task (hybrid/random/vision_attention) — direct evidence that accuracy cannot rank
scorers at this sample size. We therefore fix the scorer to **hybrid** and report the magnitude≈hybrid tie
as a *robustness* property: any feature-based scorer works; attention-based ones do not.

*Figure 3: scorer accuracy vs. keep-ratio per task (`results/20260604_001256/figures/scoring_accuracy_curves.png`).*

### 5.4 How far can each group be pruned in isolation? (per-group sweep)

We sweep each group's keep-ratio from 100%→0% (others full), all scorers, n=100/task. Two outcomes.
**Validity:** the keep-ratio=0 endpoint of each curve reproduces the Section 5.2 ablation **exactly** for
all nine (task × group) cells (e.g. TextVQA drop-G2 = −4.0% in both, independently coded experiments) —
the pipeline is correct and the intermediate curve values are trustworthy. **Redundancy:** dropping a
whole group is nearly free except for TextVQA-G2:

| Task | drop G0 | drop G1 | drop G2 |
|---|---|---|---|
| VQAv2 (general) | −1.3% | +0.7% | 0.0% |
| DocVQA | +1.0% | +1.0% | +0.9% |
| TextVQA | +1.3% | +0.3% | **−4.0%** |

On VQAv2 and DocVQA an entire injected depth-group (~33% of the injected refinement) is removable at ≤1%.
The clean per-group structure (protect G2, starve G0) appears only on TextVQA; DocVQA's groups are
*interchangeable*. This sharpened the question for the decisive test below: per-group budgeting's only
clear opening is TextVQA, and only at aggressive total budgets.

### 5.5 Does depth-aware allocation beat uniform? — No (held-out head-to-head)

The decisive experiment: water-filled **per-group** budgets vs. **uniform** vs. **global top-k**, all at
**equal retained-token count**, on the **held-out** split (n=300, scorer = hybrid, bootstrap 95% CIs).
Values are accuracy change vs. full (%); entries are *uniform / best-per-group / global*:

| Task (base) | keep 50% | keep 30% | keep 20% | keep 15% |
|---|---|---|---|---|
| VQAv2 (.818) | +0.3 / −0.3 / +0.8 | +0.4 / −0.4 / +0.4 | +0.3 / +0.1 / +0.1 | −0.2 / +0.1 / +0.2 |
| TextVQA (.822) | −0.2 / −2.1 / −2.3 | −0.3 / −2.1 / −2.1 | −2.0 / −1.7 / −1.7 | −2.9 / −2.1 / −3.0 |
| DocVQA (.884) | +0.2 / +0.1 / −1.2 | −1.8 / −2.2 / −2.7 | −2.6 / −2.8 / −3.2 | −2.5 / −2.6 / −2.6 |

**Per-group budgeting is equal-or-worse than uniform in every cell.** The only nominal per-group "wins"
(TextVQA/VQAv2 at the most aggressive budgets) are ≤+0.8% with fully overlapping 95% CIs — noise.
**Global top-k also merely ties uniform.** So neither respecting depth-group boundaries (per-group) nor
ignoring them entirely (global) beats the simplest allocation. First-token KL is consistently *higher*
for per-group than uniform (zeroing whole groups perturbs the residual stream more), corroborating that
uniform is the gentler allocation.

**Mechanism.** The Section 5.4 isolation effects were real but did **not transfer** to joint pruning. For
example, TextVQA's water-filled 50%-budget zeros G0 entirely, `(0, 0.8, 0.7)`, scoring −2.1%, whereas
uniform `(0.5, 0.5, 0.5)` scores −0.2%. Dropping G0 was "free" only while G1/G2 were full; once all groups
are pruned together, the groups' **mutual redundancy** means keeping a fraction of *every* group (with a
good scorer selecting each group's most useful tokens) preserves the cross-group-overlapping information,
while concentrating cuts in one group throws away its unique coverage for no compensating gain. **Uniform
allocation plus feature-based selection is already near-optimal.**

*Figure 4: per-group vs. uniform vs. global at equal token count, per task
(`results/20260604_232301/figures/{textvqa,general_vqa,docvqa}/budgeting_validation.png`).*

### 5.6 Compressibility summary (the positive headline)

Because selection is what matters and the injected mass is redundant, **feature-based uniform pruning is
highly compressible** at equal retained-token count on held-out data:

| Task | uniform keep-ratio at ≤~2% accuracy change | injected refinement removed |
|---|---|---|
| VQAv2 (general) | keep 15% → −0.2% | **~85%** |
| TextVQA | keep 30% → −0.3% (keep 20% → −2.0%) | **~70%** |
| DocVQA | keep 50% → +0.2% (keep 30% → −1.8%) | **~50%** |

These are diagnostic (zeroing) numbers: they quantify how much DeepStack-injected refinement is
removable without hurting accuracy, not a measured latency reduction.

---

## 6. Discussion

**The honest arc — and why the intuitive method failed.** We began with a well-motivated hypothesis:
DeepStack's depth-groups differ in dispersion (5.1) and in single-group ablation sensitivity (5.2, 5.4),
so a depth-aware per-group budget *should* beat uniform pruning. Every isolation measurement supported it.
The hypothesis failed only under the correct, decisive test — joint pruning on held-out data at equal
token count (5.5) — because **single-group sensitivity is not additive across groups.** The groups are
mutually redundant: their information overlaps enough that you cannot bank the "G0 is free" saving while
also pruning G1 and G2. This is a clean example of why importance measured one-factor-at-a-time can
mislead, and why held-out, equal-budget joint evaluation is the right standard.

**Why attention fails for DeepStack selection.** Decoder attention is a near-uniform null (5.1); vision-
encoder attention, strong at a model's final layer with a `[CLS]` anchor, does not transfer to Qwen3-VL's
CLS-free intermediate source layers, where attention-received is dominated by a few sink patches and
becomes diffuse under aggressive pruning. Feature magnitude (which identifies the functionally important
high-norm/"massive-activation" tokens) plus a diversity term is both simpler and more robust.

**Practical recommendation.** To compress DeepStack visual tokens: use a **uniform** keep-ratio across
depth-groups with a **feature-based scorer** (magnitude, optionally plus diversity); do not compute
attention scores and do not tune per-depth budgets — they add complexity without benefit. Choose the
uniform ratio per task tolerance (e.g. ~15% for coarse VQA, ~30% for scene text, ~50% for documents).

---

## 7. Limitations

- **Diagnostic compression, not measured latency.** All accuracy results use *zeroing* of injected
  refinements at equal retained-token count; they establish redundancy/headroom, not wall-clock speedup.
  The natural follow-up ("Stage B") is real sequence-shortening of the uniform feature-based prune with
  preserved position encodings, and end-to-end latency/memory benchmarking — token cuts do not
  automatically become speed, so this must be measured.
- **Single model / scale.** Results are on Qwen3-VL-2B-Instruct; larger models and other DeepStack-style
  architectures may differ.
- **Sample sizes.** Calibration uses n=100/task and the head-to-head n=300/task; soft VQA accuracy and
  ANLS carry roughly ±1–3% variance at these sizes, so we lead with directional findings and bootstrap CIs
  and avoid over-interpreting sub-1% differences.
- **Task coverage.** Three VQA tasks (plus counting in ablation); captioning, chart/table, spatial, and
  video tasks are untested.
- **Metric.** First-token KL is structurally biased toward magnitude selection; we use it only to
  corroborate accuracy, never as the primary arbiter.

---

## 8. Conclusion

We asked whether DeepStack's depth-structured visual injection can be exploited for smarter token
compression. The answer is a clear and useful "mostly no, with one strong yes": **(1)** DeepStack's
injected visual refinement is highly redundant — feature-based **uniform** pruning removes 50–85% of it at
≤~2% accuracy; **(2)** the right way to choose which tokens to keep is **feature-based, not attention-
based**; and **(3)** **depth-aware per-group budgeting does not beat uniform allocation**, because the
depth-groups are mutually redundant. The deployable recipe is simple and robust. Measuring the real
latency/memory Pareto of the uniform feature-based prune is the clear next step.

---

## References

*Bibliographic details to be finalized; entries below capture the works cited and their relevant claims.*

1. VisPruner — *Vision-encoder attention for visual-token pruning; LM text–visual attention is not a
   reliable indicator.* ICCV 2025, arXiv:2412.01818. [full citation TBD]
2. FasterVLM — *CLS→patch vision-encoder attention; ~90% performance retained at ~95% token pruning.*
   [full citation TBD]
3. LaCo — *Compresses visual tokens within intermediate vision-encoder layers; >15% throughput.*
   [full citation TBD]
4. HiPrune — *Middle ViT layers capture object-centric features; deeper layers encode global context.*
   [full citation TBD]
5. ST3 — *Progressive visual-token pruning across decoder layers and decoding steps; ~2× inference, ~30%
   KV-cache vs. LLaVA.* [full citation TBD]
6. VL-Cache — *Modality-aware KV-cache compression; retains ~10% of KV cache, up to 2.33× latency.*
   [full citation TBD]
7. CoViPAL — *Layer-wise contextualized visual-token pruning before LVLM processing.* [full citation TBD]
8. DeepStack — *Depth-aligned injection of intermediate visual features into aligned LLM depths; strong
   performance at ~1/5 context length.* [full citation TBD]

---

## Appendix A. Reproducibility

All experiments run on Qwen3-VL-2B-Instruct (fp16) via a controlled harness; the model source is used
locally so importance hooks and pruning act on the true computation path. Raw outputs (JSON + figures) per
phase:

| Section | Phase | Results directory |
|---|---|---|
| 3 (mechanism/profiling) | Phase 1 probe | `results/20260603_050848/` |
| 5.1 (dispersion) | Phase 2 instrument | `results/20260603_080941/` |
| 5.2 (ablation) | Phase 3 | `results/20260603_184741/` |
| 5.3 (scorers) | Phase 4 / 4b | `results/20260603_204751/`, `results/20260604_001256/` |
| 5.4 (per-group sweep) | Phase 5 sweep | `results/20260604_232301/` |
| 5.5 (head-to-head) | Phase 5 validate | `results/20260605_050905/` |

Comparisons use an equal-retained-token-count fairness unit; held-out evaluation (Section 5.5) uses a
split disjoint from calibration with bootstrap 95% confidence intervals. Code, configuration, and the
one-click GPU runner are in the project repository.
