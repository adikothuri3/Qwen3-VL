"""
Vision-encoder (ViT) attention saliency capture for DeepStack pruning (Phase 4b).

Implements the literature's strong, training-free token-importance signal — the
*vision-encoder* attention, not the LM/decoder attention (VisPruner, ICCV 2025,
arXiv:2412.01818; FasterVLM). VisPruner's canonical signal is [CLS]→patch
attention, but Qwen3-VL's ViT has **no [CLS] token** (pure patch tokens:
patch_embed → pos_embed → blocks → merger). VisPruner specifies the exact
CLS-free recipe for that case: "average the rows of A as the average attention
each patch token receives." That per-patch *attention-received* vector is what
this module captures, at the DeepStack source ViT layers (`deepstack_visual_indexes`).

Mechanism (edits no model source):
  - Qwen3VLVisionAttention.forward returns only attn_output (weights discarded),
    but it calls the module-global `eager_attention_forward`, resolved as a global
    at call time. We temporarily wrap that global so that, for calls whose `module`
    is one of the target `blocks[idx].attn` instances (identity-matched), we stash
    the per-key attention-received vector. No second matmul — we reduce the weights
    the model already computes.
  - Eager attention is forced on the vision config only (so attn_weights exist);
    the text tower is untouched. Restored on exit.

Pre-merge → post-merge mapping: the DeepStack feature is the PatchMerger output,
which does `x.view(-1, hidden * spatial_merge_size**2)`, so the 4 pre-merge patches
of merged token m are contiguous. Hence per-merged-token saliency =
`received.reshape(N, 4).mean(1)`. A runtime guard asserts received.numel() == 4*N.

Vision saliency is image-only (independent of the pruning condition) and is
produced before injection, so it is captured once during a single model forward
and reused across keep-ratios.
"""

from typing import Any, Dict, List, Optional

import torch


def _find_vision_model(model: Any) -> Optional[Any]:
    for module in model.modules():
        if type(module).__name__ == "Qwen3VLVisionModel":
            return module
    return None


class VisionAttentionCapturer:
    """Context manager that captures per-group ViT attention-received saliency.

    Usage:
        with VisionAttentionCapturer(model) as cap:
            model.generate(**inputs, ...)        # one forward (prefill) populates it
        scores = cap.collect(num_tokens_per_group)   # {group_idx: Tensor(N,)} or None
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.vision_model = _find_vision_model(model)
        indexes: List[int] = list(model.config.vision_config.deepstack_visual_indexes)
        self.source_layers = indexes
        # id(attn module) -> group index (position in deepstack_visual_indexes)
        self._target_ids: Dict[int, int] = {}
        if self.vision_model is not None:
            for gi, layer in enumerate(indexes):
                attn = self.vision_model.blocks[layer].attn
                self._target_ids[id(attn)] = gi
        # group index -> list of per-chunk received vectors (concatenated at collect)
        self._received: Dict[int, List[torch.Tensor]] = {gi: [] for gi in range(len(indexes))}
        self._orig_eager: Any = None
        self._orig_vision_impl: Optional[str] = None
        self._mod: Any = None

    # ── wrapping ────────────────────────────────────────────────────────────────

    def _make_wrapper(self) -> Any:
        orig = self._orig_eager
        targets = self._target_ids
        store = self._received

        def wrapper(module: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            out = orig(module, query, key, value, *args, **kwargs)
            gi = targets.get(id(module))
            if gi is not None:
                attn_weights = out[1]
                if attn_weights is not None:
                    # (batch, heads, q, k) -> per-key received = mean over batch, heads, queries
                    received = attn_weights.float().mean(dim=(0, 1, 2)).detach()
                    store[gi].append(received)
            return out

        return wrapper

    def __enter__(self) -> "VisionAttentionCapturer":
        from local_transformers.models.qwen3_vl import modeling_qwen3_vl as _mod

        self._mod = _mod
        self._orig_eager = _mod.eager_attention_forward
        _mod.eager_attention_forward = self._make_wrapper()

        vcfg = self.model.config.vision_config
        self._orig_vision_impl = getattr(vcfg, "_attn_implementation", None)
        vcfg._attn_implementation = "eager"
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._mod is not None and self._orig_eager is not None:
            self._mod.eager_attention_forward = self._orig_eager
        vcfg = self.model.config.vision_config
        # Restore the original impl; never leave it None (line 202-203 of the model
        # would do ALL_ATTENTION_FUNCTIONS[None] -> KeyError on the next forward).
        vcfg._attn_implementation = self._orig_vision_impl or "sdpa"
        self._mod = None
        self._orig_eager = None

    # ── readout ─────────────────────────────────────────────────────────────────

    def collect(self, num_tokens_per_group: int) -> Optional[Dict[int, torch.Tensor]]:
        """Per-group merged-token saliency, or None if capture failed/misaligned.

        ``num_tokens_per_group`` is N (the post-merge token count == a group's
        deepstack-embed row count). Returns {group_idx: Tensor(N,)} on success.
        """
        n = int(num_tokens_per_group)
        scores: Dict[int, torch.Tensor] = {}
        for gi, chunks in self._received.items():
            if not chunks:
                return None
            received = torch.cat(chunks, dim=0).float()  # pre-merge order, length should be 4*N
            if received.numel() != 4 * n:
                print(
                    f"[saliency] group {gi}: received {received.numel()} != 4*N ({4 * n}); "
                    "skipping vision_attention for this sample",
                    flush=True,
                )
                return None
            scores[gi] = received.reshape(n, 4).mean(dim=1)
        return scores
