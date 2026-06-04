"""
Shared generation + metric helpers for the pruning/budgeting experiments.

Refactored out of exp_scoring.py so exp_scoring (Phase 4) and exp_budgeting
(Phase 5) compute the headline signals identically: greedy decode, first-token
KL, the no-pruning baseline with one-shot vision-attention saliency capture, and
the per-group merged-token count. No behavior change vs. the original inlined
versions.

Importing this module performs no model load; the Qwen3VLEvaluator type is only
used as a typing/runtime handle passed in by the caller.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from src.deepstack.prune import DeepStackPruner
from src.deepstack.saliency import VisionAttentionCapturer
from src.evaluate import Qwen3VLEvaluator


def generate(
    evaluator: Qwen3VLEvaluator, inputs: Dict[str, Any], max_new_tokens: int
) -> Tuple[str, torch.Tensor]:
    """Greedy generate; return (decoded answer, first-token logits as a float CPU vec)."""
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = evaluator.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    seq = out.sequences[0][input_len:]
    text = evaluator.processor.batch_decode(
        [seq], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    first_logits = out.scores[0][0].detach().float().cpu()
    return text, first_logits


def kl_first_token(logits_full: torch.Tensor, logits_cond: torch.Tensor) -> float:
    """KL(P_full || P_cond) over the first-token distribution."""
    logp_full = torch.log_softmax(logits_full, dim=-1)
    logp_cond = torch.log_softmax(logits_cond, dim=-1)
    p_full = logp_full.exp()
    return float((p_full * (logp_full - logp_cond)).sum())


def num_visual_tokens(inputs: Dict[str, Any], merge_size: int) -> Optional[int]:
    """Per-group merged-token count N = Σ t·(h/ms)·(w/ms) from image_grid_thw."""
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    return int((grid.prod(dim=-1) // (merge_size * merge_size)).sum().item())


def baseline_with_capture(
    evaluator: Qwen3VLEvaluator,
    pruner: DeepStackPruner,
    inputs: Dict[str, Any],
    max_new_tokens: int,
    n_tokens: Optional[int],
    need_vision: bool,
) -> Tuple[str, torch.Tensor, Optional[Dict[int, torch.Tensor]]]:
    """Run the no-pruning baseline; if ``need_vision``, also capture per-group
    vision attention-received saliency from that forward (eager). Returns
    (answer, first-token logits, vision_scores-or-None)."""
    pruner.set_specs({})  # baseline: no pruning
    pruner.set_keep_indices(None)
    if need_vision and n_tokens is not None:
        try:
            with VisionAttentionCapturer(evaluator.model) as cap:
                ans, logits = generate(evaluator, inputs, max_new_tokens)
            return ans, logits, cap.collect(n_tokens)
        except Exception as e:  # noqa: BLE001 — capture is best-effort; fall back to plain baseline
            print(f"  [vision capture failed: {type(e).__name__}: {e}]", flush=True)
            if evaluator.device == "cuda":
                torch.cuda.empty_cache()
    ans, logits = generate(evaluator, inputs, max_new_tokens)
    return ans, logits, None
