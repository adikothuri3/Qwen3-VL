"""
DeepStack internals probe (Phase 1).

Non-invasive forward-hook probe that maps how Qwen3-VL's DeepStack mechanism
works at runtime, without editing the model source. It captures, for one real
image+text inference:

  - the number of DeepStack groups and the vision-encoder layers they tap
  - per-group feature tensor shape / dtype / device / per-token L2-norm stats
  - the decoder depths at which each group is injected
  - the "count contract": deepstack_visual_embeds[i].shape[0] must equal the
    number of visual positions (visual_pos_masks.sum()) for the injection add
    at Qwen3VLTextModel._deepstack_process to succeed
  - a cross-check of the visual-token count against image_grid_thw
  - a mutation test proving that naively dropping tokens from a group breaks the
    injection add (shape mismatch), while reconstructing to full length (scatter
    kept tokens, zero-fill dropped) keeps the count valid

Output: results/<timestamp>/deepstack_probe.json

This is read/measure only — it changes no model logic. It exists to de-risk the
pruning design in later phases (see paper.md section 13, Phase 1).
"""

import argparse
import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Reuse the evaluator's local_transformers wiring, model+processor loading, and
# default image+text test case. Importing evaluate performs the sys.modules
# redirect to local_transformers before any transformers import.
from src.evaluate import Qwen3VLEvaluator, create_default_test_case


@dataclass
class GroupReport:
    """Per-DeepStack-group facts captured at runtime."""

    group_index: int
    vision_layer: int
    injection_decoder_layer: int
    num_tokens: int
    hidden_dim: int
    dtype: str
    device: str
    token_norm_mean: float
    token_norm_std: float
    token_norm_min: float
    token_norm_max: float
    count_match: bool


@dataclass
class ProbeReport:
    """Full Phase 1 internals map for one inference."""

    model_source: str
    num_groups: int
    vision_layers: List[int]
    injection_layers: List[int]
    out_hidden_size: int
    spatial_merge_size: int
    num_visual_positions: int
    image_grid_thw: List[List[int]]
    expected_visual_tokens: int
    visual_token_count_matches_grid: bool
    groups: List[GroupReport] = field(default_factory=list)
    mutation_test: Dict[str, str] = field(default_factory=dict)


def _norm_stats(feat: torch.Tensor) -> Dict[str, float]:
    """Per-token L2-norm summary stats for a (tokens, dim) feature tensor."""
    norms = feat.detach().float().norm(dim=-1)
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()) if norms.numel() > 1 else 0.0,
        "min": float(norms.min()),
        "max": float(norms.max()),
    }


class DeepStackProbe:
    """Attaches forward hooks to capture the DeepStack feature flow."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._handles: List[Any] = []
        # Captured state, filled in by hooks during the forward pass.
        self.deepstack_feature_lists: Optional[List[torch.Tensor]] = None
        self.visual_pos_masks: Optional[torch.Tensor] = None
        self.deepstack_visual_embeds: Optional[List[torch.Tensor]] = None

    def _register(self) -> None:
        for module in self.model.modules():
            name = type(module).__name__
            if name == "Qwen3VLVisionModel":
                self._handles.append(module.register_forward_hook(self._vision_hook))
            elif name == "Qwen3VLTextModel":
                self._handles.append(module.register_forward_pre_hook(self._text_pre_hook, with_kwargs=True))

    def _vision_hook(self, module: Any, args: Any, output: Any) -> None:
        # Qwen3VLVisionModel.forward returns (hidden_states, deepstack_feature_lists).
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], list):
            self.deepstack_feature_lists = [f.detach() for f in output[1]]

    def _text_pre_hook(self, module: Any, args: Any, kwargs: Dict[str, Any]) -> None:
        # Capture the DeepStack injection inputs at the point they reach the decoder.
        masks = kwargs.get("visual_pos_masks")
        embeds = kwargs.get("deepstack_visual_embeds")
        if masks is not None and self.visual_pos_masks is None:
            self.visual_pos_masks = masks.detach()
        if embeds is not None and self.deepstack_visual_embeds is None:
            self.deepstack_visual_embeds = [e.detach() for e in embeds]

    def __enter__(self) -> "DeepStackProbe":
        self._register()
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _injection_add(hidden_visual: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
    """Replicates the core of _deepstack_process: add embeds onto the visual rows.

    Raises if the row counts disagree — which is exactly the count contract.
    """
    if hidden_visual.shape[0] != embeds.shape[0]:
        raise ValueError(
            f"count mismatch: {hidden_visual.shape[0]} visual positions vs {embeds.shape[0]} embeds"
        )
    return hidden_visual + embeds


def _run_mutation_test(num_positions: int, embeds: torch.Tensor) -> Dict[str, str]:
    """Prove naive token-drop breaks injection, and reconstruct-to-full fixes it.

    `hidden_visual` stands in for hidden_states[visual_pos_masks, :]: it always has
    `num_positions` rows (fixed by the prompt's image placeholders).
    """
    result: Dict[str, str] = {}
    hidden_visual = torch.zeros(num_positions, embeds.shape[-1], dtype=embeds.dtype)

    # 1) Naive prune: drop k tokens from the group, then try to inject -> must fail.
    k = max(1, num_positions // 4)
    pruned = embeds[: num_positions - k]
    try:
        _injection_add(hidden_visual, pruned)
        result["naive_drop"] = "UNEXPECTED PASS — injection accepted a shorter group"
    except ValueError as e:
        result["naive_drop"] = f"raised ValueError as expected ({e})"

    # 2) Reconstruct-to-full: keep the same pruned tokens but scatter them back to
    #    full length with zero-fill for dropped rows -> count matches, add succeeds.
    keep_idx = torch.arange(num_positions - k)
    reconstructed = embeds.new_zeros(num_positions, embeds.shape[-1])
    reconstructed[keep_idx] = pruned
    try:
        _injection_add(hidden_visual, reconstructed)
        result["reconstruct"] = "PASS — zero-filled reconstruction keeps the count valid"
    except ValueError as e:
        result["reconstruct"] = f"UNEXPECTED FAIL ({e})"

    return result


def probe(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    output_dir: str = "results",
    max_new_tokens: int = 8,
) -> ProbeReport:
    evaluator = Qwen3VLEvaluator(model_id=model_id, device=device, dtype=dtype)
    model = evaluator.model
    processor = evaluator.processor

    vision_cfg = model.config.vision_config
    vision_layers: List[int] = list(vision_cfg.deepstack_visual_indexes)
    out_hidden_size = int(vision_cfg.out_hidden_size)
    spatial_merge_size = int(vision_cfg.spatial_merge_size)

    messages = create_default_test_case()["messages"]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(evaluator.device)

    grid = inputs["image_grid_thw"]  # (num_images, 3) = (t, h, w) in patch units
    expected_visual_tokens = 0
    for t, h, w in grid.tolist():
        expected_visual_tokens += t * (h // spatial_merge_size) * (w // spatial_merge_size)

    with DeepStackProbe(model) as p:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if p.deepstack_feature_lists is None or p.deepstack_visual_embeds is None or p.visual_pos_masks is None:
        raise RuntimeError("Probe captured no DeepStack tensors — hooks did not fire as expected.")

    num_positions = int(p.visual_pos_masks.sum().item())

    groups: List[GroupReport] = []
    for i, embeds in enumerate(p.deepstack_visual_embeds):
        stats = _norm_stats(embeds)
        groups.append(
            GroupReport(
                group_index=i,
                vision_layer=vision_layers[i],
                injection_decoder_layer=i,  # injected after decoder layer i (layer_idx in range(len))
                num_tokens=int(embeds.shape[0]),
                hidden_dim=int(embeds.shape[-1]),
                dtype=str(embeds.dtype),
                device=str(embeds.device),
                token_norm_mean=stats["mean"],
                token_norm_std=stats["std"],
                token_norm_min=stats["min"],
                token_norm_max=stats["max"],
                count_match=int(embeds.shape[0]) == num_positions,
            )
        )

    mutation = _run_mutation_test(num_positions, p.deepstack_visual_embeds[0].cpu())

    report = ProbeReport(
        model_source=inspect.getfile(type(model)),
        num_groups=len(p.deepstack_visual_embeds),
        vision_layers=vision_layers,
        injection_layers=list(range(len(p.deepstack_visual_embeds))),
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
        num_visual_positions=num_positions,
        image_grid_thw=grid.tolist(),
        expected_visual_tokens=expected_visual_tokens,
        visual_token_count_matches_grid=(expected_visual_tokens == num_positions),
        groups=groups,
        mutation_test=mutation,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "deepstack_probe.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    _print_summary(report, report_path)
    return report


def _print_summary(r: ProbeReport, path: Path) -> None:
    print("\n=== DeepStack Probe Report ===")
    print(f"Model source:          {r.model_source}")
    print(f"Groups:                {r.num_groups}  (vision layers {r.vision_layers})")
    print(f"Injection decoder layers: {r.injection_layers}")
    print(f"Visual positions:      {r.num_visual_positions}")
    print(f"image_grid_thw:        {r.image_grid_thw}")
    print(
        f"Grid cross-check:      expected {r.expected_visual_tokens} tokens -> "
        f"{'MATCH' if r.visual_token_count_matches_grid else 'MISMATCH'}"
    )
    for g in r.groups:
        print(
            f"  group {g.group_index}: vit_layer={g.vision_layer} "
            f"inject@dec{g.injection_decoder_layer} shape=({g.num_tokens},{g.hidden_dim}) "
            f"dtype={g.dtype} norm μ={g.token_norm_mean:.3f}±{g.token_norm_std:.3f} "
            f"count_match={'PASS' if g.count_match else 'FAIL'}"
        )
    print(f"Mutation test:         {r.mutation_test}")
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Qwen3-VL DeepStack internals (Phase 1).")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps", "auto"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "float32", "auto"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    device = None if args.device in (None, "auto") else args.device
    dtype = None if args.dtype in (None, "auto") else args.dtype
    probe(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
