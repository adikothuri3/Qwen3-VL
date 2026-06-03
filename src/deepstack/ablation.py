"""
DeepStack group ablation switches (Phase 3).

Non-invasive mechanism to ablate whole DeepStack visual feature groups at
inference time, used by the group-sensitivity experiment (Experiment 2 in
paper.md). It edits no model source.

How it works (verified against local_transformers in Phase 1):

  - DeepStack injection is a *pure additive* op: after decoder layer ``i``,
    ``hidden_states[visual_pos_masks] += deepstack_visual_embeds[i]``
    (Qwen3VLTextModel._deepstack_process, modeling_qwen3_vl.py).
  - ``deepstack_visual_embeds`` reaches the text model as a forward kwarg
    (Qwen3VLModel.forward -> self.language_model(..., deepstack_visual_embeds=...)).

Therefore zeroing group ``i``'s embeds tensor reproduces "remove group ``i``"
exactly: the additive contribution becomes 0 while the 1:1 count contract and
the MRoPE positions stay valid (nothing about shapes or placeholder counts
changes). Injection only fires on the prefill forward — decode steps carry no
visual embeds — so the hook naturally affects only prefill.

The 8 standard conditions (paper.md Experiment 2): full, drop_g{0,1,2},
keep_g{0,1,2}, drop_all (the latter == the paper's "No DeepStack" baseline).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

import torch


@dataclass(frozen=True)
class AblationCondition:
    """One ablation setting: which DeepStack groups to zero out before injection."""

    name: str
    disabled_groups: FrozenSet[int] = field(default_factory=frozenset)

    @property
    def is_full(self) -> bool:
        return len(self.disabled_groups) == 0


def standard_conditions(num_groups: int) -> List[AblationCondition]:
    """Build the 8 conditions of paper.md Experiment 2 for ``num_groups`` groups.

    full, drop_g{i}, keep_g{i} for each i, and drop_all. ``keep_g{i}`` disables
    every group except ``i``; ``drop_all`` disables all groups (== No DeepStack).
    """
    all_groups = set(range(num_groups))
    conditions: List[AblationCondition] = [AblationCondition("full", frozenset())]
    for i in range(num_groups):
        conditions.append(AblationCondition(f"drop_g{i}", frozenset({i})))
    for i in range(num_groups):
        conditions.append(AblationCondition(f"keep_g{i}", frozenset(all_groups - {i})))
    conditions.append(AblationCondition("drop_all", frozenset(all_groups)))
    return conditions


class DeepStackAblator:
    """Context manager that zeros selected DeepStack groups via a forward hook.

    Lifecycle mirrors DeepStackProbe: enter to register the hook, set the active
    condition between forwards with ``set_condition``, exit to remove the hook.
    The active condition persists across forwards until changed, so ``generate``
    (prefill + many decode steps) and a separate logits forward all see it.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._handles: List[Any] = []
        self._condition: AblationCondition = AblationCondition("full", frozenset())
        self.num_groups = int(len(model.config.vision_config.deepstack_visual_indexes))

    def set_condition(self, condition: AblationCondition) -> None:
        self._condition = condition

    def _register(self) -> None:
        for module in self.model.modules():
            if type(module).__name__ == "Qwen3VLTextModel":
                self._handles.append(
                    module.register_forward_pre_hook(self._text_pre_hook, with_kwargs=True)
                )

    def _text_pre_hook(self, _module: Any, _args: Any, kwargs: Dict[str, Any]) -> None:
        if self._condition.is_full:
            return
        embeds: Optional[List[torch.Tensor]] = kwargs.get("deepstack_visual_embeds")
        if not embeds:  # None (decode step) or empty -> nothing to ablate
            return
        for gi in self._condition.disabled_groups:
            if 0 <= gi < len(embeds):
                # In-place: embeds[gi] is freshly built each prefill forward, so
                # zeroing it here does not corrupt any persistent state and is
                # seen by _deepstack_process without needing to return kwargs.
                embeds[gi] = torch.zeros_like(embeds[gi])

    def __enter__(self) -> "DeepStackAblator":
        self._register()
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
