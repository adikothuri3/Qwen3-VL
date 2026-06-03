"""
DeepStack within-group visual-token pruning (Phase 4).

Implements the four scoring strategies of paper.md section 6 (Level 2) and the
*reconstruct-to-full-length* contract proven necessary in Phase 1.

Why reconstruct-to-full-length (the hard constraint from Phase 1):
    DeepStack injection is a strict 1:1 additive op
        hidden_states[visual_pos_masks] += deepstack_visual_embeds[i]
    so ``deepstack_visual_embeds[i].shape[0]`` MUST equal ``visual_pos_masks.sum()``
    (fixed by the prompt's image-placeholder tokens; MRoPE positions are computed
    once and never touched by injection). Naively dropping rows crashes the add
    (confirmed by the Phase 1 probe mutation test). So "pruning" here means: score
    all N tokens, choose k to keep, and **zero out the (N-k) pruned rows in place**
    while keeping the tensor at shape (N, D). A zeroed row contributes 0 to the
    additive injection — exactly as if that token's feature were never injected —
    while the count contract and positions stay valid.

Scorers (paper.md section 6, Level 2):
    random               control — uniform random keep
    spatial_uniform      preserves spatial coverage (even subsample on the merged
                         token grid; falls back to flat stride for multi-image)
    activation_magnitude keep the highest per-token L2 norm (strong feature
                         activity; note Phase 1/2: G0 has outlier sink tokens that
                         hijack this — the diversity term exists to counter that)
    diversity            farthest-point sampling in a random low-d projection —
                         keeps a feature-space-spread subset, avoids near-duplicates
    hybrid               magnitude-seeded, magnitude+diversity-weighted greedy
                         selection (paper.md: saliency + magnitude + diversity_bonus)

All scorers return exactly ``k = round(keep_ratio * N)`` kept indices so methods
are compared at an identical retained-token count (the fairness requirement of
paper.md Experiment 3/4).

DeepStackPruner is a non-invasive context manager (mirrors DeepStackAblator): it
hooks the text model's forward to prune each group's embeds before injection, and
the outer model's forward to capture ``image_grid_thw`` for the spatial scorer.
It edits no model source.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

# Methods that need only the embeds tensor (no grid) — kept here so callers can
# validate a method name without instantiating anything.
SCORING_METHODS = (
    "random",
    "spatial_uniform",
    "activation_magnitude",
    "diversity",
    "hybrid",
)

_PROJ_DIM = 32  # random-projection dim for diversity / hybrid greedy selection
_HYBRID_ALPHA = 0.5  # magnitude weight in the hybrid objective
_HYBRID_BETA = 0.5  # diversity (min-distance) weight in the hybrid objective


# ═══════════════════════════════════════════════════════════════════════════════
#  Index selection (each returns exactly k indices to KEEP, as a LongTensor)
# ═══════════════════════════════════════════════════════════════════════════════


def _keep_count(n: int, keep_ratio: float) -> int:
    """k = round(keep_ratio * n), clamped to [0, n]. keep_ratio>=1 keeps all."""
    if keep_ratio >= 1.0:
        return n
    if keep_ratio <= 0.0:
        return 0
    return max(0, min(n, int(round(keep_ratio * n))))


def _random_indices(n: int, k: int, device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
    # _generator() builds the generator on `device`, so device and generator.device agree.
    perm = torch.randperm(n, generator=generator, device=device)
    return perm[:k]


def _magnitude_indices(embeds: torch.Tensor, k: int) -> torch.Tensor:
    norms = embeds.norm(dim=-1)
    return torch.topk(norms, k, largest=True, sorted=False).indices


def _resize_to_k(idx: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Force an index set to exactly k unique indices in [0, n).

    Trims extras (evenly) or fills the shortfall with as-yet-unused indices in
    order. Used by the spatial scorer, whose grid math lands *near* k.
    """
    idx = torch.unique(idx)  # sorted, deduped
    m = idx.numel()
    if m == k:
        return idx
    if m > k:
        # k evenly-spaced, guaranteed-distinct positions: floor(i*m/k) is strictly
        # increasing when m >= k, so this yields exactly k unique kept indices.
        pos = (torch.arange(k, device=idx.device) * m) // k
        return idx[pos]
    # shortfall: append unused indices in ascending order
    used = torch.zeros(n, dtype=torch.bool, device=idx.device)
    used[idx] = True
    extra = torch.nonzero(~used, as_tuple=False).flatten()
    need = k - m
    return torch.cat([idx, extra[:need]])


def _spatial_uniform_indices(
    n: int, k: int, device: torch.device, grid_dims: Optional[Sequence[Sequence[int]]]
) -> torch.Tensor:
    """Evenly spaced keep set.

    Single image with a known merged grid (T, Hm, Wm): subsample rows and columns
    on an even lattice (true 2D spatial coverage), then resize to exactly k.
    Otherwise (no grid / multi-image): even stride on the flat raster sequence.
    """
    if grid_dims is not None and len(grid_dims) == 1:
        t, hm, wm = (int(x) for x in grid_dims[0])
        if t * hm * wm == n and hm > 0 and wm > 0:
            frac = (k / n) ** 0.5
            keep_h = max(1, min(hm, int(round(hm * frac))))
            keep_w = max(1, min(wm, int(round(wm * frac))))
            rows = torch.linspace(0, hm - 1, keep_h).round().long()
            cols = torch.linspace(0, wm - 1, keep_w).round().long()
            rows = torch.unique(rows)
            cols = torch.unique(cols)
            block = (rows.view(-1, 1) * wm + cols.view(1, -1)).flatten()
            frames = torch.arange(t).view(-1, 1) * (hm * wm)
            idx = (frames + block.view(1, -1)).flatten().to(device)
            return _resize_to_k(idx, k, n)
    # flat fallback
    idx = torch.linspace(0, n - 1, k).round().long().to(device)
    return _resize_to_k(idx, k, n)


def _greedy_indices(embeds: torch.Tensor, k: int, alpha: float, beta: float) -> torch.Tensor:
    """Magnitude-seeded greedy selection in a random projection.

    Objective at each step: pick argmax(alpha * norm_z + beta * min_dist), where
    norm_z is the standardized per-token norm and min_dist is the distance to the
    nearest already-selected token in a fixed 32-d random projection of the
    L2-normalized features.

    - diversity:  alpha=0, beta=1  -> pure farthest-point sampling
    - hybrid:     alpha=beta=0.5   -> balance strong + spread (paper.md hybrid)

    O(n * k * proj_dim); for n=2752, k<=2064, proj=32 this is ~tens of ms on a T4.
    The projection matrix is seeded by the hidden size, so it is deterministic and
    identical across runs/groups of the same width.
    """
    n, d = embeds.shape
    if k <= 0:
        return embeds.new_empty(0, dtype=torch.long)
    if k >= n:
        return torch.arange(n, device=embeds.device)

    feats = embeds.float()
    norms = feats.norm(dim=-1)
    # standardize norm into a comparable scale with the [0,1]-ish distance term
    norm_z = (norms - norms.mean()) / (norms.std() + 1e-6)

    gen = torch.Generator(device="cpu").manual_seed(d)
    proj = torch.randn(d, _PROJ_DIM, generator=gen).to(feats.device, feats.dtype)
    pts = torch.nn.functional.normalize(feats, dim=-1) @ proj  # (n, proj_dim)

    # Keep indices as 0-dim device tensors throughout to avoid a host sync every
    # step (the loop runs up to k times on the GPU during the experiment).
    selected = torch.empty(k, dtype=torch.long, device=embeds.device)
    first = torch.argmax(norms)
    selected[0] = first
    min_dist = (pts - pts[first]).norm(dim=-1)  # (n,)
    chosen = torch.zeros(n, dtype=torch.bool, device=embeds.device)
    chosen[first] = True

    for step in range(1, k):
        score = alpha * norm_z + beta * min_dist
        score = score.masked_fill(chosen, float("-inf"))
        nxt = torch.argmax(score)
        selected[step] = nxt
        chosen[nxt] = True
        d_new = (pts - pts[nxt]).norm(dim=-1)
        min_dist = torch.minimum(min_dist, d_new)
    return selected


def select_keep_indices(
    embeds: torch.Tensor,
    keep_ratio: float,
    method: str,
    grid_dims: Optional[Sequence[Sequence[int]]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return the indices (into rows of ``embeds``) to KEEP for one group."""
    n = embeds.shape[0]
    k = _keep_count(n, keep_ratio)
    if k >= n:
        return torch.arange(n, device=embeds.device)
    if k <= 0:
        return embeds.new_empty(0, dtype=torch.long)

    if method == "random":
        return _random_indices(n, k, embeds.device, generator)
    if method == "spatial_uniform":
        return _spatial_uniform_indices(n, k, embeds.device, grid_dims)
    if method == "activation_magnitude":
        return _magnitude_indices(embeds, k)
    if method == "diversity":
        return _greedy_indices(embeds, k, alpha=0.0, beta=1.0)
    if method == "hybrid":
        return _greedy_indices(embeds, k, alpha=_HYBRID_ALPHA, beta=_HYBRID_BETA)
    raise ValueError(f"unknown scoring method {method!r}; choose from {SCORING_METHODS}")


def prune_group(
    embeds: torch.Tensor,
    keep_ratio: float,
    method: str,
    grid_dims: Optional[Sequence[Sequence[int]]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Reconstruct-to-full-length prune of one group's embeds.

    Returns a new (N, D) tensor where the (N-k) pruned rows are zeroed and the k
    kept rows carry their original values. Shape is preserved so the downstream
    1:1 injection contract holds. Does not mutate ``embeds``.
    """
    if keep_ratio >= 1.0:
        return embeds
    keep = select_keep_indices(embeds, keep_ratio, method, grid_dims, generator)
    out = torch.zeros_like(embeds)
    if keep.numel():
        out[keep] = embeds[keep]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Non-invasive runtime hook
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PruneSpec:
    """How to prune one DeepStack group."""

    method: str
    keep_ratio: float

    def __post_init__(self) -> None:
        if self.method not in SCORING_METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {SCORING_METHODS}")
        if not 0.0 <= self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in [0,1], got {self.keep_ratio}")


class DeepStackPruner:
    """Context manager that prunes DeepStack groups in-place before injection.

    Two hooks (both pre-hooks, both reading kwargs; neither edits model source):
      - on Qwen3VLModel.forward: cache ``image_grid_thw`` for the spatial scorer.
      - on Qwen3VLTextModel.forward: replace each configured group's embeds with
        its reconstruct-to-full-length pruned version. Injection only fires on the
        prefill forward (decode steps carry no deepstack embeds), so the hook
        naturally affects only prefill.

    Lifecycle mirrors DeepStackAblator: enter to register, set the active spec map
    with ``set_specs`` between forwards, exit to remove the hooks. ``seed`` makes
    the random scorer reproducible across conditions.
    """

    def __init__(
        self,
        model: Any,
        group_specs: Optional[Dict[int, PruneSpec]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        self.model = model
        self.num_groups = int(len(model.config.vision_config.deepstack_visual_indexes))
        self._specs: Dict[int, PruneSpec] = dict(group_specs or {})
        self._seed = seed
        self._grid: Optional[List[List[int]]] = None
        self._handles: List[Any] = []

    def set_specs(self, group_specs: Dict[int, PruneSpec]) -> None:
        self._specs = dict(group_specs)

    def set_uniform(self, method: str, keep_ratio: float) -> None:
        """Convenience: same (method, keep_ratio) for every group."""
        self._specs = {i: PruneSpec(method, keep_ratio) for i in range(self.num_groups)}

    def _generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(self._seed)
        return g

    def _model_pre_hook(self, _module: Any, _args: Any, kwargs: Dict[str, Any]) -> None:
        grid = kwargs.get("image_grid_thw")
        if grid is None:
            self._grid = None
            return
        try:
            self._grid = [[int(x) for x in row] for row in grid.tolist()]
        except Exception:  # noqa: BLE001 — grid is advisory (spatial scorer only)
            self._grid = None

    def _text_pre_hook(self, _module: Any, _args: Any, kwargs: Dict[str, Any]) -> None:
        if not self._specs:
            return
        embeds: Optional[List[torch.Tensor]] = kwargs.get("deepstack_visual_embeds")
        if not embeds:  # None (decode step) or empty -> nothing to prune
            return
        for gi, spec in self._specs.items():
            if not (0 <= gi < len(embeds)) or spec.keep_ratio >= 1.0:
                continue
            embeds[gi] = prune_group(
                embeds[gi],
                spec.keep_ratio,
                spec.method,
                grid_dims=self._grid,
                generator=self._generator(embeds[gi].device),
            )

    def _register(self) -> None:
        for module in self.model.modules():
            tn = type(module).__name__
            if tn == "Qwen3VLModel":
                self._handles.append(module.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
            elif tn == "Qwen3VLTextModel":
                self._handles.append(module.register_forward_pre_hook(self._text_pre_hook, with_kwargs=True))

    def __enter__(self) -> "DeepStackPruner":
        self._register()
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
