"""
Per-DeepStack-group budget allocation (Phase 5, Stage A).

Pure logic — no model load, no CUDA — so the whole module is CPU unit-testable.
It supports the Stage-A study on General VQA (paper.md Phase 5):

  1. enumerate the independent 1D per-group sweep (each group's keep-ratio swept
     100% -> 0% in `step` increments, the other groups held full);
  2. from the measured per-group sensitivity curves, construct candidate *joint*
     per-group budgets at a target average keep-ratio (separable allocation, a.k.a.
     water-filling), to be validated against `uniform_budget` and
     `global_topk_keepsets` at an **equal retained-token count** (paper.md §10
     Experiment 3's fairness unit).

Accounting unit (Stage A uses zeroing, not real pruning — see exp_budgeting /
paper.md §13 Phase 4): "retained tokens" = the count of non-zeroed DeepStack
refinement rows, summed across groups. Methods are compared at equal retained
total, not at equal sequence length (real sequence-shortening is Stage B).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# A per-group budget is a tuple of keep-ratios, one per DeepStack group (G0,G1,G2).
Budget = Tuple[float, ...]
# A measured curve for one (group, scorer): {keep_ratio: accuracy}. Includes the
# 1.0 baseline point so allocation can reason about "keep all".
Curve = Dict[float, float]


def _r(x: float, ndigits: int = 6) -> float:
    """Round to kill float drift so ratios are dict-key stable across the grid."""
    return round(float(x), ndigits)


def sweep_ratios(step: float = 0.05) -> Tuple[float, ...]:
    """The per-group sweep grid below full: (1-step, 1-2*step, ..., step, 0.0).

    `1.0` (full) is the shared baseline and is intentionally excluded here. For
    step=0.05 this is the 20 values 0.95, 0.90, ..., 0.05, 0.0.
    """
    if not 0.0 < step < 1.0:
        raise ValueError(f"step must be in (0,1), got {step}")
    n = int(round(1.0 / step))
    return tuple(_r(step * k) for k in range(n - 1, -1, -1))


def sweep_conditions(
    num_groups: int, scorers: Sequence[str], step: float = 0.05
) -> List[Tuple[int, str, float]]:
    """The independent 1D sweep: (group, scorer, keep_ratio) conditions.

    Each group is swept across every scorer and every ratio in `sweep_ratios`,
    except `ratio == 0.0` (keep zero tokens) is **scorer-independent**, so it is
    emitted once per group rather than once per scorer. The 0.0 endpoint is a
    sanity anchor — it must reproduce Phase 3's `drop_g{i}` accuracy.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if not scorers:
        raise ValueError("scorers must be non-empty")
    ratios = sweep_ratios(step)
    out: List[Tuple[int, str, float]] = []
    for gi in range(num_groups):
        for r in ratios:
            if r == 0.0:
                out.append((gi, scorers[0], 0.0))  # one zero condition per group
            else:
                for s in scorers:
                    out.append((gi, s, r))
    return out


def keep_count(n: int, keep_ratio: float) -> int:
    """k = round(keep_ratio * n), clamped to [0, n] — matches prune._keep_count."""
    if keep_ratio >= 1.0:
        return n
    if keep_ratio <= 0.0:
        return 0
    return max(0, min(n, int(round(keep_ratio * n))))


def retained_total(ratios: Sequence[float], n: int) -> int:
    """Total retained refinement rows Σ keep_count(n, r) — the equal-budget unit."""
    return sum(keep_count(n, r) for r in ratios)


def uniform_budget(target_avg: float, num_groups: int) -> Budget:
    """All groups at the target average keep-ratio."""
    if not 0.0 <= target_avg <= 1.0:
        raise ValueError(f"target_avg must be in [0,1], got {target_avg}")
    return tuple(_r(target_avg) for _ in range(num_groups))


def _grid_with_full(step: float) -> Tuple[float, ...]:
    """The full per-group grid including 1.0 (for joint budget enumeration)."""
    return (1.0,) + sweep_ratios(step)


def waterfill_budgets(
    curves: Dict[int, Curve],
    target_avg: float,
    n: int,
    step: float = 0.05,
    n_candidates: int = 3,
) -> List[Budget]:
    """Candidate joint per-group budgets at `target_avg`, ranked by predicted accuracy.

    `curves[group]` maps each grid keep-ratio (including 1.0) to that group's
    measured accuracy when only that group is pruned to that ratio (others full).
    Under a separability prior, the predicted accuracy of a joint budget
    `(r0,r1,r2)` is approximated by `sum_g curves[g][r_g] - (G-1)*baseline`
    (each single-group curve already includes the baseline, so we subtract the
    over-counted baselines). Ranking is invariant to that constant, so we just
    rank by `sum_g curves[g][r_g]`.

    We enumerate the full grid (|grid|^G; tiny for G=3, ~21^3) and keep vectors
    whose `retained_total` is closest to the target `K = round(target_avg*G*n)`,
    then return the top `n_candidates` by predicted accuracy. Exact (not greedy),
    robust to non-monotonic / noisy curves.
    """
    groups = sorted(curves)
    g = len(groups)
    if g == 0:
        return []
    grid = _grid_with_full(step)
    target_k = round(target_avg * g * n)

    def predicted_acc(vec: Budget) -> float:
        return sum(curves[gi].get(_r(r), float("-inf")) for gi, r in zip(groups, vec))

    # Enumerate all G-dim ratio vectors over the grid.
    vectors: List[Budget] = [()]
    for _ in range(g):
        vectors = [v + (r,) for v in vectors for r in grid]

    scored = [
        (abs(retained_total(v, n) - target_k), -predicted_acc(v), v) for v in vectors
    ]
    scored.sort()
    # Keep vectors at the closest achievable retained total, ranked by accuracy.
    best_dist = scored[0][0]
    at_target = [s for s in scored if s[0] == best_dist]
    return [v for _, _, v in at_target[:n_candidates]]


def global_topk_keepsets(
    scores_by_group: Dict[int, torch.Tensor], k_total: int
) -> Dict[int, torch.Tensor]:
    """Flat top-k across all groups (RQ3 baseline: does group structure matter?).

    Concatenates the per-token scalar scores of every group, takes the global
    top-`k_total`, and returns the kept *local* indices per group. Groups that win
    no tokens get an empty index tensor. The selection ignores DeepStack group
    boundaries entirely, which is exactly the comparison we want against per-group
    budgeting.
    """
    groups = sorted(scores_by_group)
    if not groups:
        return {}
    sizes = [int(scores_by_group[gi].numel()) for gi in groups]
    total = sum(sizes)
    k = max(0, min(int(k_total), total))
    flat = torch.cat([scores_by_group[gi].reshape(-1).float() for gi in groups], dim=0)
    if k == 0:
        return {gi: flat.new_empty(0, dtype=torch.long) for gi in groups}
    top = torch.topk(flat, k, largest=True, sorted=False).indices
    # Map flat indices back to (group, local index) via cumulative offsets.
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    out: Dict[int, torch.Tensor] = {}
    for gi_pos, gi in enumerate(groups):
        lo, hi = offsets[gi_pos], offsets[gi_pos + 1]
        in_group = top[(top >= lo) & (top < hi)] - lo
        out[gi] = in_group.to(torch.long)
    return out


def extract_curves(
    sweep_json: Dict[str, Any],
) -> Tuple[Dict[Tuple[int, str], Curve], float]:
    """Read `budgeting_sweep.json` into per-(group,scorer) accuracy curves.

    Returns (curves, baseline_accuracy) where `curves[(group, scorer)]` maps each
    keep-ratio (with the 1.0 baseline point injected) to accuracy. The `0.0`
    condition is stored once in the JSON (scorer-independent); it is broadcast to
    every scorer's curve here so each curve spans the whole grid.
    """
    baseline = float(sweep_json.get("baseline_accuracy", 0.0))
    scorers: List[str] = list(sweep_json.get("scorers", []))
    acc: Dict[str, Dict[str, Dict[str, float]]] = sweep_json.get("accuracy", {})

    curves: Dict[Tuple[int, str], Curve] = {}
    for g_key, by_scorer in acc.items():
        gi = int(g_key)
        zero_acc = None
        for vals in by_scorer.values():
            if "0.0" in vals or "0.00" in vals:
                zero_acc = vals.get("0.0", vals.get("0.00"))
                break
        for s in scorers:
            curve: Curve = {1.0: baseline}
            for r_key, v in by_scorer.get(s, {}).items():
                curve[_r(float(r_key))] = float(v)
            if zero_acc is not None:
                curve[0.0] = float(zero_acc)
            curves[(gi, s)] = curve
    return curves, baseline


def curves_for_scorer(
    curves: Dict[Tuple[int, str], Curve], scorer: str
) -> Dict[int, Curve]:
    """Slice the (group,scorer)->curve map down to one scorer: group->curve."""
    return {gi: c for (gi, s), c in curves.items() if s == scorer}


def best_scorer(
    curves: Dict[Tuple[int, str], Curve],
    scorers: Sequence[str],
    num_groups: int,
    aggressive_ratio: float = 0.25,
) -> Optional[str]:
    """Pick the scorer that best preserves accuracy at an aggressive keep-ratio,
    averaged over groups (the discriminating regime; mild ratios are all-noise)."""
    target = _r(aggressive_ratio)
    best: Optional[str] = None
    best_score = float("-inf")
    for s in scorers:
        vals = [curves[(gi, s)].get(target) for gi in range(num_groups) if (gi, s) in curves]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        mean_acc = sum(present) / len(present)
        if mean_acc > best_score:
            best_score, best = mean_acc, s
    return best
