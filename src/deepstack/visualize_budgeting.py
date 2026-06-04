"""
Visualize the Stage-A per-group budgeting study (Phase 5) — General VQA.

Reads `budgeting_sweep.json` (from `src/experiments/exp_budgeting.py sweep`) and,
optionally, `budgeting_validation.json` (from `... validate`), and renders the
per-group sensitivity curves, scorer comparison, joint-budget head-to-head, and a
written EXPLAINER. Does NOT load the model (JSON only), so it runs locally:

    python -m src.deepstack.visualize_budgeting results/<ts>/budgeting_sweep.json \
        [--validation results/<ts2>/budgeting_validation.json]

Outputs (in a `figures/` subdir next to the sweep JSON):
    budgeting_sensitivity_curves.png   accuracy vs keep-ratio, one panel per group, one line/scorer
    budgeting_kl_curves.png            first-token KL vs keep-ratio, per group
    budgeting_scorer_bars.png          accuracy at an aggressive keep-ratio, per group × scorer
    budgeting_validation.png           per-group-optimal vs uniform vs global at equal token count
    EXPLAINER_budgeting.md             plain-language read of the optimum + comparison
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from src.deepstack import budget

_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
_AGGR_RATIO = 0.25


# ═══════════════════════════════════════════════════════════════════════════════
#  Curve assembly (shared by accuracy + KL figures)
# ═══════════════════════════════════════════════════════════════════════════════


def _assemble(report: Dict[str, Any], metric: str) -> Dict[Tuple[int, str], List[Tuple[float, float]]]:
    """Per-(group, scorer) sorted [(ratio, value)] for `metric` in {accuracy, kl}.

    Injects the 1.0 baseline point (accuracy = baseline_accuracy; KL = 0.0) and
    broadcasts the shared 0.0 condition (stored once) to every scorer's curve.
    """
    scorers: List[str] = list(report.get("scorers", []))
    data: Dict[str, Dict[str, Dict[str, float]]] = report.get(metric, {})
    base = float(report.get("baseline_accuracy", 0.0)) if metric == "accuracy" else 0.0
    curves: Dict[Tuple[int, str], List[Tuple[float, float]]] = {}
    for g_key, by_scorer in data.items():
        gi = int(g_key)
        zero = None
        for vals in by_scorer.values():
            if "0.00" in vals:
                zero = vals["0.00"]
                break
        for s in scorers:
            pts: Dict[float, float] = {1.0: base}
            for rk, v in by_scorer.get(s, {}).items():
                pts[round(float(rk), 6)] = float(v)
            if zero is not None:
                pts[0.0] = float(zero)
            curves[(gi, s)] = sorted(pts.items(), reverse=True)
    return curves


# ═══════════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════════


def _plot_curves(report: Dict[str, Any], out: Path, metric: str, fname: str, ylabel: str, title: str) -> None:
    curves = _assemble(report, metric)
    num_groups = int(report.get("num_groups", 0))
    scorers = list(report.get("scorers", []))
    layers = report.get("vision_layers", [])
    groups = sorted({gi for (gi, _s) in curves})
    if not groups:
        return
    ncol = min(3, len(groups))
    nrow = (len(groups) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.5 * nrow), squeeze=False)
    base = float(report.get("baseline_accuracy", 0.0))
    for pos, gi in enumerate(groups):
        ax = axes[pos // ncol][pos % ncol]
        for mi, s in enumerate(scorers):
            pts = curves.get((gi, s))
            if not pts:
                continue
            xs = [r for r, _v in pts]
            ys = [v for _r, v in pts]
            ax.plot(xs, ys, marker=_MARKERS[mi % len(_MARKERS)], label=s, linewidth=1.7, markersize=4)
        if metric == "accuracy":
            ax.axhline(base, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        lyr = f" (ViT L{layers[gi]})" if gi < len(layers) else ""
        ax.set_title(f"Group {gi}{lyr}")
        ax.set_xlabel("keep-ratio (this group only; others full)")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()  # most-pruned on the right
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(groups), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{title}  (baseline acc = {base:.3f}, n={report.get('samples')})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / fname, dpi=130)
    plt.close(fig)
    _ = num_groups


def _plot_scorer_bars(report: Dict[str, Any], out: Path) -> None:
    curves = _assemble(report, "accuracy")
    scorers = list(report.get("scorers", []))
    groups = sorted({gi for (gi, _s) in curves})
    if not groups:
        return
    target = round(_AGGR_RATIO, 6)
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(scorers))
    fig, ax = plt.subplots(figsize=(2.5 * len(groups) + 3, 5))
    for mi, s in enumerate(scorers):
        vals = []
        for gi in groups:
            pts = dict(curves.get((gi, s), []))
            vals.append(pts.get(target, np.nan))
        bars = ax.bar(x + mi * width, vals, width, label=s)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    base = float(report.get("baseline_accuracy", 0.0))
    ax.axhline(base, color="black", linestyle=":", linewidth=1.2, label="full")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([f"G{gi}" for gi in groups])
    ax.set_ylabel("accuracy")
    ax.set_title(f"Scorer comparison at keep-ratio {_AGGR_RATIO:.2f} (dotted = full)")
    ax.legend(fontsize=8, ncol=len(scorers) + 1)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "budgeting_scorer_bars.png", dpi=130)
    plt.close(fig)


def _plot_validation(vreport: Dict[str, Any], out: Path) -> None:
    conds: Dict[str, Dict[str, Any]] = vreport.get("conditions", {})
    acc: Dict[str, float] = vreport.get("accuracy", {})
    ci: Dict[str, List[float]] = vreport.get("accuracy_ci", {})
    if not acc:
        return
    targets = sorted({c["target"] for c in conds.values()}, reverse=True)
    base = float(vreport.get("baseline_accuracy", 0.0))
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), squeeze=False)
    for ti, t in enumerate(targets):
        ax = axes[0][ti]
        keys = [k for k, c in conds.items() if c["target"] == t and k in acc]
        keys.sort(key=lambda k: (conds[k]["type"] != "pergroup", k))  # pergroup first
        vals = [acc[k] for k in keys]
        lo = [acc[k] - ci.get(k, [acc[k], acc[k]])[0] for k in keys]
        hi = [ci.get(k, [acc[k], acc[k]])[1] - acc[k] for k in keys]
        cmap = {"pergroup": "#2a7", "global": "#a55", "uniform": "#57a"}
        colors = [cmap.get(conds[k]["type"], "#57a") for k in keys]
        ax.bar(range(len(keys)), vals, yerr=[lo, hi], capsize=3, color=colors)
        ax.axhline(base, color="black", linestyle=":", linewidth=1.2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([conds[k]["type"] + (str(conds[k]["budget"]) if conds[k]["budget"] else "") for k in keys],
                           rotation=40, ha="right", fontsize=7)
        ax.set_title(f"avg keep ≈ {t:.2f}")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Joint budget at equal retained-token count — best scorer = {vreport.get('best_scorer')} "
        f"(dotted = full {base:.3f}, n={vreport.get('samples')})", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "budgeting_validation.png", dpi=130)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════════


def _build_explainer(report: Dict[str, Any], vreport: Optional[Dict[str, Any]]) -> str:
    scorers = list(report.get("scorers", []))
    num_groups = int(report.get("num_groups", 0))
    curves, base_acc = budget.extract_curves(report)
    best = budget.best_scorer(curves, scorers, num_groups, _AGGR_RATIO)
    lines = [
        "# DeepStack Phase 5 (Stage A) — Per-Group Budget + Scorer Search",
        "",
        f"- **Task:** `{report.get('task')}` (General VQA / VQAv2).",
        f"- **Model:** `{report.get('model_source')}`",
        f"- **Groups:** {num_groups} — ViT layers {report.get('vision_layers')}.",
        f"- **Scorers swept:** {scorers}",
        f"- **Samples:** {report.get('samples')}  •  baseline acc = {base_acc:.3f}",
        "",
        "## What this measures",
        "Each DeepStack group is pruned **independently** (the other two held full) from 100% → 0% in "
        f"{report.get('step')} steps, for every scorer. This is **zeroing** (the refinement at a token "
        "is set to 0; the base token still occupies the sequence), so there is no latency change here — "
        "methods are compared at an equal count of retained refinement rows. Real sequence-shortening + "
        "latency is Stage B.",
        "",
        "## Figures",
        "- `budgeting_sensitivity_curves.png` — per-group accuracy as each group is pruned alone. Flat "
        "lines = that group tolerates pruning; a steep drop = a fragile group.",
        "- `budgeting_kl_curves.png` — the same in first-token KL (dense, but biased toward magnitude).",
        "- `budgeting_scorer_bars.png` — which scorer best preserves accuracy at an aggressive keep-ratio.",
        "",
        f"## Best within-group scorer (at keep-ratio {_AGGR_RATIO:.2f}): `{best}`",
        "",
    ]
    # Per-group prunability read at the aggressive ratio.
    target = round(_AGGR_RATIO, 6)
    lines.append("## Per-group prunability read")
    for gi in range(num_groups):
        c = curves.get((gi, best or scorers[0]))
        if not c:
            continue
        v = c.get(target)
        if v is None:
            continue
        lines.append(f"- **Group {gi}** at keep {target:.2f} ({best}): acc {v:.3f} ({v - base_acc:+.3f} vs full).")
    lines.append("")

    if vreport:
        lines += [
            "## Joint budget validation (held-out, equal retained-token count)",
            "",
            f"- **Best scorer used:** `{vreport.get('best_scorer')}`  •  held-out n = {vreport.get('samples')}  "
            f"•  baseline acc = {vreport.get('baseline_accuracy', 0.0):.3f}",
            "",
            "| condition | type | budget | acc | 95% CI | retained |",
            "|---|---|---|---|---|---|",
        ]
        conds = vreport.get("conditions", {})
        acc = vreport.get("accuracy", {})
        ci = vreport.get("accuracy_ci", {})
        ret = vreport.get("retained_tokens", {})
        for k in sorted(acc, key=lambda k: (conds[k]["target"], conds[k]["type"]), reverse=True):
            c = conds[k]
            ci_k = ci.get(k, [float("nan"), float("nan")])
            lines.append(
                f"| {k} | {c['type']} | {c['budget']} | {acc[k]:.3f} | "
                f"[{ci_k[0]:.3f}, {ci_k[1]:.3f}] | {ret.get(k, 0):.0f} |"
            )
        lines += [
            "",
            "**How to read it:** compare `pergroup@T` vs `uniform@T` vs `global@T` *within each target T* "
            "(they share a retained-token count). If the per-group CI overlaps uniform's, per-group "
            "budgeting gives no benefit on this task — which, for General VQA, is the expected and "
            "honest outcome (Phase 3/4b found it compression-insensitive); the headline then is the "
            "large compressibility headroom and the scorer choice, with the per-group win reserved for "
            "the OCR/text tasks.",
            "",
        ]
    return "\n".join(lines) + "\n"


def visualize(sweep_path: str, validation_path: Optional[str] = None, output_dir: Optional[str] = None) -> Path:
    sp = Path(sweep_path)
    with open(sp, encoding="utf-8") as f:
        report = json.load(f)
    vreport = None
    if validation_path:
        with open(validation_path, encoding="utf-8") as f:
            vreport = json.load(f)

    out = Path(output_dir) if output_dir else sp.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    _plot_curves(report, out, "accuracy", "budgeting_sensitivity_curves.png", "accuracy",
                 "Per-group sensitivity (accuracy)")
    _plot_curves(report, out, "kl", "budgeting_kl_curves.png", "first-token KL",
                 "Per-group sensitivity (KL)")
    _plot_scorer_bars(report, out)
    if vreport:
        _plot_validation(vreport, out)
    (out / "EXPLAINER_budgeting.md").write_text(_build_explainer(report, vreport), encoding="utf-8")

    print(f"Wrote budgeting figures + EXPLAINER_budgeting.md to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the Stage-A budgeting study (Phase 5).")
    parser.add_argument("sweep_path", type=str, help="Path to budgeting_sweep.json")
    parser.add_argument("--validation", type=str, default=None, help="Path to budgeting_validation.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <sweep dir>/figures")
    args = parser.parse_args()
    visualize(args.sweep_path, args.validation, args.output_dir)


if __name__ == "__main__":
    main()
