"""
Visualize a DeepStack within-group scoring report (Phase 4).

Reads a `scoring.json` (produced by src/experiments/exp_scoring.py) and renders
the scorer-comparison figures plus a written EXPLAINER.

This script does NOT load the model — it only reads the JSON — so it is safe to
run locally:

    python -m src.deepstack.visualize_scoring results/<ts>/scoring.json

Outputs (next to the JSON, in a `figures/` subdir):
    scoring_accuracy_curves.png   per-task accuracy vs keep-ratio, one line/scorer
    scoring_bar_at_50pct.png      per-task grouped bars of accuracy at keep_ratio 0.50
    EXPLAINER_scoring.md          plain-language read: which scorer wins, by how much
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

_BASELINE_KEY = "full"
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _pruning_ratios(report: Dict[str, Any]) -> List[float]:
    """Keep-ratios that represent actual pruning (everything below 1.0), sorted."""
    return sorted(r for r in report["keep_ratios"] if r < 1.0)


def _cond_key(method: str, ratio: float) -> str:
    return f"{method}@{ratio:.2f}"


def _baseline_acc(report: Dict[str, Any], task: str) -> Optional[float]:
    return report.get("accuracy", {}).get(task, {}).get(_BASELINE_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — accuracy vs keep-ratio, one panel per task
# ═══════════════════════════════════════════════════════════════════════════════


def _plot_curves(report: Dict[str, Any], out: Path) -> None:
    tasks = [t for t in report["tasks"] if t in report.get("accuracy", {})]
    methods = report["methods"]
    ratios = _pruning_ratios(report)
    if not tasks or not ratios:
        return

    ncol = min(2, len(tasks))
    nrow = (len(tasks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.5 * nrow), squeeze=False)
    acc = report["accuracy"]

    # x-axis values in monotonic (descending) order so the connecting line does not
    # zig-zag: full (1.0) then the pruning ratios from largest to smallest.
    x_full = [1.0] + sorted(ratios, reverse=True)
    for ti, task in enumerate(tasks):
        ax = axes[ti // ncol][ti % ncol]
        base = _baseline_acc(report, task)
        for mi, method in enumerate(methods):
            ys = [base] + [acc[task].get(_cond_key(method, r)) for r in sorted(ratios, reverse=True)]
            xs = [x for x, y in zip(x_full, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            ax.plot(xs, yy, marker=_MARKERS[mi % len(_MARKERS)], label=method, linewidth=1.8)
        if base is not None:
            ax.axhline(base, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_title(f"{task}  (full = {base:.3f})" if base is not None else task)
        ax.set_xlabel("keep-ratio (fraction of visual tokens retained, per group)")
        ax.set_ylabel("accuracy")
        ax.set_xticks(x_full)
        ax.invert_xaxis()  # most-pruned on the right -> reads as "degradation"
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # hide any unused panels
    for j in range(len(tasks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(
        "Within-group scoring: accuracy vs retained-token ratio (uniform budget across groups)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "scoring_accuracy_curves.png", dpi=130)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — grouped bars at a single mid-compression keep-ratio
# ═══════════════════════════════════════════════════════════════════════════════


def _pick_mid_ratio(ratios: List[float]) -> Optional[float]:
    """Prefer 0.50 if present, else the median pruning ratio."""
    if not ratios:
        return None
    if any(abs(r - 0.50) < 1e-6 for r in ratios):
        return 0.50
    return ratios[len(ratios) // 2]


def _plot_bars(report: Dict[str, Any], out: Path) -> Optional[float]:
    tasks = [t for t in report["tasks"] if t in report.get("accuracy", {})]
    methods = report["methods"]
    ratios = _pruning_ratios(report)
    mid = _pick_mid_ratio(ratios)
    if not tasks or mid is None:
        return None

    acc = report["accuracy"]
    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(2.2 * len(tasks) + 3, 5))

    for mi, method in enumerate(methods):
        vals = [acc[t].get(_cond_key(method, mid), np.nan) for t in tasks]
        bars = ax.bar(x + mi * width, vals, width, label=method)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
    # baseline ticks per task
    for ti, t in enumerate(tasks):
        base = _baseline_acc(report, t)
        if base is not None:
            ax.hlines(base, x[ti] - 0.1, x[ti] + 0.8, color="black", linestyle=":", linewidth=1.2)

    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("accuracy")
    ax.set_title(
        f"Scorer comparison at keep-ratio {mid:.2f} (dotted line = full, no pruning)"
    )
    ax.legend(fontsize=8, ncol=len(methods))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "scoring_bar_at_50pct.png", dpi=130)
    plt.close(fig)
    return mid


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════════


def _winners_at(report: Dict[str, Any], ratio: float) -> Dict[str, str]:
    """Best-accuracy scorer per task at a given keep-ratio."""
    acc = report["accuracy"]
    methods = report["methods"]
    winners: Dict[str, str] = {}
    for t in acc:
        scores = {m: acc[t].get(_cond_key(m, ratio)) for m in methods}
        scores = {m: v for m, v in scores.items() if v is not None}
        if scores:
            winners[t] = max(scores, key=lambda m: scores[m])
    return winners


def _build_explainer(report: Dict[str, Any], mid: Optional[float]) -> str:
    ratios = _pruning_ratios(report)
    methods = report["methods"]
    tasks = [t for t in report["tasks"] if t in report.get("accuracy", {})]
    lines = [
        "# DeepStack Phase 4 — Within-Group Scoring Comparison",
        "",
        f"- **Model:** `{report['model_source']}`",
        f"- **Groups:** {report['num_groups']} — ViT layers {report['vision_layers']}.",
        f"- **Budget mode:** `{report.get('budget_mode')}` — the *same* keep-ratio is applied to "
        "every DeepStack group, so this isolates scorer quality from budget allocation (Phase 5).",
        f"- **Scorers:** {methods}",
        f"- **Keep-ratios:** {report['keep_ratios']}  (1.0 = full / no pruning baseline)",
        f"- **Tasks / samples:** {report.get('samples_per_task', {})}",
        "",
        "## What this experiment answers",
        "Given a fixed retained-token budget, *which tokens should we keep?* Every scorer keeps the "
        "exact same number of tokens per group, so any accuracy difference is purely about token "
        "*selection*. `random` is the control — a useful scorer must beat it. `activation_magnitude`, "
        "`diversity`, and `hybrid` are vision-side feature signals; `vision_attention` is the "
        "literature's strong vision-encoder attention-received signal (VisPruner / FasterVLM, CLS-free "
        "recipe). Decoder attention is deliberately excluded — Phase 2 showed it is a null.",
        "",
        "**Rank by accuracy, not KL.** The first-token KL reported alongside is a secondary signal and "
        "is *structurally biased* toward `activation_magnitude` (KL rewards a small output shift, and "
        "magnitude keeps the largest additive vectors, minimizing that shift almost by construction). "
        "At this sample size the verdict is read off accuracy; KL is a cross-check only.",
        "",
        "## Figures",
        "- `scoring_accuracy_curves.png` — per task, accuracy as the keep-ratio drops from 1.0. A "
        "scorer whose line stays high as we move right (more pruning) preserves the right tokens.",
        "- `scoring_bar_at_50pct.png` — a single-glance comparison at a mid-compression keep-ratio.",
        "",
    ]

    # Data-driven read.
    if mid is not None:
        winners = _winners_at(report, mid)
        acc = report["accuracy"]
        lines += [f"## Read at keep-ratio {mid:.2f}", ""]
        for t in tasks:
            base = _baseline_acc(report, t)
            scores = {m: acc[t].get(_cond_key(m, mid)) for m in methods}
            scores = {m: v for m, v in scores.items() if v is not None}
            if not scores:
                continue
            rand = scores.get("random")
            best_m = max(scores, key=lambda m: scores[m])
            best_v = scores[best_m]
            gap = f"{(best_v - rand):+.3f} vs random" if rand is not None else ""
            drop = f"{(best_v - base):+.3f} vs full" if base is not None else ""
            lines.append(f"- **{t}**: best = `{best_m}` ({best_v:.3f}; {gap}; {drop}).")
        lines.append("")
        distinct = set(winners.values())
        if len(distinct) == 1:
            lines.append(
                f"A single scorer (`{next(iter(distinct))}`) wins every task at this ratio — a clean "
                "default for the Phase 5 budgeting experiments."
            )
        else:
            lines.append(
                "The best scorer is task-dependent: "
                + ", ".join(f"{t}→`{m}`" for t, m in winners.items())
                + ". Consider this when fixing the within-group scorer for Phase 5."
            )
        lines.append("")

    # Full table.
    cond_cols = [_BASELINE_KEY] + [_cond_key(m, r) for r in ratios for m in methods]
    lines += [
        "## Full accuracy table",
        "",
        "| task | " + " | ".join(cond_cols) + " |",
        "|" + "---|" * (len(cond_cols) + 1),
    ]
    acc = report.get("accuracy", {})
    for t in tasks:
        row = [t] + [
            (f"{acc[t][c]:.3f}" if c in acc.get(t, {}) else "—") for c in cond_cols
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def visualize(json_path: str, output_dir: Optional[str] = None) -> Path:
    jp = Path(json_path)
    with open(jp, encoding="utf-8") as f:
        report = json.load(f)

    out = Path(output_dir) if output_dir else jp.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    _plot_curves(report, out)
    mid = _plot_bars(report, out)
    (out / "EXPLAINER_scoring.md").write_text(_build_explainer(report, mid), encoding="utf-8")

    print(f"Wrote scoring figures + EXPLAINER_scoring.md to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a DeepStack scoring report (Phase 4).")
    parser.add_argument("json_path", type=str, help="Path to scoring.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <json dir>/figures")
    args = parser.parse_args()
    visualize(args.json_path, args.output_dir)


if __name__ == "__main__":
    main()
