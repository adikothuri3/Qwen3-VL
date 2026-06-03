"""
Visualize a DeepStack sensitivity report (Phase 3).

Reads a `sensitivity.json` (produced by src/experiments/exp_sensitivity.py) and
renders the go/no-go figures: the per-task accuracy-drop heatmap (paper.md
Figure 3) and the first-token KL bar chart, plus a written EXPLAINER.

This script does NOT load the model — it only reads the JSON — so it is safe to
run locally:

    python -m src.deepstack.visualize_sensitivity results/<ts>/sensitivity.json

Outputs (next to the JSON, in a `figures/` subdir):
    sensitivity_heatmap.png        rows = tasks, cols = conditions, color = accuracy drop
    kl_by_condition.png            mean first-token KL per condition
    EXPLAINER_sensitivity.md       plain-language read + go/no-go verdict
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np


def _ablation_conditions(report: Dict[str, Any]) -> List[str]:
    """Conditions that actually represent an ablation (drop full, which is the ref)."""
    return [c for c in report["conditions"] if c != "full"]


def _drop_matrix(report: Dict[str, Any], tasks: List[str], conds: List[str]) -> np.ndarray:
    """tasks × conditions matrix of accuracy drop vs full (NaN where missing)."""
    mat = np.full((len(tasks), len(conds)), np.nan, dtype=float)
    drops = report.get("accuracy_drop", {})
    for r, t in enumerate(tasks):
        for c, cn in enumerate(conds):
            if t in drops and cn in drops[t]:
                mat[r, c] = drops[t][cn]
    return mat


def _plot_heatmap(ax: "plt.Axes", report: Dict[str, Any]) -> None:
    tasks = [t for t in report["tasks"] if t in report.get("accuracy_drop", {})]
    conds = _ablation_conditions(report)
    mat = _drop_matrix(report, tasks, conds)

    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-3)
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=10)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            if np.isfinite(mat[r, c]):
                ax.text(c, r, f"{mat[r, c]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(
        "Accuracy drop when a DeepStack group is ablated\n"
        "(redder = bigger drop = that group matters more for that task)"
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy drop vs full")


def _plot_kl(ax: "plt.Axes", report: Dict[str, Any]) -> None:
    conds = _ablation_conditions(report)
    kl = report.get("kl", {})
    vals = [kl.get(cn, 0.0) for cn in conds]
    x = np.arange(len(conds))
    bars = ax.bar(x, vals, color="#8172B3", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("mean KL(P_full || P_cond)")
    ax.set_title("Output shift per ablation (first-token KL)\n(larger = removing that group changes the answer more)")
    ax.grid(alpha=0.3, axis="y")


def _build_explainer(report: Dict[str, Any]) -> str:
    tasks = [t for t in report["tasks"] if t in report.get("accuracy", {})]
    groups = report["num_groups"]
    lines = [
        "# DeepStack Phase 3 — Group Sensitivity (Go/No-Go)",
        "",
        f"- **Model:** `{report['model_source']}`",
        f"- **Groups:** {report['num_groups']} — ViT layers {report['vision_layers']}, "
        "injected at the first decoder layers.",
        f"- **Tasks / samples:** {report.get('samples_per_task', {})}",
        f"- **Conditions:** {report['conditions']}  (drop_all == No-DeepStack baseline)",
        "",
        "## What this experiment answers",
        "Phase 2 showed groups differ in *feature structure* (the dispersion lens). It did **not** "
        "show they differ in *accuracy sensitivity*. This experiment removes each group and measures "
        "the per-task accuracy drop. **Different drops across groups = per-depth budgeting is "
        "justified** (paper.md §4, §16).",
        "",
        "## Figures",
        "- `sensitivity_heatmap.png` — rows = tasks, columns = ablations, color = accuracy drop vs "
        "full. A redder cell means that group is more load-bearing for that task. Read each row: if "
        "`drop_g0`/`drop_g1`/`drop_g2` differ within a row, that task relies on the groups unequally.",
        "- `kl_by_condition.png` — a dense, label-free cross-check: how much each ablation shifts the "
        "first-token distribution away from full DeepStack.",
        "",
    ]
    # Data-driven verdict.
    verdict = _verdict(report, tasks, groups)
    lines += ["## Verdict from this run", verdict, ""]
    lines += [
        "Per-task accuracy under each condition:",
        "",
        "| task | " + " | ".join(report["conditions"]) + " |",
        "|" + "---|" * (len(report["conditions"]) + 1),
    ]
    acc = report.get("accuracy", {})
    for t in tasks:
        row = [t] + [f"{acc[t].get(cn, float('nan')):.3f}" for cn in report["conditions"]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _verdict(report: Dict[str, Any], tasks: List[str], num_groups: int) -> str:
    """Heuristic read: do single-group drops vary across groups / across tasks?"""
    drops = report.get("accuracy_drop", {})
    drop_conds = [f"drop_g{i}" for i in range(num_groups)]
    per_group_spreads = []
    most_sensitive: Dict[str, str] = {}
    for t in tasks:
        if t not in drops:
            continue
        vals = {c: drops[t].get(c) for c in drop_conds if drops[t].get(c) is not None}
        if not vals:
            continue
        spread = max(vals.values()) - min(vals.values())
        per_group_spreads.append(spread)
        most_sensitive[t] = max(vals, key=lambda c: vals[c])
    if not per_group_spreads:
        return "_No scored tasks — cannot judge. Check that datasets loaded._"
    max_spread = max(per_group_spreads)
    distinct = len(set(most_sensitive.values()))
    sens_str = ", ".join(f"{t}→{c}" for t, c in most_sensitive.items())
    if max_spread >= 0.03:
        head = (
            f"**GO signal.** Within-task spread across single-group drops reaches {max_spread:.3f} "
            "accuracy — groups are *not* interchangeable."
        )
    else:
        head = (
            f"**Weak/NO signal.** Largest within-task spread is only {max_spread:.3f} accuracy — "
            "groups look similarly important so far (consider more samples/tasks before concluding)."
        )
    tail = (
        f" Most-load-bearing group per task: {sens_str}"
        + (" — and it differs across tasks (task-dependent sensitivity)." if distinct > 1 else ".")
    )
    return head + tail


def visualize(json_path: str, output_dir: Optional[str] = None) -> Path:
    jp = Path(json_path)
    with open(jp, encoding="utf-8") as f:
        report = json.load(f)

    out = Path(output_dir) if output_dir else jp.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_heatmap(ax, report)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_heatmap.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_kl(ax, report)
    fig.tight_layout()
    fig.savefig(out / "kl_by_condition.png", dpi=130)
    plt.close(fig)

    (out / "EXPLAINER_sensitivity.md").write_text(_build_explainer(report), encoding="utf-8")

    print(f"Wrote sensitivity figures + EXPLAINER_sensitivity.md to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a DeepStack sensitivity report (Phase 3).")
    parser.add_argument("json_path", type=str, help="Path to sensitivity.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <json dir>/figures")
    args = parser.parse_args()
    visualize(args.json_path, args.output_dir)


if __name__ == "__main__":
    main()
