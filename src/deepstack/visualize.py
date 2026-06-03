"""
Visualize a DeepStack instrumentation report (Phase 2).

Reads a `deepstack_instrument.json` (produced by src/deepstack/instrument.py) and
renders plain-English figures that explain what the per-group statistics mean,
plus a one-page dashboard and a written EXPLAINER.md.

This script does NOT load the model — it only reads the JSON — so it is safe to
run locally:

    python -m src.deepstack.visualize results/<ts>/deepstack_instrument.json

Outputs (next to the JSON, in a `figures/` subdir):
    01_norm_distribution.png   per-group token-strength distributions (overlay)
    02_norm_summary.png        mean strength per group, with spread
    03_prunability.png         % of each group's tokens below a low-strength cut
    04_latency.png             extraction vs injection time per group
    05_attention.png           per-group attention saliency (only if captured)
    dashboard.png              all of the above on one page
    EXPLAINER.md               plain-language walkthrough of every figure
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

# Colour per group, reused across every figure so groups are visually consistent.
_GROUP_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
_LOW_STRENGTH_PERCENTILE = 25  # "low-value" cutoff for the prunability figure


def _color(i: int) -> str:
    return _GROUP_COLORS[i % len(_GROUP_COLORS)]


def _group_label(g: Dict[str, Any]) -> str:
    return f"Group {g['group_index']} (ViT L{g['vision_layer']} → dec L{g['injection_decoder_layer']})"


def _hist_centers(dist: Dict[str, Any]) -> "tuple[np.ndarray, np.ndarray]":
    """Bin centers and per-bin token fraction for a stored histogram."""
    edges = np.asarray(dist["hist_edges"], dtype=float)
    counts = np.asarray(dist["hist_counts"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    frac = counts / total if total > 0 else counts
    return centers, frac


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual figures (each takes a matplotlib Axes so they compose into a dashboard)
# ═══════════════════════════════════════════════════════════════════════════════


def _plot_norm_distribution(ax: "plt.Axes", groups: List[Dict[str, Any]]) -> None:
    p90s = []
    for i, g in enumerate(groups):
        d = g.get("norm_dist")
        if not d:
            continue
        centers, frac = _hist_centers(d)
        ax.plot(centers, frac, color=_color(i), lw=2, label=_group_label(g))
        ax.axvline(d["p50"], color=_color(i), ls="--", lw=1, alpha=0.7)
        p90s.append(d["p90"])
    if p90s:
        # Focus on the bulk; a single heavy tail would otherwise squish everything left.
        ax.set_xlim(left=0, right=max(p90s) * 1.8)
    ax.set_xlabel("Token strength  (L2 norm of the 2048-d visual token)")
    ax.set_ylabel("Fraction of tokens")
    ax.set_title("How strong each group's visual tokens are\n(dashed line = group median; right = stronger)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _plot_norm_summary(ax: "plt.Axes", groups: List[Dict[str, Any]]) -> None:
    labels, means, stds, p10s, p50s, p90s, colors = [], [], [], [], [], [], []
    for i, g in enumerate(groups):
        d = g.get("norm_dist")
        if not d:
            continue
        labels.append(f"G{g['group_index']}\nViT L{g['vision_layer']}")
        means.append(d["mean"])
        stds.append(d["std"])
        p10s.append(d["p10"])
        p50s.append(d["p50"])
        p90s.append(d["p90"])
        colors.append(_color(i))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, label="mean ± std")
    ax.plot(x, p50s, "k_", ms=18, mew=2, label="median (p50)")
    ax.plot(x, p10s, "kv", ms=6, label="p10 / p90")
    ax.plot(x, p90s, "k^", ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Token strength (L2 norm)")
    ax.set_title("Average token strength per group\n(grows with depth = deeper features carry more)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def _plot_prunability(ax: "plt.Axes", groups: List[Dict[str, Any]]) -> None:
    # Shared, model-derived cutoff = the median strength pooled across ALL groups.
    # A group with more tokens below the overall median is weaker / more prunable.
    dists = [g["norm_dist"] for g in groups if g.get("norm_dist")]
    if not dists:
        ax.set_axis_off()
        return
    pooled = np.concatenate([_reconstruct_samples(d) for d in dists])
    cut = float(np.median(pooled))
    labels, fracs, colors = [], [], []
    for i, g in enumerate(groups):
        d = g.get("norm_dist")
        if not d:
            continue
        vals = _reconstruct_samples(d)
        labels.append(f"G{g['group_index']}\nViT L{g['vision_layer']}")
        fracs.append(100.0 * float((vals < cut).mean()))
        colors.append(_color(i))
    x = np.arange(len(labels))
    bars = ax.bar(x, fracs, color=colors, alpha=0.85)
    for b, f in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{f:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"% tokens below overall median strength ({cut:.1f})")
    ax.set_title(
        "How much of each group is low-value\n"
        "(weak tokens = safe to prune; cutoff = median strength across all groups)"
    )
    ax.set_ylim(0, max(fracs + [1]) * 1.25)
    ax.grid(alpha=0.3, axis="y")


def _reconstruct_samples(dist: Dict[str, Any], n: int = 20000) -> np.ndarray:
    """Approximate the underlying values by resampling the stored histogram.

    Lets us compute comparable thresholds/fractions without the raw per-token data.
    """
    edges = np.asarray(dist["hist_edges"], dtype=float)
    counts = np.asarray(dist["hist_counts"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total <= 0:
        return np.array([dist.get("mean", 0.0)])
    reps = np.rint(counts / total * n).astype(int)
    return np.repeat(centers, reps) if reps.sum() > 0 else centers


def _plot_latency(ax: "plt.Axes", groups: List[Dict[str, Any]]) -> None:
    labels = [f"G{g['group_index']}" for g in groups]
    extract = [g["extraction_ms_mean"] for g in groups]
    inject = [g["injection_ms_mean"] for g in groups]
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, extract, w, label="extraction", color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, inject, w, label="injection", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time per image (ms)")
    total = sum(extract) + sum(inject)
    ax.set_title(f"DeepStack overhead per group\n(total ≈ {total:.1f} ms — cheap; savings must come from fewer tokens)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def _plot_attention(ax: "plt.Axes", groups: List[Dict[str, Any]]) -> bool:
    drawn = False
    for i, g in enumerate(groups):
        d = g.get("attention_dist")
        if not d:
            continue
        centers, frac = _hist_centers(d)
        ax.plot(centers, frac, color=_color(i), lw=2, label=_group_label(g))
        ax.axvline(d["p50"], color=_color(i), ls="--", lw=1, alpha=0.7)
        drawn = True
    if not drawn:
        ax.text(
            0.5, 0.5, "Attention not captured\n(run with --capture-attention)",
            ha="center", va="center", fontsize=11, color="gray",
        )
        ax.set_axis_off()
        return False
    ax.set_xlabel("Attention received per visual token (avg over heads/queries)")
    ax.set_ylabel("Fraction of tokens")
    ax.set_title("How much attention each group's tokens get\n(right = more attended = more relied upon)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestration
# ═══════════════════════════════════════════════════════════════════════════════


def _save_single(fig_fn, groups: List[Dict[str, Any]], path: Path, figsize=(8, 5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig_fn(ax, groups)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _build_explainer(report: Dict[str, Any], has_attention: bool) -> str:
    groups = report["groups"]
    lines = [
        "# DeepStack Phase 2 — What the Figures Mean",
        "",
        f"- **Model:** `{report['model_source']}`",
        f"- **Samples:** {report['num_samples']}  (sources: {report['per_sample_source']})",
        f"- **Groups:** {report['num_groups']} — visual features tapped at ViT layers "
        f"{report['vision_layers']} and injected into decoder layers {report['injection_layers']}.",
        f"- **Visual tokens per image:** {report['per_sample_visual_tokens']}",
        "",
        "Every group carries the **same number** of tokens for a given image — DeepStack "
        "injects the same token positions at each depth. So the question is not *how many* "
        "tokens a group has, but *how valuable its tokens are*. The figures below answer that.",
        "",
        "## 1. Token-strength distribution (`01_norm_distribution.png`)",
        "Each visual token is a 2048-number vector; its **strength** is the vector's length "
        "(L2 norm) — loosely, how much signal it injects. The curves show how strength is "
        "spread within each group. A curve pushed to the **right** = stronger tokens. A tall "
        "spike on the **left** = lots of weak, likely-redundant tokens.",
        "",
        "## 2. Average strength per group (`02_norm_summary.png`)",
        "The same thing, summarized: mean ± spread, with median and the p10/p90 range. If "
        "strength **rises with depth**, deeper groups pack more information per token.",
        "",
        "## 3. Prunability (`03_prunability.png`)",
        "The percentage of each group's tokens that fall **below a low-strength cutoff**. "
        "Higher bar = more of that group is low-value = safer to prune. This is the most "
        "direct hint at where a token budget can be cut with least damage.",
        "",
        "## 4. DeepStack overhead (`04_latency.png`)",
        "Time to build (extraction) and add (injection) each group's tokens. These are tiny, "
        "which matters: it means real speedups come from **reducing the token count the "
        "decoder must process**, not from touching the DeepStack machinery itself.",
        "",
    ]
    if has_attention:
        lines += [
            "## 5. Attention saliency (`05_attention.png`)",
            "How much attention each group's tokens **receive** from the rest of the sequence "
            "(averaged over heads and query positions). Tokens that get more attention are "
            "relied upon more, so a group whose tokens are widely ignored is more compressible.",
            "",
        ]
    # A small data-driven takeaway.
    norms = [(g["group_index"], g["vision_layer"], g["norm_dist"]["mean"]) for g in groups if g.get("norm_dist")]
    if norms:
        norms_sorted = sorted(norms, key=lambda t: t[2])
        weak, strong = norms_sorted[0], norms_sorted[-1]
        lines += [
            "## Takeaway from this run",
            f"- Weakest group on average: **Group {weak[0]}** (ViT L{weak[1]}, mean strength {weak[2]:.1f}) "
            "→ most prunable.",
            f"- Strongest group on average: **Group {strong[0]}** (ViT L{strong[1]}, mean strength {strong[2]:.1f}) "
            "→ most fragile, keep more of it.",
            "",
            "This non-uniformity across groups is exactly the signal the per-group budgeting "
            "method needs (paper.md §4 hypothesis).",
        ]
    return "\n".join(lines) + "\n"


def visualize(json_path: str, output_dir: Optional[str] = None) -> Path:
    jp = Path(json_path)
    with open(jp, encoding="utf-8") as f:
        report = json.load(f)
    groups = report["groups"]

    out = Path(output_dir) if output_dir else jp.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)

    _save_single(_plot_norm_distribution, groups, out / "01_norm_distribution.png")
    _save_single(_plot_norm_summary, groups, out / "02_norm_summary.png")
    _save_single(_plot_prunability, groups, out / "03_prunability.png")
    _save_single(_plot_latency, groups, out / "04_latency.png", figsize=(7, 5))
    has_attention = any(g.get("attention_dist") for g in groups)
    _save_single(_plot_attention, groups, out / "05_attention.png")

    # Dashboard: all panels on one page.
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    _plot_norm_distribution(axes[0, 0], groups)
    _plot_norm_summary(axes[0, 1], groups)
    _plot_prunability(axes[0, 2], groups)
    _plot_latency(axes[1, 0], groups)
    _plot_attention(axes[1, 1], groups)
    axes[1, 2].set_axis_off()
    axes[1, 2].text(
        0.0,
        0.95,
        _dashboard_caption(report, has_attention),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        transform=axes[1, 2].transAxes,
    )
    fig.suptitle("DeepStack Phase 2 — per-group measurement dashboard", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "dashboard.png", dpi=130)
    plt.close(fig)

    explainer = _build_explainer(report, has_attention)
    (out / "EXPLAINER.md").write_text(explainer, encoding="utf-8")

    print(f"Wrote {len(list(out.glob('*.png')))} figures + EXPLAINER.md to {out}")
    return out


def _dashboard_caption(report: Dict[str, Any], has_attention: bool) -> str:
    g = report["groups"]
    rows = [
        "READING THIS DASHBOARD",
        "",
        f"samples : {report['num_samples']} ({'+'.join(sorted(set(report['per_sample_source'])))})",
        f"groups  : {report['num_groups']}  ViT {report['vision_layers']}",
        f"          -> dec {report['injection_layers']}",
        f"tokens  : {report['per_sample_visual_tokens']}",
        "",
        "per-group mean token strength:",
    ]
    for grp in g:
        d = grp.get("norm_dist")
        if d:
            rows.append(f"  G{grp['group_index']} (ViT L{grp['vision_layer']}): {d['mean']:6.1f}")
    rows += ["", "attention captured: " + ("yes" if has_attention else "no")]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a DeepStack instrumentation report (Phase 2).")
    parser.add_argument("json_path", type=str, help="Path to deepstack_instrument.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <json dir>/figures")
    args = parser.parse_args()
    visualize(args.json_path, args.output_dir)


if __name__ == "__main__":
    main()
