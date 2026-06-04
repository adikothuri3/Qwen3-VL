"""
Per-group DeepStack budget + scorer search (Phase 5, Stage A).

Two subcommands, both run on Colab via colab_run.ipynb (never locally), and both
run over one or more tasks (default: general_vqa, docvqa, textvqa — the minimum
that tells the whole paper story: general_vqa = "uniform pruning is ~free"; doc/
textvqa = where per-group budgeting actually beats uniform). One result file is
written per task: budgeting_sweep__<task>.json / budgeting_validation__<task>.json.

  sweep    Independent 1D per-group sweep: prune each DeepStack group on its own
           from 100% -> 0% in `step` (default 5%) increments, the other groups held
           full, for every within-group scorer (random / activation_magnitude /
           hybrid / vision_attention). Measures task accuracy + first-token KL vs
           the no-pruning baseline. The per-group sensitivity curves and the best
           scorer fall out of this.

  validate Reads the per-task budgeting_sweep__<task>.json from a sweep dir,
           constructs candidate *joint* per-group budgets at target average
           keep-ratios (water-filling on the measured curves, best scorer), and
           runs them head-to-head against `uniform` and flat `global_topk`
           baselines on a disjoint held-out split, all at an equal retained-token
           count.

Stage A uses **zeroing** (the prune.py reconstruct-to-full-length contract): it does
not shorten the sequence, so there is no latency claim here — methods are compared at
equal *retained refinement rows* (paper.md §10 Experiment 3's fairness unit). Real
sequence-shortening + latency is Stage B (deferred).

Reuses the Phase-3 task registry/scorers (exp_sensitivity) and the shared generation
helpers (_genutil), so metrics are identical to exp_scoring.
"""

import argparse
import glob
import inspect
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from src.deepstack import budget
from src.deepstack.prune import SCORING_METHODS, DeepStackPruner, PruneSpec
from src.evaluate import Qwen3VLEvaluator
from src.experiments._genutil import (
    baseline_with_capture,
    generate,
    kl_first_token,
    num_visual_tokens,
)
from src.experiments.exp_sensitivity import _ANSWER_SUFFIX, TASKS, load_task_samples

# Default task set for the paper: general_vqa (insensitive -> uniform is ~free) plus
# the OCR/detail tasks where per-group budgeting can beat uniform (G2 OCR-critical).
_DEFAULT_TASKS = ("general_vqa", "docvqa", "textvqa")
_DEFAULT_SCORERS = ("random", "activation_magnitude", "hybrid", "vision_attention")
_DEFAULT_SWEEP_SAMPLES = 100
_DEFAULT_VALIDATE_SAMPLES = 300
_DEFAULT_MAX_NEW_TOKENS = 32
_DEFAULT_STEP = 0.05
_DEFAULT_TARGETS = (0.7, 0.5, 0.3)
_VISION = "vision_attention"


def _ratio_key(r: float) -> str:
    return f"{r:.2f}"


def sweep_filename(task: str) -> str:
    return f"budgeting_sweep__{task}.json"


def validate_filename(task: str) -> str:
    return f"budgeting_validation__{task}.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-group embed-norm capture (for the flat global-top-k baseline)
# ═══════════════════════════════════════════════════════════════════════════════


class EmbedNormCapturer:
    """Capture each DeepStack group's per-token L2 norm during the (prefill) forward.

    The global-top-k baseline needs a per-token scalar score across groups; the
    canonical flat signal is activation magnitude (L2 norm of each group's embeds).
    Those embeds only exist inside the forward, so we grab them once via a pre-hook
    on Qwen3VLTextModel (same kwarg the pruner reads), capturing the first occurrence
    (prefill) only.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._norms: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    def _hook(self, _module: Any, _args: Any, kwargs: Dict[str, Any]) -> None:
        embeds = kwargs.get("deepstack_visual_embeds")
        if not embeds:
            return
        for gi, e in enumerate(embeds):
            if gi not in self._norms:  # prefill only
                self._norms[gi] = e.detach().norm(dim=-1).float().cpu()

    def collect(self) -> Optional[Dict[int, torch.Tensor]]:
        return dict(self._norms) if self._norms else None

    def __enter__(self) -> "EmbedNormCapturer":
        for module in self.model.modules():
            if type(module).__name__ == "Qwen3VLTextModel":
                self._handles.append(module.register_forward_pre_hook(self._hook, with_kwargs=True))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared setup + sample loading
# ═══════════════════════════════════════════════════════════════════════════════


def _setup(model_id: str, device: Optional[str], dtype: Optional[str]) -> Tuple[Qwen3VLEvaluator, int, List[int], int]:
    evaluator = Qwen3VLEvaluator(model_id=model_id, device=device, dtype=dtype)
    vcfg = evaluator.model.config.vision_config
    vision_layers = list(vcfg.deepstack_visual_indexes)
    return evaluator, len(vision_layers), vision_layers, int(vcfg.spatial_merge_size)


def _build_inputs(evaluator: Qwen3VLEvaluator, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"] + _ANSWER_SUFFIX},
            ],
        }
    ]
    try:
        inputs = evaluator.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        return inputs.to(evaluator.device)
    except Exception as e:  # noqa: BLE001 — skip a bad sample, keep going
        print(f"  [preprocess failed: {type(e).__name__}: {e}]", flush=True)
        return None


def _load_split(task: str, n: int, skip: int) -> List[Dict[str, Any]]:
    """Load `n` samples for `task`, skipping the first `skip` usable rows (so the
    validate split is disjoint from the sweep's calibration split)."""
    spec = TASKS[task]
    samples = load_task_samples(spec, n + skip)
    return samples[skip : skip + n]


def _bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ═══════════════════════════════════════════════════════════════════════════════
#  sweep
# ═══════════════════════════════════════════════════════════════════════════════


def run_sweep(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    output_dir: str = "results",
    num_samples: int = _DEFAULT_SWEEP_SAMPLES,
    tasks: Optional[List[str]] = None,
    scorers: Optional[List[str]] = None,
    groups: Optional[List[int]] = None,
    step: float = _DEFAULT_STEP,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Dict[str, Any]]:
    """Run the independent per-group sweep for each task; one file per task.

    Returns {task: report}. All tasks share one model load, one pruner, one output
    timestamp dir.
    """
    evaluator, num_groups, vision_layers, merge_size = _setup(model_id, device, dtype)
    scorer_list = scorers or list(_DEFAULT_SCORERS)
    task_list = tasks or list(_DEFAULT_TASKS)
    group_list = groups if groups is not None else list(range(num_groups))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)

    reports: Dict[str, Dict[str, Any]] = {}
    with DeepStackPruner(evaluator.model) as pruner:
        for task in task_list:
            if task not in TASKS:
                print(f"[sweep] unknown task {task!r} — skipping", flush=True)
                continue
            try:
                rep = _sweep_one_task(
                    evaluator, pruner, task, out, num_groups, vision_layers, merge_size,
                    scorer_list, group_list, step, num_samples, max_new_tokens,
                )
                reports[task] = rep
            except Exception as e:  # noqa: BLE001 — a bad task must not kill the others
                print(f"[sweep] task {task} failed ({type(e).__name__}: {e}); continuing", flush=True)
    return reports


def _sweep_one_task(
    evaluator: Qwen3VLEvaluator,
    pruner: DeepStackPruner,
    task: str,
    out: Path,
    num_groups: int,
    vision_layers: List[int],
    merge_size: int,
    scorer_list: List[str],
    group_list: List[int],
    step: float,
    num_samples: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    need_vision = _VISION in scorer_list
    conditions = [
        (gi, s, r) for (gi, s, r) in budget.sweep_conditions(num_groups, scorer_list, step) if gi in group_list
    ]
    # group -> scorer -> ratio_key -> [values]
    acc: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    kl: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    baseline_acc: List[float] = []
    n_tokens_list: List[int] = []

    report_path = out / sweep_filename(task)
    spec = TASKS[task]
    samples = load_task_samples(spec, num_samples)
    print(f"[sweep] {task}: loaded {len(samples)} samples; {len(conditions)} conditions/sample", flush=True)

    scored = 0
    for si, sample in enumerate(samples):
        inputs = _build_inputs(evaluator, sample)
        if inputs is None:
            continue
        n_tokens = num_visual_tokens(inputs, merge_size)
        try:
            ans, base_logits, vision_scores = baseline_with_capture(
                evaluator, pruner, inputs, max_new_tokens, n_tokens, need_vision
            )
            pruner.set_vision_scores(vision_scores)
            base_acc = spec.scorer(ans, sample["answers"])
            for gi, scorer, ratio in conditions:
                if scorer == _VISION and vision_scores is None:
                    continue  # capture failed -> N/A this sample
                pruner.set_keep_indices(None)
                pruner.set_specs({gi: PruneSpec(scorer, ratio)})
                a, lg = generate(evaluator, inputs, max_new_tokens)
                acc[gi][scorer][_ratio_key(ratio)].append(spec.scorer(a, sample["answers"]))
                kl[gi][scorer][_ratio_key(ratio)].append(kl_first_token(base_logits, lg))
        except Exception as e:  # noqa: BLE001 — skip a failed sample, free memory
            print(f"  [{task} #{si}] failed ({type(e).__name__}: {e})", flush=True)
            if evaluator.device == "cuda":
                torch.cuda.empty_cache()
            continue

        baseline_acc.append(base_acc)
        if n_tokens is not None:
            n_tokens_list.append(n_tokens)
        scored += 1
        if scored % 5 == 0 or si == len(samples) - 1:
            print(
                f"  [sweep {task}] scored {scored}/{len(samples)}  baseline_acc={np.mean(baseline_acc):.3f}",
                flush=True,
            )
        _write_sweep(
            report_path, evaluator, task, vision_layers, scorer_list, step, num_groups,
            num_samples, max_new_tokens, scored, baseline_acc, n_tokens_list, acc, kl,
        )

    report = _write_sweep(
        report_path, evaluator, task, vision_layers, scorer_list, step, num_groups,
        num_samples, max_new_tokens, scored, baseline_acc, n_tokens_list, acc, kl,
    )
    print(f"[sweep] wrote {report_path}  (scored {scored})", flush=True)
    return report


def _mean_nested(d: Dict[int, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        str(gi): {s: {rk: float(np.mean(vals)) for rk, vals in by_r.items() if vals} for s, by_r in by_s.items()}
        for gi, by_s in d.items()
    }


def _write_sweep(
    path: Path,
    evaluator: Qwen3VLEvaluator,
    task: str,
    vision_layers: List[int],
    scorers: List[str],
    step: float,
    num_groups: int,
    requested: int,
    max_new_tokens: int,
    scored: int,
    baseline_acc: List[float],
    n_tokens_list: List[int],
    acc: Dict[int, Dict[str, Dict[str, List[float]]]],
    kl: Dict[int, Dict[str, Dict[str, List[float]]]],
) -> Dict[str, Any]:
    report = {
        "phase": "stageA_sweep",
        "task": task,
        "model_source": inspect.getfile(type(evaluator.model)),
        "device": evaluator.device,
        "dtype": evaluator.dtype,
        "num_groups": num_groups,
        "vision_layers": vision_layers,
        "scorers": scorers,
        "step": step,
        "requested_samples": requested,
        "samples": scored,
        "max_new_tokens": max_new_tokens,
        "mean_tokens_per_group": float(np.mean(n_tokens_list)) if n_tokens_list else None,
        "baseline_accuracy": float(np.mean(baseline_acc)) if baseline_acc else 0.0,
        "accuracy": _mean_nested(acc),
        "kl": _mean_nested(kl),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  validate
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_conditions(
    curves: Dict[Tuple[int, str], budget.Curve],
    scorers: List[str],
    num_groups: int,
    targets: List[float],
    mean_n: int,
    step: float,
    n_candidates: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build the list of validate conditions. Returns (best_scorer, conditions).

    Each condition: {key, type in {pergroup,uniform,global}, target, budget|None, scorer}.
    `global` budget is None (its keep-sets are computed per sample from embed norms).
    """
    best = budget.best_scorer(curves, scorers, num_groups) or scorers[0]
    per_scorer = budget.curves_for_scorer(curves, best)
    conds: List[Dict[str, Any]] = []
    for t in targets:
        cands = budget.waterfill_budgets(per_scorer, t, mean_n, step, n_candidates)
        for i, b in enumerate(cands):
            conds.append(
                {"key": f"pergroup@{t:.2f}#{i}", "type": "pergroup", "target": t, "budget": list(b), "scorer": best}
            )
        u = budget.uniform_budget(t, num_groups)
        conds.append(
            {"key": f"uniform@{t:.2f}", "type": "uniform", "target": t, "budget": list(u), "scorer": best}
        )
        conds.append(
            {"key": f"global@{t:.2f}", "type": "global", "target": t, "budget": None, "scorer": "activation_magnitude"}
        )
    return best, conds


def _find_sweep_dir(output_dir: str, tasks: List[str]) -> Optional[Path]:
    """Latest results/<ts> dir that contains a budgeting_sweep__<task>.json for any task."""
    candidates = set()
    for task in tasks:
        for p in glob.glob(f"{output_dir}/*/{sweep_filename(task)}"):
            candidates.add(str(Path(p).parent))
    if not candidates:
        return None
    return Path(sorted(candidates)[-1])


def run_validate(
    model_id: str,
    sweep_dir: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    output_dir: str = "results",
    num_samples: int = _DEFAULT_VALIDATE_SAMPLES,
    tasks: Optional[List[str]] = None,
    sample_offset: Optional[int] = None,
    targets: Optional[List[float]] = None,
    n_candidates: int = 3,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, Dict[str, Any]]:
    """Validate joint budgets for each task; one file per task. Reads each task's
    budgeting_sweep__<task>.json from `sweep_dir` (auto = latest if omitted)."""
    task_list = tasks or list(_DEFAULT_TASKS)
    sdir = Path(sweep_dir) if sweep_dir else _find_sweep_dir(output_dir, task_list)
    if sdir is None:
        raise RuntimeError("no budgeting_sweep__<task>.json found — run `sweep` first or pass --sweep-dir")
    evaluator, num_groups, vision_layers, merge_size = _setup(model_id, device, dtype)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)
    target_list = targets or list(_DEFAULT_TARGETS)

    reports: Dict[str, Dict[str, Any]] = {}
    with DeepStackPruner(evaluator.model) as pruner:
        for task in task_list:
            sweep_path = sdir / sweep_filename(task)
            if not sweep_path.exists():
                print(f"[validate] no sweep for task {task} at {sweep_path} — skipping", flush=True)
                continue
            try:
                rep = _validate_one_task(
                    evaluator, pruner, task, str(sweep_path), out, num_groups, vision_layers,
                    merge_size, target_list, num_samples, sample_offset, n_candidates, max_new_tokens,
                )
                reports[task] = rep
            except Exception as e:  # noqa: BLE001 — a bad task must not kill the others
                print(f"[validate] task {task} failed ({type(e).__name__}: {e}); continuing", flush=True)
    return reports


def _validate_one_task(
    evaluator: Qwen3VLEvaluator,
    pruner: DeepStackPruner,
    task: str,
    sweep_path: str,
    out: Path,
    num_groups: int,
    vision_layers: List[int],
    merge_size: int,
    target_list: List[float],
    num_samples: int,
    sample_offset: Optional[int],
    n_candidates: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    with open(sweep_path, encoding="utf-8") as f:
        sweep = json.load(f)
    scorers = list(sweep.get("scorers", _DEFAULT_SCORERS))
    step = float(sweep.get("step", _DEFAULT_STEP))
    mean_n = int(round(sweep.get("mean_tokens_per_group") or 0))
    curves, _ = budget.extract_curves(sweep)
    best, conds = _validate_conditions(curves, scorers, num_groups, target_list, mean_n, step, n_candidates)
    need_vision = best == _VISION
    print(f"[validate {task}] best_scorer={best}; {len(conds)} conditions", flush=True)

    skip = sample_offset if sample_offset is not None else int(sweep.get("samples", 0))
    samples = _load_split(task, num_samples, skip)
    print(f"[validate {task}] loaded {len(samples)} held-out samples (skip={skip})", flush=True)

    acc: Dict[str, List[float]] = defaultdict(list)
    kl: Dict[str, List[float]] = defaultdict(list)
    retained: Dict[str, List[int]] = defaultdict(list)
    baseline_acc: List[float] = []
    report_path = out / validate_filename(task)
    spec = TASKS[task]
    scored = 0

    for si, sample in enumerate(samples):
        inputs = _build_inputs(evaluator, sample)
        if inputs is None:
            continue
        n_tokens = num_visual_tokens(inputs, merge_size)
        try:
            with EmbedNormCapturer(evaluator.model) as ecap:
                ans, base_logits, vision_scores = baseline_with_capture(
                    evaluator, pruner, inputs, max_new_tokens, n_tokens, need_vision
                )
            norms = ecap.collect()
            pruner.set_vision_scores(vision_scores)
            base_acc = spec.scorer(ans, sample["answers"])

            n = n_tokens or mean_n
            for cond in conds:
                if cond["scorer"] == _VISION and vision_scores is None:
                    continue
                pruner.set_keep_indices(None)
                if cond["type"] == "global":
                    if norms is None:
                        continue
                    k = round(cond["target"] * num_groups * n)
                    keepsets = budget.global_topk_keepsets(norms, k)
                    pruner.set_specs({})
                    pruner.set_keep_indices(keepsets)
                    ret = int(sum(v.numel() for v in keepsets.values()))
                else:
                    b = cond["budget"]
                    pruner.set_specs({gi: PruneSpec(cond["scorer"], float(b[gi])) for gi in range(num_groups)})
                    ret = budget.retained_total([float(x) for x in b], n)
                a, lg = generate(evaluator, inputs, max_new_tokens)
                acc[cond["key"]].append(spec.scorer(a, sample["answers"]))
                kl[cond["key"]].append(kl_first_token(base_logits, lg))
                retained[cond["key"]].append(ret)
        except Exception as e:  # noqa: BLE001
            print(f"  [{task} #{si}] failed ({type(e).__name__}: {e})", flush=True)
            if evaluator.device == "cuda":
                torch.cuda.empty_cache()
            continue

        baseline_acc.append(base_acc)
        scored += 1
        if scored % 10 == 0 or si == len(samples) - 1:
            print(f"  [validate {task}] scored {scored}/{len(samples)}", flush=True)
        _write_validate(
            report_path, evaluator, task, sweep_path, best, vision_layers, target_list,
            num_groups, mean_n, num_samples, skip, scored, conds, baseline_acc, acc, kl, retained,
        )

    report = _write_validate(
        report_path, evaluator, task, sweep_path, best, vision_layers, target_list,
        num_groups, mean_n, num_samples, skip, scored, conds, baseline_acc, acc, kl, retained,
    )
    _print_validate(report)
    print(f"[validate] wrote {report_path}  (scored {scored})", flush=True)
    return report


def _write_validate(
    path: Path,
    evaluator: Qwen3VLEvaluator,
    task: str,
    sweep_path: str,
    best: str,
    vision_layers: List[int],
    targets: List[float],
    num_groups: int,
    mean_n: int,
    requested: int,
    skip: int,
    scored: int,
    conds: List[Dict[str, Any]],
    baseline_acc: List[float],
    acc: Dict[str, List[float]],
    kl: Dict[str, List[float]],
    retained: Dict[str, List[int]],
) -> Dict[str, Any]:
    report = {
        "phase": "stageA_validate",
        "task": task,
        "sweep_source": sweep_path,
        "model_source": inspect.getfile(type(evaluator.model)),
        "device": evaluator.device,
        "dtype": evaluator.dtype,
        "best_scorer": best,
        "num_groups": num_groups,
        "vision_layers": vision_layers,
        "mean_tokens_per_group": mean_n,
        "targets": targets,
        "requested_samples": requested,
        "sample_offset": skip,
        "samples": scored,
        "baseline_accuracy": float(np.mean(baseline_acc)) if baseline_acc else 0.0,
        "conditions": {c["key"]: {k: c[k] for k in ("type", "target", "budget", "scorer")} for c in conds},
        "accuracy": {k: float(np.mean(v)) for k, v in acc.items() if v},
        "accuracy_ci": {k: list(_bootstrap_ci(v)) for k, v in acc.items() if v},
        "kl": {k: float(np.mean(v)) for k, v in kl.items() if v},
        "retained_tokens": {k: float(np.mean(v)) for k, v in retained.items() if v},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _print_validate(r: Dict[str, Any]) -> None:
    print(f"\n=== Stage-A Budget Validation ({r['task']}) ===")
    print(f"best_scorer={r['best_scorer']}  baseline_acc={r['baseline_accuracy']:.3f}  samples={r['samples']}")
    for key in r["accuracy"]:
        c = r["conditions"][key]
        ci = r["accuracy_ci"].get(key, [float("nan"), float("nan")])
        print(
            f"  {key:18s} acc={r['accuracy'][key]:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
            f"kl={r['kl'].get(key, float('nan')):.3f}  retained={r['retained_tokens'].get(key, 0):.0f}  "
            f"budget={c['budget']}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-group DeepStack budgeting (Phase 5, Stage A).")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    common.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps", "auto"])
    common.add_argument("--dtype", type=str, default=None, choices=["float16", "float32", "auto"])
    common.add_argument("--output-dir", type=str, default="results")
    common.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    common.add_argument(
        "--tasks", type=str, default=None,
        help=f"comma-separated subset of {list(TASKS.keys())}; default = {list(_DEFAULT_TASKS)}",
    )

    sp = sub.add_parser("sweep", parents=[common], help="independent per-group sweep")
    sp.add_argument("--num-samples", type=int, default=_DEFAULT_SWEEP_SAMPLES)
    sp.add_argument("--scorers", type=str, default=None, help=f"subset of {list(SCORING_METHODS)}")
    sp.add_argument("--groups", type=str, default=None, help="comma-separated group indices; default all")
    sp.add_argument("--step", type=float, default=_DEFAULT_STEP)

    vp = sub.add_parser("validate", parents=[common], help="validate joint budgets vs baselines")
    vp.add_argument(
        "--sweep-dir", type=str, default=None,
        help="results/<ts> dir with budgeting_sweep__*.json; default = latest",
    )
    vp.add_argument("--num-samples", type=int, default=_DEFAULT_VALIDATE_SAMPLES)
    vp.add_argument("--sample-offset", type=int, default=None, help="held-out skip; default = sweep samples")
    vp.add_argument("--targets", type=str, default=None, help="comma-separated avg keep-ratios; default 0.7,0.5,0.3")
    vp.add_argument("--n-candidates", type=int, default=3)

    args = parser.parse_args()
    device = None if args.device in (None, "auto") else args.device
    dtype = None if args.dtype in (None, "auto") else args.dtype
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    if args.mode == "sweep":
        scorers = [s.strip() for s in args.scorers.split(",")] if args.scorers else None
        groups = [int(x) for x in args.groups.split(",")] if args.groups else None
        run_sweep(
            model_id=args.model_id, device=device, dtype=dtype, output_dir=args.output_dir,
            num_samples=args.num_samples, tasks=tasks, scorers=scorers, groups=groups, step=args.step,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        targets = [float(x) for x in args.targets.split(",")] if args.targets else None
        run_validate(
            model_id=args.model_id, sweep_dir=args.sweep_dir, device=device, dtype=dtype,
            output_dir=args.output_dir, num_samples=args.num_samples, tasks=tasks,
            sample_offset=args.sample_offset, targets=targets, n_candidates=args.n_candidates,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
