"""
Within-group scoring comparison (Phase 4 / Experiment 4).

Compares the five within-group token-scoring strategies of paper.md section 6
(Level 2) at several keep-ratios, using a *uniform* budget across all DeepStack
groups (the same keep-ratio for every group). Holding the budget uniform isolates
the question "which scorer keeps the right tokens?" from the separate question
"how should the budget be split across groups?" (that split is Phase 5).

For each (task, method, keep_ratio) it measures, over a labeled subset:

  - labeled accuracy   -> the headline signal (VQA soft-accuracy / ANLS / integer
                          exact-match, per task; same scorers as Phase 3)
  - first-token KL      -> dense reference-free signal: KL(P_full || P_pruned) of
                          the first generated-token distribution vs no-pruning

Pruning is done by DeepStackPruner (src/deepstack/prune.py), a forward hook that
replaces each group's embeds with its reconstruct-to-full-length pruned version —
no model-source edits. keep_ratio=1.0 is the shared "full" baseline (run once; all
methods are identical there), so the comparison is reported as accuracy *drop*
from that baseline at each pruning level.

Task loading, image size-capping, answer normalization and the per-task scorers
are reused from exp_sensitivity to keep the two experiments metric-compatible.

Output: results/<timestamp>/scoring.json

Run on Colab via colab_run.ipynb (RUN_SCORING=True), never locally.
"""

import argparse
import inspect
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Reuse Phase 3 task registry + scorers (importing it performs the
# local_transformers redirect via src.evaluate) and the prune hook.
from src.deepstack.prune import SCORING_METHODS, DeepStackPruner, PruneSpec
from src.evaluate import Qwen3VLEvaluator
from src.experiments.exp_sensitivity import _ANSWER_SUFFIX, TASKS, load_task_samples

_DEFAULT_NUM_SAMPLES = 100
_DEFAULT_MAX_NEW_TOKENS = 32
_DEFAULT_KEEP_RATIOS = (1.0, 0.75, 0.50, 0.25)
_DEFAULT_METHODS = SCORING_METHODS  # all five
_BASELINE_KEY = "full"  # keep_ratio == 1.0 condition


def _cond_key(method: str, keep_ratio: float) -> str:
    """Stable per-condition key, e.g. 'hybrid@0.50'. keep_ratio 1.0 -> 'full'."""
    if keep_ratio >= 1.0:
        return _BASELINE_KEY
    return f"{method}@{keep_ratio:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Generation under a pruning condition
# ═══════════════════════════════════════════════════════════════════════════════


def _generate(
    evaluator: Qwen3VLEvaluator, inputs: Dict[str, Any], max_new_tokens: int
) -> Tuple[str, torch.Tensor]:
    """Greedy generate; return (decoded answer, first-token logits as float vec)."""
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


def _kl_first_token(logits_full: torch.Tensor, logits_cond: torch.Tensor) -> float:
    """KL(P_full || P_cond) over the first-token distribution."""
    logp_full = torch.log_softmax(logits_full, dim=-1)
    logp_cond = torch.log_softmax(logits_cond, dim=-1)
    p_full = logp_full.exp()
    return float((p_full * (logp_full - logp_cond)).sum())


# ═══════════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScoringReport:
    model_source: str
    device: str
    dtype: str
    num_groups: int
    vision_layers: List[int]
    budget_mode: str  # "uniform_all_groups"
    methods: List[str]
    keep_ratios: List[float]
    tasks: List[str]
    requested_samples: int
    max_new_tokens: int
    samples_per_task: Dict[str, int] = field(default_factory=dict)
    # accuracy[task][cond_key]; cond_key is 'full' or '<method>@<ratio>'
    accuracy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    accuracy_drop: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # mean first-token KL(P_full || P_cond), pooled over all scored samples, per task
    kl: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════════════


def _build_conditions(
    methods: List[str], keep_ratios: List[float]
) -> List[Tuple[str, str, float]]:
    """List of (cond_key, method, keep_ratio). The baseline (ratio 1.0) appears once."""
    conditions: List[Tuple[str, str, float]] = []
    if any(r >= 1.0 for r in keep_ratios):
        conditions.append((_BASELINE_KEY, methods[0], 1.0))
    for r in keep_ratios:
        if r >= 1.0:
            continue
        for m in methods:
            conditions.append((_cond_key(m, r), m, r))
    return conditions


def run_scoring(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    output_dir: str = "results",
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    tasks: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    keep_ratios: Optional[List[float]] = None,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> ScoringReport:
    evaluator = Qwen3VLEvaluator(model_id=model_id, device=device, dtype=dtype)
    model = evaluator.model
    processor = evaluator.processor

    vision_cfg = model.config.vision_config
    vision_layers: List[int] = list(vision_cfg.deepstack_visual_indexes)
    num_groups = len(vision_layers)

    task_names = tasks or list(TASKS.keys())
    method_list = methods or list(_DEFAULT_METHODS)
    ratio_list = keep_ratios or list(_DEFAULT_KEEP_RATIOS)
    conditions = _build_conditions(method_list, ratio_list)

    acc_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    kl_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    samples_per_task: Dict[str, int] = {}

    with DeepStackPruner(model) as pruner:
        for task_name in task_names:
            spec = TASKS.get(task_name)
            if spec is None:
                print(f"[task {task_name}] unknown task — skipping", flush=True)
                continue
            try:
                samples = load_task_samples(spec, num_samples)
            except Exception as e:  # noqa: BLE001 — a bad/gated dataset must not kill the run
                print(f"[task {task_name}] dataset load failed ({type(e).__name__}: {e}); skipping", flush=True)
                continue
            print(f"[task {task_name}] loaded {len(samples)} samples", flush=True)

            scored = 0
            for si, sample in enumerate(samples):
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
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    )
                    inputs.pop("token_type_ids", None)
                    inputs = inputs.to(evaluator.device)
                except Exception as e:  # noqa: BLE001 — skip a bad sample, keep going
                    print(f"  [{task_name} #{si}] preprocess failed ({type(e).__name__}: {e})", flush=True)
                    continue

                try:
                    logits_by_cond: Dict[str, torch.Tensor] = {}
                    answer_by_cond: Dict[str, str] = {}
                    for cond_key, method, ratio in conditions:
                        if ratio >= 1.0:
                            pruner.set_specs({})  # baseline: no pruning
                        else:
                            pruner.set_specs(
                                {gi: PruneSpec(method, ratio) for gi in range(num_groups)}
                            )
                        ans, logits = _generate(evaluator, inputs, max_new_tokens)
                        answer_by_cond[cond_key] = ans
                        logits_by_cond[cond_key] = logits
                except Exception as e:  # noqa: BLE001 — skip a failed forward, free memory
                    print(f"  [{task_name} #{si}] generation failed ({type(e).__name__}: {e})", flush=True)
                    if evaluator.device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                for cond_key, _m, _r in conditions:
                    acc_acc[task_name][cond_key].append(
                        spec.scorer(answer_by_cond[cond_key], sample["answers"])
                    )
                full_logits = logits_by_cond[_BASELINE_KEY]
                for cond_key, _m, _r in conditions:
                    kl_acc[task_name][cond_key].append(
                        _kl_first_token(full_logits, logits_by_cond[cond_key])
                    )
                scored += 1

            samples_per_task[task_name] = scored
            if scored:
                _print_task_line(task_name, acc_acc[task_name])

    report = _finalize(
        evaluator, vision_layers, method_list, ratio_list, task_names,
        num_samples, max_new_tokens, samples_per_task, acc_acc, kl_acc,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "scoring.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    _print_summary(report, report_path)
    return report


def _finalize(
    evaluator: Qwen3VLEvaluator,
    vision_layers: List[int],
    methods: List[str],
    keep_ratios: List[float],
    task_names: List[str],
    num_samples: int,
    max_new_tokens: int,
    samples_per_task: Dict[str, int],
    acc_acc: Dict[str, Dict[str, List[float]]],
    kl_acc: Dict[str, Dict[str, List[float]]],
) -> ScoringReport:
    accuracy: Dict[str, Dict[str, float]] = {}
    accuracy_drop: Dict[str, Dict[str, float]] = {}
    kl: Dict[str, Dict[str, float]] = {}
    for task_name in acc_acc:
        per_cond = {ck: float(np.mean(vals)) for ck, vals in acc_acc[task_name].items() if vals}
        accuracy[task_name] = per_cond
        base = per_cond.get(_BASELINE_KEY, 0.0)
        accuracy_drop[task_name] = {ck: base - v for ck, v in per_cond.items()}
        kl[task_name] = {ck: float(np.mean(vals)) for ck, vals in kl_acc[task_name].items() if vals}

    return ScoringReport(
        model_source=inspect.getfile(type(evaluator.model)),
        device=evaluator.device,
        dtype=evaluator.dtype,
        num_groups=len(vision_layers),
        vision_layers=vision_layers,
        budget_mode="uniform_all_groups",
        methods=methods,
        keep_ratios=keep_ratios,
        tasks=task_names,
        requested_samples=num_samples,
        max_new_tokens=max_new_tokens,
        samples_per_task=samples_per_task,
        accuracy=accuracy,
        accuracy_drop=accuracy_drop,
        kl=kl,
    )


def _print_task_line(task_name: str, per_cond: Dict[str, List[float]]) -> None:
    parts = [f"{ck}={np.mean(vals):.3f}" for ck, vals in per_cond.items() if vals]
    print(f"  [{task_name}] " + "  ".join(parts), flush=True)


def _print_summary(r: ScoringReport, path: Path) -> None:
    print("\n=== DeepStack Within-Group Scoring Report (Phase 4) ===")
    print(f"Model source: {r.model_source}")
    print(f"Groups: {r.num_groups} (vision layers {r.vision_layers})  device={r.device} dtype={r.dtype}")
    print(f"Budget: {r.budget_mode}  methods={r.methods}  keep_ratios={r.keep_ratios}")
    print(f"Samples/task: {r.samples_per_task}")
    print("\nAccuracy drop vs full (lower = scorer preserves more at that keep-ratio):")
    for task_name in r.accuracy_drop:
        drops = r.accuracy_drop[task_name]
        ordered = [ck for ck in drops if ck != _BASELINE_KEY]
        line = "  ".join(f"{ck}={drops[ck]:+.3f}" for ck in ordered)
        print(f"  {task_name:12s} full={r.accuracy[task_name].get(_BASELINE_KEY, float('nan')):.3f}  {line}")
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepStack within-group scoring comparison (Phase 4).")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps", "auto"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "float32", "auto"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-samples", type=int, default=_DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--tasks", type=str, default=None,
        help=f"Comma-separated subset of {list(TASKS.keys())}; default = all.",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help=f"Comma-separated subset of {list(_DEFAULT_METHODS)}; default = all.",
    )
    parser.add_argument(
        "--keep-ratios", type=str, default=None,
        help="Comma-separated keep ratios in (0,1]; default = 1.0,0.75,0.50,0.25.",
    )
    args = parser.parse_args()

    device = None if args.device in (None, "auto") else args.device
    dtype = None if args.dtype in (None, "auto") else args.dtype
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    keep_ratios = (
        [float(x) for x in args.keep_ratios.split(",")] if args.keep_ratios else None
    )
    run_scoring(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        tasks=tasks,
        methods=methods,
        keep_ratios=keep_ratios,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
