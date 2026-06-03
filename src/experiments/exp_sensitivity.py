"""
DeepStack group sensitivity ablation (Phase 3 / Experiment 2).

The go/no-go experiment of paper.md: ablate each DeepStack visual feature group
and measure the per-task accuracy drop. If different groups cause different
drops, per-depth budgeting is justified; if all groups behave the same, the
method needs rethinking (paper.md section 16).

For each task and each of the 8 ablation conditions (full, drop_g{i}, keep_g{i},
drop_all) it measures, over a small labeled subset:

  - labeled accuracy   -> the headline Figure-3 signal (VQA soft-accuracy / ANLS
                          / integer exact-match, per task)
  - first-token KL      -> a dense, reference-free signal: KL(P_full || P_cond) of
                          the first generated-token distribution vs full DeepStack

Ablation is done by DeepStackAblator (src/deepstack/ablation.py), a forward hook
that zeros a group's additive embeds — no model-source edits. Both signals come
from a single greedy generate() per condition (return_dict_in_generate gives the
answer and the first-token logits together).

Robustness mirrors instrument.py: dataset images are size-capped (<=1024px) to
avoid T4 OOM, a bad sample is skipped rather than killing the run, and a task
whose dataset fails to load is skipped with a diagnostic.

Output: results/<timestamp>/sensitivity.json

Run on Colab via colab_run.ipynb (RUN_SENSITIVITY=True), never locally.
"""

import argparse
import inspect
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

# Importing evaluate performs the sys.modules redirect to local_transformers
# before any transformers import. Reuse its loader and the image size-cap helper.
from src.deepstack.ablation import AblationCondition, DeepStackAblator, standard_conditions
from src.deepstack.instrument import _cap_image
from src.evaluate import Qwen3VLEvaluator

_DEFAULT_NUM_SAMPLES = 100
_DEFAULT_MAX_NEW_TOKENS = 32
_ANSWER_SUFFIX = "\nAnswer the question using a single word or phrase."
_SCAN_LIMIT_MULT = 40  # for filtered tasks (counting), cap rows scanned at mult*N


# ═══════════════════════════════════════════════════════════════════════════════
#  Answer normalization + scorers (pure Python, no heavy deps)
# ═══════════════════════════════════════════════════════════════════════════════

_ARTICLES = {"a", "an", "the"}
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_CONTRACTIONS = {"dont": "don't", "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_answer(ans: str) -> str:
    """Canonical VQA-style normalization: lowercase, strip punctuation/articles,
    map number words to digits, collapse whitespace."""
    text = ans.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = []
    for tok in _WS_RE.sub(" ", text).strip().split(" "):
        if not tok or tok in _ARTICLES:
            continue
        tok = _NUM_WORDS.get(tok, tok)
        tok = _CONTRACTIONS.get(tok, tok)
        tokens.append(tok)
    return " ".join(tokens)


def vqa_soft_accuracy(pred: str, golds: List[str]) -> float:
    """VQA accuracy: min(1, matches/3) over the (typically 10) annotator answers."""
    p = normalize_answer(pred)
    matches = sum(1 for g in golds if normalize_answer(g) == p)
    return min(1.0, matches / 3.0)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: List[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA metric), best over golds."""
    p = pred.strip().lower()
    best = 0.0
    for g in golds:
        gg = g.strip().lower()
        denom = max(len(p), len(gg))
        sim = 1.0 - (_levenshtein(p, gg) / denom) if denom else 1.0
        best = max(best, sim if sim >= tau else 0.0)
    return best


def _first_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def count_exact(pred: str, golds: List[str]) -> float:
    """Integer exact-match: pred's first integer must equal any gold integer."""
    pi = _first_int(pred)
    if pi is None:
        return 0.0
    gold_ints = {gi for g in golds if (gi := _first_int(g)) is not None}
    return 1.0 if pi in gold_ints else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Task registry
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TaskSpec:
    """A sensitivity-eval task: where to load it and how to score it.

    Dataset ids are best-effort and may need adjustment on the first Colab run
    (swap any that are gated/renamed). ``row_filter`` selects a subset of rows
    (used for the counting proxy = VQAv2 "how many" questions).
    """

    name: str
    hf_id: str
    config: Optional[str]
    split: str
    scorer: Callable[[str, List[str]], float]
    row_filter: Optional[Callable[[Dict[str, Any]], bool]] = None


def _answers_to_strings(row: Dict[str, Any]) -> List[str]:
    """Normalize the many 'answers' shapes (list[str] / list[dict] / str)."""
    raw = row.get("answers")
    if raw is None:
        raw = row.get("answer")
    if raw is None:
        mc = row.get("multiple_choice_answer")
        return [str(mc)] if mc is not None else []
    if isinstance(raw, str):
        return [raw]
    out: List[str] = []
    for a in raw:
        if isinstance(a, dict):
            out.append(str(a.get("answer", "")))
        else:
            out.append(str(a))
    return out


def _is_counting(row: Dict[str, Any]) -> bool:
    return str(row.get("question", "")).strip().lower().startswith("how many")


TASKS: Dict[str, TaskSpec] = {
    "general_vqa": TaskSpec("general_vqa", "lmms-lab/VQAv2", None, "validation", vqa_soft_accuracy),
    "textvqa": TaskSpec("textvqa", "lmms-lab/textvqa", None, "validation", vqa_soft_accuracy),
    "docvqa": TaskSpec("docvqa", "lmms-lab/DocVQA", "DocVQA", "validation", anls),
    "counting": TaskSpec(
        "counting", "lmms-lab/VQAv2", None, "validation", count_exact, row_filter=_is_counting
    ),
}


def _iter_task_rows(spec: TaskSpec) -> Iterator[Dict[str, Any]]:
    """Stream a dataset (no full download) and yield rows passing the filter."""
    from datasets import load_dataset  # local import: optional heavy dep

    ds = load_dataset(spec.hf_id, spec.config, split=spec.split, streaming=True)
    for row in ds:
        if spec.row_filter is None or spec.row_filter(row):
            yield row


def load_task_samples(spec: TaskSpec, n: int) -> List[Dict[str, Any]]:
    """Collect up to ``n`` usable {image, question, answers} samples for a task."""
    samples: List[Dict[str, Any]] = []
    scanned = 0
    scan_cap = n * _SCAN_LIMIT_MULT if spec.row_filter is not None else n * 4
    for row in _iter_task_rows(spec):
        scanned += 1
        if scanned > scan_cap:
            break
        image = row.get("image")
        question = row.get("question")
        answers = _answers_to_strings(row)
        if image is None or not question or not answers:
            continue
        samples.append(
            {"image": _cap_image(image.convert("RGB")), "question": str(question), "answers": answers}
        )
        if len(samples) >= n:
            break
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
#  Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SensitivityReport:
    model_source: str
    device: str
    dtype: str
    num_groups: int
    vision_layers: List[int]
    conditions: List[str]
    tasks: List[str]
    requested_samples: int
    max_new_tokens: int
    samples_per_task: Dict[str, int] = field(default_factory=dict)
    # accuracy[task][condition] and accuracy_drop[task][condition] (full - cond)
    accuracy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    accuracy_drop: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # mean first-token KL(P_full || P_cond), pooled over all scored samples
    kl: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-sample evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_under_condition(
    evaluator: Qwen3VLEvaluator,
    inputs: Dict[str, Any],
    max_new_tokens: int,
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
#  Driver
# ═══════════════════════════════════════════════════════════════════════════════


def run_sensitivity(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    output_dir: str = "results",
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    tasks: Optional[List[str]] = None,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> SensitivityReport:
    evaluator = Qwen3VLEvaluator(model_id=model_id, device=device, dtype=dtype)
    model = evaluator.model
    processor = evaluator.processor

    vision_cfg = model.config.vision_config
    vision_layers: List[int] = list(vision_cfg.deepstack_visual_indexes)
    conditions = standard_conditions(len(vision_layers))
    task_names = tasks or list(TASKS.keys())

    acc_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    kl_acc: Dict[str, List[float]] = defaultdict(list)
    samples_per_task: Dict[str, int] = {}

    with DeepStackAblator(model) as ablator:
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
                    for cond in conditions:
                        ablator.set_condition(cond)
                        ans, logits = _generate_under_condition(evaluator, inputs, max_new_tokens)
                        answer_by_cond[cond.name] = ans
                        logits_by_cond[cond.name] = logits
                except Exception as e:  # noqa: BLE001 — skip a failed forward, free memory
                    print(f"  [{task_name} #{si}] generation failed ({type(e).__name__}: {e})", flush=True)
                    if evaluator.device == "cuda":
                        torch.cuda.empty_cache()
                    continue

                for cond in conditions:
                    acc_acc[task_name][cond.name].append(
                        spec.scorer(answer_by_cond[cond.name], sample["answers"])
                    )
                full_logits = logits_by_cond["full"]
                for cond in conditions:
                    kl_acc[cond.name].append(_kl_first_token(full_logits, logits_by_cond[cond.name]))
                scored += 1
            samples_per_task[task_name] = scored
            if scored:
                _print_task_line(task_name, acc_acc[task_name])

    report = _finalize(
        evaluator, vision_layers, conditions, task_names, num_samples, max_new_tokens,
        samples_per_task, acc_acc, kl_acc,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "sensitivity.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    _print_summary(report, report_path)
    return report


def _finalize(
    evaluator: Qwen3VLEvaluator,
    vision_layers: List[int],
    conditions: List[AblationCondition],
    task_names: List[str],
    num_samples: int,
    max_new_tokens: int,
    samples_per_task: Dict[str, int],
    acc_acc: Dict[str, Dict[str, List[float]]],
    kl_acc: Dict[str, List[float]],
) -> SensitivityReport:
    cond_names = [c.name for c in conditions]
    accuracy: Dict[str, Dict[str, float]] = {}
    accuracy_drop: Dict[str, Dict[str, float]] = {}
    for task_name in acc_acc:
        per_cond = {cn: float(np.mean(acc_acc[task_name][cn])) for cn in cond_names if acc_acc[task_name][cn]}
        accuracy[task_name] = per_cond
        full_acc = per_cond.get("full", 0.0)
        accuracy_drop[task_name] = {cn: full_acc - v for cn, v in per_cond.items()}
    kl = {cn: float(np.mean(kl_acc[cn])) for cn in cond_names if kl_acc[cn]}

    return SensitivityReport(
        model_source=inspect.getfile(type(evaluator.model)),
        device=evaluator.device,
        dtype=evaluator.dtype,
        num_groups=len(vision_layers),
        vision_layers=vision_layers,
        conditions=cond_names,
        tasks=task_names,
        requested_samples=num_samples,
        max_new_tokens=max_new_tokens,
        samples_per_task=samples_per_task,
        accuracy=accuracy,
        accuracy_drop=accuracy_drop,
        kl=kl,
    )


def _print_task_line(task_name: str, per_cond: Dict[str, List[float]]) -> None:
    parts = [f"{cn}={np.mean(vals):.3f}" for cn, vals in per_cond.items() if vals]
    print(f"  [{task_name}] " + "  ".join(parts), flush=True)


def _print_summary(r: SensitivityReport, path: Path) -> None:
    print("\n=== DeepStack Sensitivity Report (Phase 3) ===")
    print(f"Model source: {r.model_source}")
    print(f"Groups: {r.num_groups} (vision layers {r.vision_layers})  device={r.device} dtype={r.dtype}")
    print(f"Samples/task: {r.samples_per_task}")
    print("\nAccuracy drop vs full (higher = group matters more for that task):")
    for task_name in r.accuracy_drop:
        drops = r.accuracy_drop[task_name]
        line = "  ".join(f"{cn}={drops[cn]:+.3f}" for cn in r.conditions if cn in drops)
        print(f"  {task_name:12s} {line}")
    print("\nFirst-token KL(P_full||P_cond):")
    print("  " + "  ".join(f"{cn}={r.kl[cn]:.3f}" for cn in r.conditions if cn in r.kl))
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepStack group sensitivity ablation (Phase 3).")
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
    args = parser.parse_args()

    device = None if args.device in (None, "auto") else args.device
    dtype = None if args.dtype in (None, "auto") else args.dtype
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    run_sensitivity(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        tasks=tasks,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
