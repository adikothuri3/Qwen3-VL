"""
Qwen3-VL Evaluation & Profiling Script

Loads the model from local_transformers/ so any edits to the model source are
immediately reflected at runtime. On startup, the script prints the exact file
the model class was loaded from — use this to confirm local code is active.

Each run produces a timestamped directory under --output-dir containing:

  Evaluation outputs:
    - evaluation_metrics.json   (per-sample timing & output data)
    - evaluation_summary.json   (aggregate statistics)

  Profiling outputs:
    - profile_data.json         (component-level timing data)
    - profile_chart.png         (5-panel matplotlib visualisation)
    - trace_<timestamp>.json    (Chrome trace, viewable at chrome://tracing)
    - tensorboard/              (optional, if --tensorboard is passed)
"""

import sys
from pathlib import Path

# ── local_transformers redirect ───────────────────────────────────────────────
# Must happen before any other import that touches transformers.
# Imports the model class directly from the local source file so edits are
# guaranteed to be reflected without needing to reinstall anything.
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

import local_transformers
sys.modules["transformers"] = local_transformers

# Direct import from local source — this is what makes your edits take effect.
from local_transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from local_transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

import inspect
_model_source = inspect.getfile(Qwen3VLForConditionalGeneration)
print(f"[local_transformers] Model loaded from: {_model_source}")

# ── standard library & third-party imports ────────────────────────────────────
import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Data classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run."""

    sample_id: str
    preprocessing_time: float
    generation_time: float
    decoding_time: float
    total_time: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    max_new_tokens: int
    temperature: float
    output_text: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary statistics for an evaluation run."""

    total_samples: int
    successful_samples: int
    failed_samples: int
    avg_preprocessing_time: float
    avg_generation_time: float
    avg_decoding_time: float
    avg_total_time: float
    avg_tokens_per_second: float
    total_tokens_generated: int
    total_time: float
    median_generation_time: float
    p95_generation_time: float
    p99_generation_time: float


@dataclass
class ProfileResult:
    """All data gathered from one profiling run."""

    component_timings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preprocessing_ms: float = 0.0
    generation_ms: float = 0.0
    decoding_ms: float = 0.0
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    torch_profiler_key_averages: Optional[str] = None
    trace_path: Optional[str] = None


@dataclass
class ComponentTiming:
    """Accumulated timing for one named module."""

    name: str
    total_ms: float = 0.0
    call_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Evaluator
# ═══════════════════════════════════════════════════════════════════════════════


class Qwen3VLEvaluator:
    """Loads and evaluates a Qwen3-VL model using local_transformers source."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or ("float16" if self.device == "cuda" else "float32")
        self.device_map = device_map

        print("=== Initializing Qwen3-VL Evaluator ===")
        print(f"Model ID:  {model_id}")
        print(f"Device:    {self.device}")
        print(f"Dtype:     {self.dtype}")

        self.model: Any = None
        self.processor: Any = None
        self._load_model()
        self._load_processor()

    def _load_model(self) -> None:
        print("\n=== Loading Model ===")
        start = time.time()

        torch_dtype = getattr(torch, self.dtype) if self.dtype != "auto" else "auto"
        model_kwargs: Dict[str, Any] = {"dtype": torch_dtype}

        if self.device_map:
            model_kwargs["device_map"] = self.device_map
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, **model_kwargs
            )
            self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded in {time.time() - start:.2f}s")

    def _load_processor(self) -> None:
        print("\n=== Loading Processor ===")
        start = time.time()
        self.processor = Qwen3VLProcessor.from_pretrained(self.model_id)
        print(f"Processor loaded in {time.time() - start:.2f}s")

    def run_inference(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        sample_id: str = "sample_0",
    ) -> InferenceMetrics:
        """Run a single inference and return timing metrics."""
        total_start = time.time()
        output_text = ""
        error_message = None
        success = True
        preprocessing_time = generation_time = decoding_time = 0.0
        input_tokens = output_tokens = 0
        tokens_per_second = 0.0

        try:
            preprocess_start = time.time()
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self.device)
            preprocessing_time = time.time() - preprocess_start

            input_tokens = inputs["input_ids"].shape[1]

            generation_start = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            generation_time = time.time() - generation_start

            decode_start = time.time()
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            decoding_time = time.time() - decode_start

            output_tokens = len(generated_ids_trimmed[0])
            tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0.0

        except Exception as e:
            success = False
            error_message = str(e)

        return InferenceMetrics(
            sample_id=sample_id,
            preprocessing_time=preprocessing_time,
            generation_time=generation_time,
            decoding_time=decoding_time,
            total_time=time.time() - total_start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_text=output_text,
            success=success,
            error_message=error_message,
        )

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        show_progress: bool = True,
    ) -> List[InferenceMetrics]:
        """Run inference on a list of test cases."""
        results: List[InferenceMetrics] = []
        iterator = tqdm(test_cases, desc="Evaluating") if show_progress else test_cases
        for test_case in iterator:
            sample_id = test_case.get("sample_id", f"sample_{len(results)}")
            results.append(
                self.run_inference(
                    messages=test_case["messages"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    sample_id=sample_id,
                )
            )
        return results

    def compute_summary(self, metrics_list: List[InferenceMetrics]) -> EvaluationSummary:
        """Compute aggregate statistics from a list of per-sample metrics."""
        successful = [m for m in metrics_list if m.success]
        failed = [m for m in metrics_list if not m.success]

        if not successful:
            return EvaluationSummary(
                total_samples=len(metrics_list),
                successful_samples=0,
                failed_samples=len(failed),
                avg_preprocessing_time=0,
                avg_generation_time=0,
                avg_decoding_time=0,
                avg_total_time=0,
                avg_tokens_per_second=0,
                total_tokens_generated=0,
                total_time=0,
                median_generation_time=0,
                p95_generation_time=0,
                p99_generation_time=0,
            )

        gen_times = sorted(m.generation_time for m in successful)
        return EvaluationSummary(
            total_samples=len(metrics_list),
            successful_samples=len(successful),
            failed_samples=len(failed),
            avg_preprocessing_time=statistics.mean(m.preprocessing_time for m in successful),
            avg_generation_time=statistics.mean(gen_times),
            avg_decoding_time=statistics.mean(m.decoding_time for m in successful),
            avg_total_time=statistics.mean(m.total_time for m in successful),
            avg_tokens_per_second=statistics.mean(m.tokens_per_second for m in successful),
            total_tokens_generated=sum(m.output_tokens for m in successful),
            total_time=sum(m.total_time for m in metrics_list),
            median_generation_time=statistics.median(gen_times),
            p95_generation_time=gen_times[int(len(gen_times) * 0.95)] if gen_times else 0,
            p99_generation_time=gen_times[int(len(gen_times) * 0.99)] if gen_times else 0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Hook-based bottleneck profiler
# ═══════════════════════════════════════════════════════════════════════════════


class HookProfiler:
    """
    Registers forward-pre and forward hooks on selected modules to measure
    wall-clock time (CUDA-synchronised when applicable).
    """

    _TARGET_CLASSES = (
        "Qwen3VLVisionPatchEmbed",
        "Qwen3VLVisionRotaryEmbedding",
        "Qwen3VLVisionAttention",
        "Qwen3VLVisionMLP",
        "Qwen3VLVisionBlock",
        "Qwen3VLVisionPatchMerger",
        "Qwen3VLVisionModel",
        "Qwen3VLTextRotaryEmbedding",
        "Qwen3VLTextRMSNorm",
        "Qwen3VLTextAttention",
        "Qwen3VLTextMLP",
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLTextModel",
        "Qwen3VLModel",
        "Qwen3VLForConditionalGeneration",
    )

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.device = device
        self.timings: Dict[str, ComponentTiming] = {}
        self._handles: list = []
        self._stack: list = []
        self._register_hooks(model)

    def _register_hooks(self, model: torch.nn.Module) -> None:
        for name, mod in model.named_modules():
            cls_name = type(mod).__name__
            if cls_name in self._TARGET_CLASSES:
                label = f"{name} ({cls_name})" if name else cls_name
                self.timings[label] = ComponentTiming(name=label)
                self._handles.append(mod.register_forward_pre_hook(self._make_pre_hook(label)))
                self._handles.append(mod.register_forward_hook(self._make_post_hook(label)))

    def _sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _make_pre_hook(self, label: str):
        def hook(_module: Any, _input: Any) -> None:
            self._sync()
            self._stack.append((label, time.perf_counter()))
        return hook

    def _make_post_hook(self, label: str):
        def hook(_module: Any, _input: Any, _output: Any) -> None:
            self._sync()
            end = time.perf_counter()
            while self._stack and self._stack[-1][0] != label:
                self._stack.pop()
            if self._stack:
                _, start = self._stack.pop()
                elapsed_ms = (end - start) * 1000
                self.timings[label].total_ms += elapsed_ms
                self.timings[label].call_count += 1
        return hook

    def reset(self) -> None:
        for t in self.timings.values():
            t.total_ms = 0.0
            t.call_count = 0
        self._stack.clear()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summary_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            k: {"total_ms": v.total_ms, "calls": v.call_count}
            for k, v in sorted(self.timings.items(), key=lambda x: -x[1].total_ms)
            if v.call_count > 0
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Profiling runner
# ═══════════════════════════════════════════════════════════════════════════════


def _run_once(model: Any, processor: Any, messages: List[Dict[str, Any]], device: str, max_new_tokens: int) -> None:
    """Single inference pass with no instrumentation (used for warmup)."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens)


def run_profile(
    model: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    device: str,
    max_new_tokens: int = 128,
    use_torch_profiler: bool = True,
    tensorboard_dir: Optional[str] = None,
    warmup_runs: int = 1,
    output_dir: Optional[Path] = None,
) -> ProfileResult:
    """Run inference with hook-based profiling and optionally torch.profiler."""
    result = ProfileResult()
    trace_output_dir = Path(output_dir) if output_dir else Path("results")
    trace_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Profiler Warmup ({warmup_runs} run(s)) ===")
    for i in range(warmup_runs):
        _run_once(model, processor, messages, device, max_new_tokens)
        print(f"  warmup {i + 1}/{warmup_runs} done")

    print("\n=== Hook-based component profiling ===")
    hook_profiler = HookProfiler(model, device)
    hook_profiler.reset()

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    result.preprocessing_ms = (time.perf_counter() - t0) * 1000
    result.input_tokens = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    result.generation_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result.decoding_ms = (time.perf_counter() - t2) * 1000

    result.output_tokens = len(trimmed[0])
    result.total_ms = result.preprocessing_ms + result.generation_ms + result.decoding_ms
    result.tokens_per_second = result.output_tokens / (result.generation_ms / 1000) if result.generation_ms > 0 else 0.0
    result.component_timings = hook_profiler.summary_dict()
    hook_profiler.remove_hooks()

    print(f"  Output ({result.output_tokens} tokens): {output_text[:120]}...")

    if use_torch_profiler:
        print("\n=== torch.profiler pass (separate inference) ===")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        tb_handler = torch.profiler.tensorboard_trace_handler(tensorboard_dir) if tensorboard_dir else None

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=tb_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(2):
                _run_once(model, processor, messages, device, max_new_tokens)
                prof.step()

        sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
        result.torch_profiler_key_averages = prof.key_averages().table(sort_by=sort_key, row_limit=40)

        trace_path = trace_output_dir / f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        prof.export_chrome_trace(str(trace_path))
        result.trace_path = str(trace_path)
        print(f"  Chrome trace saved to {trace_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_components(timings: dict, class_filter: List[str]) -> Dict[str, float]:
    agg: Dict[str, float] = defaultdict(float)
    for key, val in timings.items():
        for cls in class_filter:
            if cls in key:
                short = cls.replace("Qwen3VL", "").replace("Vision", "V.")
                agg[short] += val["total_ms"]
                break
    return dict(agg)


def _extract_layer_breakdown(timings: dict):
    attn_by_layer: Dict[int, float] = {}
    mlp_by_layer: Dict[int, float] = {}
    for key, val in timings.items():
        layer_idx = _parse_layer_index(key)
        if layer_idx is None:
            continue
        if "Qwen3VLTextAttention" in key:
            attn_by_layer[layer_idx] = attn_by_layer.get(layer_idx, 0) + val["total_ms"]
        elif "Qwen3VLTextMLP" in key:
            mlp_by_layer[layer_idx] = mlp_by_layer.get(layer_idx, 0) + val["total_ms"]

    if not attn_by_layer:
        return [], [], []
    all_layers = sorted(set(attn_by_layer) | set(mlp_by_layer))
    return (
        [attn_by_layer.get(i, 0) for i in all_layers],
        [mlp_by_layer.get(i, 0) for i in all_layers],
        [f"L{i}" for i in all_layers],
    )


def _parse_layer_index(key: str) -> Optional[int]:
    match = re.search(r"layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def plot_profile(result: ProfileResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    fig.suptitle("Qwen3-VL Inference Profile", fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    phases = ["Preprocess", "Generation", "Decoding"]
    times = [result.preprocessing_ms, result.generation_ms, result.decoding_ms]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax1.barh(phases, times, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Inference Phase Breakdown")
    for bar, t in zip(bars, times):
        ax1.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{t:.1f} ms", va="center", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    vision_ms = sum(v["total_ms"] for k, v in result.component_timings.items()
                    if "Qwen3VLVisionModel" in k and k.count("(") == 1)
    text_ms = sum(v["total_ms"] for k, v in result.component_timings.items()
                  if "Qwen3VLTextModel" in k and k.count("(") == 1)
    other_ms = max(result.generation_ms - vision_ms - text_ms, 0)
    ax2.pie(
        [vision_ms, text_ms, other_ms],
        labels=[f"Vision Encoder\n{vision_ms:.1f} ms", f"Text Decoder\n{text_ms:.1f} ms", f"Other\n{other_ms:.1f} ms"],
        colors=["#C44E52", "#8172B3", "#CCB974"],
        autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9},
    )
    ax2.set_title("Generation Time Split")

    ax3 = fig.add_subplot(gs[0, 2])
    vision_components = _extract_components(result.component_timings, [
        "Qwen3VLVisionPatchEmbed", "Qwen3VLVisionAttention", "Qwen3VLVisionMLP",
        "Qwen3VLVisionPatchMerger", "Qwen3VLVisionRotaryEmbedding",
    ])
    if vision_components:
        names, vals = zip(*sorted(vision_components.items(), key=lambda x: -x[1]))
        y_pos = np.arange(len(names))
        ax3.barh(y_pos, vals, color="#C44E52", alpha=0.85, edgecolor="white")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_xlabel("Time (ms)")
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, "No vision data\n(text-only input?)", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Vision Encoder Components")

    ax4 = fig.add_subplot(gs[1, 0:2])
    attn_times, mlp_times, layer_labels = _extract_layer_breakdown(result.component_timings)
    if attn_times:
        x = np.arange(len(layer_labels))
        width = 0.35
        ax4.bar(x - width / 2, attn_times, width, label="Attention", color="#4C72B0", alpha=0.85)
        ax4.bar(x + width / 2, mlp_times, width, label="MLP", color="#DD8452", alpha=0.85)
        ax4.set_xlabel("Decoder Layer")
        ax4.set_ylabel("Time (ms)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(layer_labels, fontsize=7, rotation=45, ha="right")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No per-layer data available", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("Per-Layer Attention vs MLP Time")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary_text = (
        f"Total inference:  {result.total_ms:.1f} ms\n"
        f"  Preprocessing:  {result.preprocessing_ms:.1f} ms\n"
        f"  Generation:     {result.generation_ms:.1f} ms\n"
        f"  Decoding:       {result.decoding_ms:.1f} ms\n\n"
        f"Input tokens:     {result.input_tokens}\n"
        f"Output tokens:    {result.output_tokens}\n"
        f"Tokens/sec:       {result.tokens_per_second:.1f}\n"
    )
    if result.trace_path:
        summary_text += f"\nChrome trace:\n  {result.trace_path}\n  (open in chrome://tracing)"
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10, verticalalignment="top",
             fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax5.set_title("Summary")

    chart_path = output_dir / "profile_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Saving helpers
# ═══════════════════════════════════════════════════════════════════════════════


def create_default_test_case() -> Dict[str, Any]:
    return {
        "sample_id": "demo_sample",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
    }


def save_evaluation_results(
    metrics_list: List[InferenceMetrics],
    summary: EvaluationSummary,
    output_dir: Path,
    prefix: str = "evaluation",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / f"{prefix}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in metrics_list], f, indent=2, ensure_ascii=False)

    summary_file = output_dir / f"{prefix}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\nMetrics  → {metrics_file}")
    print(f"Summary  → {summary_file}")
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Samples:   {summary.total_samples}")
    print(f"Successful:      {summary.successful_samples}")
    print(f"Failed:          {summary.failed_samples}")
    print(f"\nAvg Preprocessing:  {summary.avg_preprocessing_time * 1000:.2f} ms")
    print(f"Avg Generation:     {summary.avg_generation_time * 1000:.2f} ms")
    print(f"Avg Decoding:       {summary.avg_decoding_time * 1000:.2f} ms")
    print(f"Avg Total:          {summary.avg_total_time * 1000:.2f} ms")
    print(f"Avg Tokens/sec:     {summary.avg_tokens_per_second:.2f}")
    print(f"Total Tokens:       {summary.total_tokens_generated}")
    print(f"Median Gen Time:    {summary.median_generation_time * 1000:.2f} ms")
    print(f"P95 Gen Time:       {summary.p95_generation_time * 1000:.2f} ms")
    print(f"P99 Gen Time:       {summary.p99_generation_time * 1000:.2f} ms")
    print("=" * 60)


def save_profile_results(result: ProfileResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("COMPONENT TIMINGS (hook-based)")
    print("=" * 80)
    print(f"  {'Component':<70} {'Time (ms)':>10} {'Calls':>6}")
    print("  " + "-" * 88)
    for key, val in result.component_timings.items():
        print(f"  {key:<70} {val['total_ms']:>10.2f} {val['calls']:>6}")

    if result.torch_profiler_key_averages:
        print("\n" + "=" * 80)
        print("TORCH PROFILER KEY AVERAGES (top 40 ops)")
        print("=" * 80)
        print(result.torch_profiler_key_averages)

    json_path = output_dir / "profile_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing_ms": result.preprocessing_ms,
                "generation_ms": result.generation_ms,
                "decoding_ms": result.decoding_ms,
                "total_ms": result.total_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "tokens_per_second": result.tokens_per_second,
                "component_timings": result.component_timings,
                "trace_path": result.trace_path,
            },
            f, indent=2,
        )
    print(f"\nProfile data → {json_path}")

    chart_path = plot_profile(result, output_dir)
    print(f"Profile chart → {chart_path}")

    if result.trace_path:
        print(f"\nTo view Chrome trace: open chrome://tracing and load {result.trace_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-VL Evaluation & Profiling")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32", "auto"], default="auto")
    parser.add_argument("--device-map", type=str, default=None,
                        help="Device map strategy, e.g. 'auto' or 'balanced'")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--test-cases-file", type=str, default=None,
                        help="JSON file with test cases (list of {sample_id, messages})")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of times to run the default test case")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--output-prefix", type=str, default="evaluation")
    parser.add_argument("--no-torch-profiler", action="store_true",
                        help="Skip torch.profiler pass (hook-based profiling only)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Export torch.profiler data for TensorBoard")
    parser.add_argument("--warmup-runs", type=int, default=1)

    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    dtype = None if args.dtype == "auto" else args.dtype

    evaluator = Qwen3VLEvaluator(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        device_map=args.device_map,
    )

    if args.test_cases_file:
        print(f"\n=== Loading test cases from {args.test_cases_file} ===")
        with open(args.test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} test cases")
    else:
        print(f"\n=== Using default test case ({args.num_samples}×) ===")
        test_cases = [create_default_test_case() for _ in range(args.num_samples)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp

    print("\n=== Running Evaluation ===")
    metrics_list = evaluator.evaluate_batch(
        test_cases=test_cases,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    summary = evaluator.compute_summary(metrics_list)
    save_evaluation_results(metrics_list, summary, output_dir, args.output_prefix)

    print("\n" + "=" * 60)
    print("RUNNING PROFILING")
    print("=" * 60)
    profile_result = run_profile(
        model=evaluator.model,
        processor=evaluator.processor,
        messages=create_default_test_case()["messages"],
        device=evaluator.device,
        max_new_tokens=args.max_new_tokens,
        use_torch_profiler=not args.no_torch_profiler,
        tensorboard_dir=str(output_dir / "tensorboard") if args.tensorboard else None,
        warmup_runs=args.warmup_runs,
        output_dir=output_dir,
    )
    save_profile_results(profile_result, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
