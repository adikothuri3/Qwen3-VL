"""
DeepStack measurement instrumentation (Phase 2).

Non-invasive forward-hook instrumentation that quantifies, across a small
calibration set, the five things later phases need to budget DeepStack visual
tokens (see paper.md section 13, Phase 2):

  1. token count per DeepStack group
  2. feature-norm distributions per group (stats + histogram + percentiles)
  3. attention/saliency over visual tokens (opt-in; --capture-attention)
  4. latency around DeepStack feature extraction and injection
  5. GPU memory around extraction and injection per group

It edits no model source. All measurements come from hooks on existing modules:

  - extraction      -> each Qwen3VLVisionPatchMerger in `deepstack_merger_list`
  - counts + norms  -> the (hidden, deepstack_feature_lists) tuple returned by
                       Qwen3VLVisionModel.forward
  - visual_pos_masks-> a pre-hook on Qwen3VLTextModel.forward
  - injection time  -> bracketing decoder layers 0,1,2 (injection i happens
                       between the return of layer i and the call of layer i+1,
                       so post-hook(layer i) .. pre-hook(layer i+1) times it)
  - attention       -> forward hook on Qwen3VLTextAttention layers 0,1,2, reading
                       the returned attn_weights (requires eager attention)

Images are fetched and size-capped locally (so a giant image can't OOM the GPU)
and every sample is wrapped so a single bad image falls back to a deterministic
synthetic image rather than killing the run.

Output: results/<timestamp>/deepstack_instrument.json

This is read/measure only. It exists to produce the per-group distributions that
calibrated budgeting (Phase 5) and within-group scoring (Phase 4) consume.
"""

import argparse
import inspect
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Reuse the evaluator's local_transformers wiring, model+processor loading, and
# default image+text test case. Importing evaluate performs the sys.modules
# redirect to local_transformers before any transformers import.
from src.evaluate import Qwen3VLEvaluator

_HIST_BINS = 30
_DEFAULT_NUM_SAMPLES = 4
_MAX_IMAGE_SIDE = 1024  # cap longest side so a huge image can't OOM the prefill


# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration set (robust: real images with a guaranteed synthetic fallback)
# ═══════════════════════════════════════════════════════════════════════════════

# Battle-tested, stable image URLs spanning task types that paper.md flags as
# differing in compression sensitivity. Fetched + size-capped locally; on any
# failure the sample falls back to a deterministic synthetic image.
REAL_CALIBRATION: List[Tuple[str, str]] = [
    ("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", "Describe this image."),
    ("http://images.cocodataset.org/val2017/000000039769.jpg", "How many cats are in this image?"),
]

_SYNTH_PROMPTS = [
    "Describe this image.",
    "What text or shapes do you see?",
    "Summarize the visual content.",
    "What is in this image?",
]


def _build_plan(num_samples: int, synthetic_only: bool) -> List[Dict[str, Any]]:
    """Plan one entry per sample: a real URL when available, else synthetic."""
    plan: List[Dict[str, Any]] = []
    for i in range(max(1, num_samples)):
        if not synthetic_only and i < len(REAL_CALIBRATION):
            url, prompt = REAL_CALIBRATION[i]
            plan.append({"kind": "url", "url": url, "prompt": prompt, "seed": i})
        else:
            plan.append({"kind": "synthetic", "prompt": _SYNTH_PROMPTS[i % len(_SYNTH_PROMPTS)], "seed": i})
    return plan


def _cap_image(img: "Any", max_side: int = _MAX_IMAGE_SIDE) -> "Any":
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def _fetch_pil(url: str, timeout: int = 20) -> "Any":
    import ssl

    from PIL import Image

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed https/http calibration URLs
            data = resp.read()
    except urllib.error.URLError as e:
        # Public calibration images: tolerate a missing local cert store by
        # retrying without verification rather than dropping to synthetic.
        if not isinstance(getattr(e, "reason", None), ssl.SSLError):
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310
            data = resp.read()
    return Image.open(BytesIO(data)).convert("RGB")


def _synthetic_image(seed: int, size: int = 512) -> "Any":
    """Deterministic, network-free image; varied pattern per seed so token
    statistics differ across synthetic samples."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    kind = seed % 4
    if kind == 0:  # smooth gradient
        ramp = np.linspace(0, 255, size, dtype=np.uint8)
        arr[..., 0] = np.tile(ramp, (size, 1))
        arr[..., 1] = np.tile(ramp[:, None], (1, size))
        arr[..., 2] = 128
    elif kind == 1:  # coloured blocks (chart-like)
        step = size // 8
        for r in range(0, size, step):
            for c in range(0, size, step):
                arr[r : r + step, c : c + step] = rng.integers(0, 256, size=3)
    elif kind == 2:  # scattered dots (counting-like)
        arr[:] = 230
        for _ in range(60):
            cy, cx = rng.integers(20, size - 20, size=2)
            rad = int(rng.integers(5, 18))
            yy, xx = np.ogrid[:size, :size]
            arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2] = rng.integers(0, 200, size=3)
    else:  # high-frequency noise (text/detail-like)
        arr[:] = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _resolve_image(spec: Dict[str, Any]) -> Tuple["Any", str]:
    """Return (PIL image, source). URL fetch falls back to synthetic on failure."""
    if spec["kind"] == "url":
        try:
            return _cap_image(_fetch_pil(spec["url"])), "url"
        except Exception as e:  # noqa: BLE001 — any fetch/decode failure -> synthetic fallback
            print(f"  image fetch failed ({type(e).__name__}: {e}); using synthetic", flush=True)
    return _synthetic_image(int(spec["seed"])), "synthetic"


# ═══════════════════════════════════════════════════════════════════════════════
#  Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Distribution:
    """Summary of a pooled scalar distribution (e.g. per-token norms/saliency)."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    p10: float
    p50: float
    p90: float
    hist_counts: List[int] = field(default_factory=list)
    hist_edges: List[float] = field(default_factory=list)


@dataclass
class GroupMeasurement:
    """Aggregated per-DeepStack-group measurements over the calibration set."""

    group_index: int
    vision_layer: int
    injection_decoder_layer: int
    hidden_dim: int
    token_count_mean: float
    token_count_min: int
    token_count_max: int
    count_matches_mask: bool
    norm_dist: Optional[Distribution]
    attention_dist: Optional[Distribution]
    extraction_ms_mean: float
    injection_ms_mean: float
    extract_mem_delta_mb_mean: float
    inject_mem_delta_mb_mean: float
    extract_mem_peak_mb_max: float


@dataclass
class InstrumentReport:
    """Full Phase 2 measurement report aggregated over the calibration set."""

    model_source: str
    num_groups: int
    num_samples: int
    per_sample_source: List[str]
    device: str
    dtype: str
    capture_attention: bool
    vision_layers: List[int]
    injection_layers: List[int]
    out_hidden_size: int
    spatial_merge_size: int
    per_sample_visual_tokens: List[int]
    groups: List[GroupMeasurement] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_distribution(values: np.ndarray) -> Optional[Distribution]:
    """Pooled-distribution summary with a fixed-bin histogram, or None if empty."""
    if values.size == 0:
        return None
    counts, edges = np.histogram(values, bins=_HIST_BINS)
    return Distribution(
        count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std()) if values.size > 1 else 0.0,
        min=float(values.min()),
        max=float(values.max()),
        p10=float(np.percentile(values, 10)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
        hist_counts=[int(c) for c in counts],
        hist_edges=[float(e) for e in edges],
    )


class _GroupAccumulator:
    """Collects per-sample measurements for one DeepStack group."""

    def __init__(self) -> None:
        self.token_counts: List[int] = []
        self.norms: List[np.ndarray] = []
        self.saliency: List[np.ndarray] = []
        self.extraction_ms: List[float] = []
        self.injection_ms: List[float] = []
        self.extract_mem_delta_mb: List[float] = []
        self.inject_mem_delta_mb: List[float] = []
        self.extract_mem_peak_mb: List[float] = []

    def finalize(
        self, group_index: int, vision_layer: int, hidden_dim: int, mask_counts: List[int]
    ) -> GroupMeasurement:
        norms = np.concatenate(self.norms) if self.norms else np.array([], dtype=np.float64)
        sal = np.concatenate(self.saliency) if self.saliency else np.array([], dtype=np.float64)
        counts = np.array(self.token_counts, dtype=np.float64)
        count_matches = bool(self.token_counts) and self.token_counts == mask_counts
        return GroupMeasurement(
            group_index=group_index,
            vision_layer=vision_layer,
            injection_decoder_layer=group_index,
            hidden_dim=hidden_dim,
            token_count_mean=float(counts.mean()) if counts.size else 0.0,
            token_count_min=int(counts.min()) if counts.size else 0,
            token_count_max=int(counts.max()) if counts.size else 0,
            count_matches_mask=count_matches,
            norm_dist=_make_distribution(norms),
            attention_dist=_make_distribution(sal),
            extraction_ms_mean=float(np.mean(self.extraction_ms)) if self.extraction_ms else 0.0,
            injection_ms_mean=float(np.mean(self.injection_ms)) if self.injection_ms else 0.0,
            extract_mem_delta_mb_mean=(
                float(np.mean(self.extract_mem_delta_mb)) if self.extract_mem_delta_mb else 0.0
            ),
            inject_mem_delta_mb_mean=(
                float(np.mean(self.inject_mem_delta_mb)) if self.inject_mem_delta_mb else 0.0
            ),
            extract_mem_peak_mb_max=(
                float(np.max(self.extract_mem_peak_mb)) if self.extract_mem_peak_mb else 0.0
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Instrumentor
# ═══════════════════════════════════════════════════════════════════════════════


def _seq_len(inp: Any) -> int:
    """Sequence length of a decoder layer's hidden_states input, or -1 if unknown."""
    try:
        t = inp[0]
        return int(t.shape[1]) if t.dim() >= 2 else -1
    except Exception:  # noqa: BLE001 — defensive; unknown input shape -> treat as prefill
        return -1


class DeepStackInstrumentor:
    """Attaches non-invasive hooks and accumulates per-group measurements.

    Lifecycle mirrors DeepStackProbe: use as a context manager around the forward
    passes. Call `harvest()` once per sample (after each forward) to fold the
    per-forward captures into the cross-sample accumulators.
    """

    def __init__(self, model: Any, device: str, capture_attention: bool) -> None:
        self.model = model
        self.device = device
        self.is_cuda = device == "cuda"
        self.capture_attention = capture_attention
        self._handles: List[Any] = []

        self.vision_model: Any = None
        self.deepstack_mergers: List[Any] = []
        self.text_layers: List[Any] = []
        self._locate_modules()
        self.num_groups = len(self.deepstack_mergers)

        self.accumulators = [_GroupAccumulator() for _ in range(self.num_groups)]
        self.mask_counts: List[int] = []  # visual_pos_masks.sum() per sample

        # Per-forward scratch, cleared by reset_sample().
        self._cur_masks: Optional[torch.Tensor] = None
        self._cur_features: Optional[List[torch.Tensor]] = None
        self._extract_t: Dict[int, float] = {}
        self._extract_ms: Dict[int, float] = {}
        self._extract_mem_pre: Dict[int, int] = {}
        self._extract_mem: Dict[int, float] = {}
        self._extract_peak: Dict[int, float] = {}
        self._layer_boundary_t: Dict[int, float] = {}
        self._layer_boundary_mem: Dict[int, int] = {}
        self._inject_ms: Dict[int, float] = {}
        self._inject_mem: Dict[int, float] = {}
        self._attn: Dict[int, np.ndarray] = {}

    def _locate_modules(self) -> None:
        for module in self.model.modules():
            name = type(module).__name__
            if name == "Qwen3VLVisionModel":
                self.vision_model = module
                self.deepstack_mergers = list(module.deepstack_merger_list)
            elif name == "Qwen3VLTextModel":
                self.text_layers = list(module.layers)
        if self.vision_model is None or not self.deepstack_mergers or not self.text_layers:
            raise RuntimeError("Could not locate DeepStack modules — model structure unexpected.")

    # ── hook registration ──────────────────────────────────────────────────────

    def _register(self) -> None:
        self._handles.append(self.vision_model.register_forward_hook(self._vision_hook))

        for gi, merger in enumerate(self.deepstack_mergers):
            self._handles.append(merger.register_forward_pre_hook(self._make_extract_pre(gi)))
            self._handles.append(merger.register_forward_hook(self._make_extract_post(gi)))

        for module in self.model.modules():
            if type(module).__name__ == "Qwen3VLTextModel":
                self._handles.append(module.register_forward_pre_hook(self._text_pre_hook, with_kwargs=True))

        for li in range(self.num_groups + 1):
            if li < len(self.text_layers):
                self._handles.append(self.text_layers[li].register_forward_pre_hook(self._make_layer_pre(li)))
                self._handles.append(self.text_layers[li].register_forward_hook(self._make_layer_post(li)))

        if self.capture_attention:
            for module in self.model.modules():
                if type(module).__name__ == "Qwen3VLTextAttention":
                    layer_idx = getattr(module, "layer_idx", None)
                    if layer_idx is not None and layer_idx < self.num_groups:
                        self._handles.append(module.register_forward_hook(self._make_attn_post(layer_idx)))

    # ── timing/memory primitives ───────────────────────────────────────────────

    def _sync(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize()

    def _now(self) -> float:
        self._sync()
        return time.perf_counter()

    def _mem(self) -> int:
        return torch.cuda.memory_allocated() if self.is_cuda else 0

    # ── hooks ───────────────────────────────────────────────────────────────────

    def _vision_hook(self, module: Any, args: Any, output: Any) -> None:
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], list):
            self._cur_features = [f.detach() for f in output[1]]

    def _make_extract_pre(self, gi: int):
        def hook(_module: Any, _inp: Any) -> None:
            if self.is_cuda:
                torch.cuda.reset_peak_memory_stats()
            self._extract_mem_pre[gi] = self._mem()
            self._extract_t[gi] = self._now()

        return hook

    def _make_extract_post(self, gi: int):
        def hook(_module: Any, _inp: Any, _out: Any) -> None:
            if gi not in self._extract_t:
                return
            self._extract_ms[gi] = (self._now() - self._extract_t[gi]) * 1000.0
            if self.is_cuda:
                self._extract_mem[gi] = (self._mem() - self._extract_mem_pre[gi]) / 1e6
                self._extract_peak[gi] = torch.cuda.max_memory_allocated() / 1e6

        return hook

    def _text_pre_hook(self, module: Any, args: Any, kwargs: Dict[str, Any]) -> None:
        masks = kwargs.get("visual_pos_masks")
        if masks is not None:
            self._cur_masks = masks.detach()

    def _make_layer_pre(self, li: int):
        def hook(_module: Any, inp: Any) -> None:
            # Only the prefill forward carries visual tokens to inject.
            if _seq_len(inp) <= 1 or self._cur_masks is None:
                return
            # A layer-i pre-hook marks the end of injection (i-1).
            if li - 1 in self._layer_boundary_t:
                t = self._now()
                self._inject_ms[li - 1] = (t - self._layer_boundary_t[li - 1]) * 1000.0
                if self.is_cuda:
                    self._inject_mem[li - 1] = (self._mem() - self._layer_boundary_mem[li - 1]) / 1e6

        return hook

    def _make_layer_post(self, li: int):
        def hook(_module: Any, inp: Any, _out: Any) -> None:
            if _seq_len(inp) <= 1 or self._cur_masks is None:
                return
            # A layer-i post-hook marks the start of injection i.
            self._layer_boundary_t[li] = self._now()
            self._layer_boundary_mem[li] = self._mem()

        return hook

    def _make_attn_post(self, layer_idx: int):
        def hook(_module: Any, _inp: Any, output: Any) -> None:
            if self._cur_masks is None or not isinstance(output, tuple) or len(output) < 2:
                return
            attn_weights = output[1]
            if attn_weights is None:
                return
            mask = self._cur_masks
            # Only the prefill forward has attention aligned to the mask length.
            if attn_weights.shape[-1] != mask.shape[-1]:
                return
            flat_mask = mask.reshape(-1).to(torch.bool)
            received = attn_weights.float().mean(dim=(0, 1, 2))  # (kv_seq,)
            self._attn[layer_idx] = received[flat_mask].detach().cpu().numpy()

        return hook

    # ── per-sample lifecycle ────────────────────────────────────────────────────

    def reset_sample(self) -> None:
        self._cur_masks = None
        self._cur_features = None
        self._extract_t.clear()
        self._extract_ms.clear()
        self._extract_mem_pre.clear()
        self._extract_mem.clear()
        self._extract_peak.clear()
        self._layer_boundary_t.clear()
        self._layer_boundary_mem.clear()
        self._inject_ms.clear()
        self._inject_mem.clear()
        self._attn.clear()

    def harvest(self) -> None:
        """Fold the current forward's captures into the cross-sample accumulators."""
        if self._cur_masks is not None:
            self.mask_counts.append(int(self._cur_masks.sum().item()))
        features = self._cur_features or []
        for gi in range(self.num_groups):
            acc = self.accumulators[gi]
            if gi < len(features):
                feat = features[gi]
                acc.token_counts.append(int(feat.shape[0]))
                acc.norms.append(feat.float().norm(dim=-1).cpu().numpy())
            if gi in self._extract_ms:
                acc.extraction_ms.append(self._extract_ms[gi])
            if gi in self._extract_mem:
                acc.extract_mem_delta_mb.append(self._extract_mem[gi])
            if gi in self._extract_peak:
                acc.extract_mem_peak_mb.append(self._extract_peak[gi])
            if gi in self._inject_ms:
                acc.injection_ms.append(self._inject_ms[gi])
            if gi in self._inject_mem:
                acc.inject_mem_delta_mb.append(self._inject_mem[gi])
            if gi in self._attn:
                acc.saliency.append(self._attn[gi])

    def __enter__(self) -> "DeepStackInstrumentor":
        self._register()
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════════════


def _force_eager_attention(model: Any) -> None:
    """Set eager attention everywhere so attn_weights are returned for capture."""
    cfg = model.config
    for sub in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
        if sub is not None:
            sub._attn_implementation = "eager"
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:  # noqa: BLE001 — best-effort; config flags above are the fallback
            pass


def instrument(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    output_dir: str = "results",
    num_samples: int = _DEFAULT_NUM_SAMPLES,
    capture_attention: bool = False,
    synthetic_only: bool = False,
) -> InstrumentReport:
    evaluator = Qwen3VLEvaluator(model_id=model_id, device=device, dtype=dtype)
    model = evaluator.model
    processor = evaluator.processor

    if capture_attention:
        _force_eager_attention(model)

    vision_cfg = model.config.vision_config
    vision_layers: List[int] = list(vision_cfg.deepstack_visual_indexes)
    out_hidden_size = int(vision_cfg.out_hidden_size)
    spatial_merge_size = int(vision_cfg.spatial_merge_size)

    plan = _build_plan(num_samples, synthetic_only)
    sources: List[str] = []

    with DeepStackInstrumentor(model, evaluator.device, capture_attention) as inst:
        for idx, spec in enumerate(plan):
            inst.reset_sample()
            image, source = _resolve_image(spec)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": spec["prompt"]},
                    ],
                }
            ]
            try:
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
                )
                inputs.pop("token_type_ids", None)
                inputs = inputs.to(evaluator.device)
            except Exception as e:  # noqa: BLE001 — one bad sample must not kill the run
                print(f"[sample {idx}] preprocessing failed ({type(e).__name__}: {e}); skipping", flush=True)
                continue
            try:
                with torch.no_grad():
                    model(**inputs)  # single prefill forward is all we measure
            except Exception as e:  # noqa: BLE001 — skip a failed forward, keep going
                print(f"[sample {idx}] forward failed ({type(e).__name__}: {e}); skipping", flush=True)
                if evaluator.device == "cuda":
                    torch.cuda.empty_cache()
                continue
            inst.harvest()
            sources.append(source)
            n_tok = inst.mask_counts[-1] if inst.mask_counts else -1
            print(f"[sample {idx}] ok (source={source}, visual_tokens={n_tok})", flush=True)

    if not inst.mask_counts:
        raise RuntimeError("Instrumentor captured no DeepStack tensors — every sample failed. See log above.")

    groups: List[GroupMeasurement] = [
        inst.accumulators[gi].finalize(
            group_index=gi,
            vision_layer=vision_layers[gi],
            hidden_dim=out_hidden_size,
            mask_counts=inst.mask_counts,
        )
        for gi in range(inst.num_groups)
    ]

    report = InstrumentReport(
        model_source=inspect.getfile(type(model)),
        num_groups=inst.num_groups,
        num_samples=len(sources),
        per_sample_source=sources,
        device=evaluator.device,
        dtype=evaluator.dtype,
        capture_attention=capture_attention,
        vision_layers=vision_layers,
        injection_layers=list(range(inst.num_groups)),
        out_hidden_size=out_hidden_size,
        spatial_merge_size=spatial_merge_size,
        per_sample_visual_tokens=inst.mask_counts,
        groups=groups,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "deepstack_instrument.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    _print_summary(report, report_path)
    return report


def _fmt_dist(d: Optional[Distribution]) -> str:
    if d is None:
        return "n/a"
    return f"μ={d.mean:.3f}±{d.std:.3f} [{d.min:.3f}, {d.max:.3f}] p50={d.p50:.3f}"


def _print_summary(r: InstrumentReport, path: Path) -> None:
    print("\n=== DeepStack Instrumentation Report (Phase 2) ===")
    print(f"Model source:       {r.model_source}")
    print(f"Samples:            {r.num_samples}  sources={r.per_sample_source}  device={r.device} dtype={r.dtype}")
    print(f"Groups:             {r.num_groups}  (vision layers {r.vision_layers})")
    print(f"Capture attention:  {r.capture_attention}")
    print(f"Visual tokens/sample: {r.per_sample_visual_tokens}")
    for g in r.groups:
        print(
            f"  group {g.group_index} (vit{g.vision_layer}->dec{g.injection_decoder_layer}): "
            f"tokens μ={g.token_count_mean:.1f} [{g.token_count_min},{g.token_count_max}] "
            f"count_match={'PASS' if g.count_matches_mask else 'FAIL'}"
        )
        print(f"      norm:      {_fmt_dist(g.norm_dist)}")
        print(f"      attention: {_fmt_dist(g.attention_dist)}")
        print(
            f"      extract:   {g.extraction_ms_mean:.3f} ms  "
            f"Δmem={g.extract_mem_delta_mb_mean:.2f} MB  peak={g.extract_mem_peak_mb_max:.2f} MB"
        )
        print(f"      inject:    {g.injection_ms_mean:.3f} ms  Δmem={g.inject_mem_delta_mb_mean:.2f} MB")
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instrument Qwen3-VL DeepStack groups (Phase 2).")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "mps", "auto"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "float32", "auto"])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-samples", type=int, default=_DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--capture-attention",
        action="store_true",
        help="Capture per-group visual-token attention saliency (forces eager attention; slow on T4).",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip network image fetches; use deterministic synthetic images only (fully offline).",
    )
    args = parser.parse_args()

    device = None if args.device in (None, "auto") else args.device
    dtype = None if args.dtype in (None, "auto") else args.dtype
    instrument(
        model_id=args.model_id,
        device=device,
        dtype=dtype,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        capture_attention=args.capture_attention,
        synthetic_only=args.synthetic_only,
    )


if __name__ == "__main__":
    main()
