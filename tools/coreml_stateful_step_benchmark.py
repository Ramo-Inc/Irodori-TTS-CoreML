#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import platform
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.model import TextToLatentRFDiT  # noqa: E402
from tools import coreml_real_step_benchmark as real_bench  # noqa: E402

DEFAULT_CHECKPOINT = real_bench.DEFAULT_CHECKPOINT
DEFAULT_CODEC_REPO = real_bench.DEFAULT_CODEC_REPO
DEFAULT_TEXT = real_bench.DEFAULT_TEXT
DEFAULT_REF_WAV = real_bench.DEFAULT_REF_WAV
MODE_COND_ONLY = "cond-only"
STATE_LAYOUT_PACKED = "packed_text_speaker_context_v1"
STATE_LAYOUT_PER_LAYER = "per_layer_text_speaker_context_v1"
STATE_LAYOUT = STATE_LAYOUT_PACKED
NO_STATE_BASELINE_MS = 14.428
STATE_READBACK_TOLERANCE = 1e-3
REQUIRED_NORMALIZED_NE_OPS = ("linear", "matmul", "softmax")
PACKED_STATE_NAMES = ("context_k_state", "context_v_state", "valid_mask_state")
STATE_NAMES = PACKED_STATE_NAMES
STATE_LAYOUT_ALIASES = {
    "packed": STATE_LAYOUT_PACKED,
    STATE_LAYOUT_PACKED: STATE_LAYOUT_PACKED,
    "per-layer": STATE_LAYOUT_PER_LAYER,
    "per_layer": STATE_LAYOUT_PER_LAYER,
    STATE_LAYOUT_PER_LAYER: STATE_LAYOUT_PER_LAYER,
}
LARGE_READ_STATE_BYTES_FP16 = 1_000_000
GIANT_READ_STATE_BYTES_FP16 = 4 * 1024 * 1024


@dataclass(frozen=True)
class PackedContextState:
    context_k_state: np.ndarray
    context_v_state: np.ndarray
    valid_mask_state: np.ndarray
    text_len: int
    speaker_context_len: int
    speaker_context_bucket: int
    c_ctx_bucket: int


def normalize_state_layout(state_layout: str) -> str:
    try:
        return STATE_LAYOUT_ALIASES[str(state_layout).strip()]
    except KeyError as exc:
        allowed = "packed, per-layer"
        raise ValueError(f"state_layout must be one of {allowed}, got {state_layout!r}") from exc


def state_layout_arg(args: argparse.Namespace) -> str:
    return normalize_state_layout(getattr(args, "state_layout", STATE_LAYOUT_PACKED))


def expected_state_names(
    state_layout: str = STATE_LAYOUT_PACKED,
    *,
    num_layers: int | None = None,
) -> tuple[str, ...]:
    normalized = normalize_state_layout(state_layout)
    if normalized == STATE_LAYOUT_PACKED:
        return PACKED_STATE_NAMES

    if num_layers is None:
        raise ValueError("num_layers is required for per-layer state layout expected state names")
    layer_count = int(num_layers)
    if layer_count <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    names: list[str] = []
    for layer_index in range(layer_count):
        names.append(f"context_k_l{layer_index:02d}")
        names.append(f"context_v_l{layer_index:02d}")
    names.append("valid_mask_state")
    return tuple(names)


def state_payloads_from_packed(
    packed: PackedContextState,
    *,
    state_layout: str = STATE_LAYOUT_PACKED,
) -> dict[str, np.ndarray]:
    normalized = normalize_state_layout(state_layout)
    if normalized == STATE_LAYOUT_PACKED:
        return {
            "context_k_state": packed.context_k_state,
            "context_v_state": packed.context_v_state,
            "valid_mask_state": packed.valid_mask_state,
        }

    num_layers = int(packed.context_k_state.shape[0])
    if int(packed.context_v_state.shape[0]) != num_layers:
        raise ValueError(
            "context_k_state/context_v_state layer count mismatch: "
            f"{packed.context_k_state.shape[0]} vs {packed.context_v_state.shape[0]}"
        )

    payloads: dict[str, np.ndarray] = {}
    for layer_index in range(num_layers):
        payloads[f"context_k_l{layer_index:02d}"] = packed.context_k_state[layer_index]
        payloads[f"context_v_l{layer_index:02d}"] = packed.context_v_state[layer_index]
    payloads["valid_mask_state"] = packed.valid_mask_state
    return payloads


def state_payload_shapes(
    packed: PackedContextState,
    *,
    state_layout: str = STATE_LAYOUT_PACKED,
) -> dict[str, list[int]]:
    return {
        name: [int(dim) for dim in payload.shape]
        for name, payload in state_payloads_from_packed(
            packed,
            state_layout=state_layout,
        ).items()
    }


def state_write_payloads_from_packed(
    packed: PackedContextState,
    *,
    state_layout: str = STATE_LAYOUT_PACKED,
) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(payload.astype(np.float32))
        for name, payload in state_payloads_from_packed(
            packed,
            state_layout=state_layout,
        ).items()
    }


def first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else repr(exc)


def compute_state_kv_bytes(
    *,
    num_layers: int,
    c_ctx_bucket: int,
    num_heads: int,
    head_dim: int,
    batch: int = 1,
    dtype_bytes: int = 2,
) -> int:
    return int(2 * num_layers * batch * c_ctx_bucket * num_heads * head_dim * dtype_bytes)


def _as_bool_cpu(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [B, T], got {tuple(mask.shape)}")
    return mask.detach().cpu().to(dtype=torch.bool)


def build_valid_mask_state(
    *,
    text_mask: torch.Tensor,
    speaker_mask: torch.Tensor,
    speaker_context_bucket: int,
    latent_mask: torch.Tensor | None = None,
) -> np.ndarray:
    del latent_mask
    if speaker_context_bucket < 0:
        raise ValueError(f"speaker_context_bucket must be >= 0, got {speaker_context_bucket}")

    text_mask_cpu = _as_bool_cpu(text_mask, name="text_mask")
    speaker_mask_cpu = _as_bool_cpu(speaker_mask, name="speaker_mask")
    if text_mask_cpu.shape[0] != 1 or speaker_mask_cpu.shape[0] != 1:
        raise ValueError(
            "P1a cond-only state packing expects batch size 1, got "
            f"text={tuple(text_mask_cpu.shape)} speaker={tuple(speaker_mask_cpu.shape)}"
        )

    text_len = int(text_mask_cpu.shape[1])
    speaker_len = int(speaker_mask_cpu.shape[1])
    if speaker_len > speaker_context_bucket:
        raise ValueError(
            f"speaker_context_len {speaker_len} exceeds speaker_context_bucket {speaker_context_bucket}"
        )

    c_ctx_bucket = text_len + int(speaker_context_bucket)
    valid = np.zeros((1, c_ctx_bucket), dtype=np.float16)
    valid[0, :text_len] = text_mask_cpu.numpy().astype(np.float16)[0]
    valid[0, text_len : text_len + speaker_len] = speaker_mask_cpu.numpy().astype(np.float16)[0]
    return valid


def _validate_layer_context_tuple(
    layer_index: int,
    context_kv: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(context_kv) != 4:
        raise ValueError(
            f"P1a cond-only expects 4 context tensors per layer "
            f"(k_text, v_text, k_speaker, v_speaker), got {len(context_kv)} at layer {layer_index}"
        )
    k_text, v_text, k_speaker, v_speaker = context_kv
    shapes = [tuple(t.shape) for t in (k_text, v_text, k_speaker, v_speaker)]
    if any(len(shape) != 4 for shape in shapes):
        raise ValueError(f"context_kv layer {layer_index} tensors must be rank-4, got {shapes}")
    if k_text.shape != v_text.shape:
        raise ValueError(
            f"text K/V shape mismatch at layer {layer_index}: {shapes[0]} vs {shapes[1]}"
        )
    if k_speaker.shape != v_speaker.shape:
        raise ValueError(
            f"speaker K/V shape mismatch at layer {layer_index}: {shapes[2]} vs {shapes[3]}"
        )
    if k_text.shape[0] != 1 or k_speaker.shape[0] != 1:
        raise ValueError(
            "P1a cond-only state packing expects batch size 1, got "
            f"text={shapes[0]} speaker={shapes[2]}"
        )
    if k_text.shape[2:] != k_speaker.shape[2:]:
        raise ValueError(
            f"text/speaker head shape mismatch at layer {layer_index}: {shapes[0]} vs {shapes[2]}"
        )
    return k_text, v_text, k_speaker, v_speaker


def pack_context_kv_state(
    context_kv_cache: list[tuple[torch.Tensor, ...]] | tuple[tuple[torch.Tensor, ...], ...],
    *,
    text_mask: torch.Tensor,
    speaker_mask: torch.Tensor,
    speaker_context_bucket: int,
) -> PackedContextState:
    if len(context_kv_cache) <= 0:
        raise ValueError("context_kv_cache must contain at least one layer.")

    first = _validate_layer_context_tuple(0, tuple(context_kv_cache[0]))
    text_len = int(first[0].shape[1])
    speaker_len = int(first[2].shape[1])
    num_heads = int(first[0].shape[2])
    head_dim = int(first[0].shape[3])
    if speaker_len > speaker_context_bucket:
        raise ValueError(
            f"speaker_context_len {speaker_len} exceeds speaker_context_bucket {speaker_context_bucket}"
        )
    if int(text_mask.shape[1]) != text_len:
        raise ValueError(
            f"text_mask length {int(text_mask.shape[1])} does not match text_len {text_len}"
        )
    if int(speaker_mask.shape[1]) != speaker_len:
        raise ValueError(
            f"speaker_mask length {int(speaker_mask.shape[1])} does not match speaker_context_len {speaker_len}"
        )

    num_layers = int(len(context_kv_cache))
    c_ctx_bucket = text_len + int(speaker_context_bucket)
    k_state = np.zeros((num_layers, 1, c_ctx_bucket, num_heads, head_dim), dtype=np.float16)
    v_state = np.zeros_like(k_state)

    for layer_index, layer_kv in enumerate(context_kv_cache):
        k_text, v_text, k_speaker, v_speaker = _validate_layer_context_tuple(
            layer_index,
            tuple(layer_kv),
        )
        expected_text_shape = (1, text_len, num_heads, head_dim)
        expected_speaker_shape = (1, speaker_len, num_heads, head_dim)
        if tuple(k_text.shape) != expected_text_shape:
            raise ValueError(
                f"text context shape changed at layer {layer_index}: "
                f"{tuple(k_text.shape)} vs {expected_text_shape}"
            )
        if tuple(k_speaker.shape) != expected_speaker_shape:
            raise ValueError(
                f"speaker context shape changed at layer {layer_index}: "
                f"{tuple(k_speaker.shape)} vs {expected_speaker_shape}"
            )

        k_state[layer_index, :, :text_len, :, :] = (
            k_text.detach().cpu().to(dtype=torch.float16).numpy()
        )
        v_state[layer_index, :, :text_len, :, :] = (
            v_text.detach().cpu().to(dtype=torch.float16).numpy()
        )
        speaker_end = text_len + speaker_len
        k_state[layer_index, :, text_len:speaker_end, :, :] = (
            k_speaker.detach().cpu().to(dtype=torch.float16).numpy()
        )
        v_state[layer_index, :, text_len:speaker_end, :, :] = (
            v_speaker.detach().cpu().to(dtype=torch.float16).numpy()
        )

    valid_mask = build_valid_mask_state(
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        speaker_context_bucket=int(speaker_context_bucket),
    )
    return PackedContextState(
        context_k_state=k_state,
        context_v_state=v_state,
        valid_mask_state=valid_mask,
        text_len=text_len,
        speaker_context_len=speaker_len,
        speaker_context_bucket=int(speaker_context_bucket),
        c_ctx_bucket=c_ctx_bucket,
    )


def normalize_operator_category(operator_name: str) -> str | None:
    suffix = str(operator_name).split(".")[-1].lower()
    if suffix in {"linear", "matmul", "softmax"}:
        return suffix
    if suffix in {"read_state", "slice_by_index", "slice", "gather"}:
        return suffix
    return None


def _empty_count() -> dict[str, Any]:
    return {"total": 0, "ne_preferred": 0, "preferred_devices": {}}


def normalize_compute_plan_counts(raw_compute_plan_counts: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "linear": _empty_count(),
        "matmul": _empty_count(),
        "softmax": _empty_count(),
    }
    for op_name, counts in raw_compute_plan_counts.items():
        category = normalize_operator_category(op_name)
        if category not in normalized or not isinstance(counts, dict):
            continue
        dst = normalized[category]
        dst["total"] += int(counts.get("total", 0) or 0)
        dst["ne_preferred"] += int(counts.get("ne_preferred", 0) or 0)
        preferred_devices = counts.get("preferred_devices", {})
        if isinstance(preferred_devices, dict):
            for device_name, value in preferred_devices.items():
                dst["preferred_devices"][str(device_name)] = int(
                    dst["preferred_devices"].get(str(device_name), 0)
                ) + int(value or 0)
    return normalized


def _has_read_state_error(read_state_error: Any) -> bool:
    if read_state_error is None:
        return False
    if isinstance(read_state_error, dict | list | tuple | set):
        return len(read_state_error) > 0
    return True


def _max_read_state_abs_diff(read_state_max_abs_diff: Any) -> float | None:
    if read_state_max_abs_diff is None:
        return None
    if isinstance(read_state_max_abs_diff, dict):
        values = read_state_max_abs_diff.values()
    elif isinstance(read_state_max_abs_diff, list | tuple | set):
        values = read_state_max_abs_diff
    else:
        values = [read_state_max_abs_diff]

    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(numeric) if numeric else None


def _read_state_op_bytes_fp16(op: Any) -> int:
    if not isinstance(op, dict):
        return 0
    try:
        return int(op.get("bytes_fp16", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _summarize_large_read_state_ops(large_read_state_ops: Any) -> str:
    if not large_read_state_ops:
        return "large_read_state_ops=0"
    if not isinstance(large_read_state_ops, list):
        return "large_read_state_ops=unknown"
    states = sorted(
        {
            str(op.get("state"))
            for op in large_read_state_ops
            if isinstance(op, dict) and op.get("state") is not None
        }
    )
    total_bytes = 0
    for op in large_read_state_ops:
        total_bytes += _read_state_op_bytes_fp16(op)
    state_text = ",".join(states) if states else "unknown"
    return (
        f"large_read_state_ops={len(large_read_state_ops)} "
        f"states={state_text} bytes_fp16={total_bytes}"
    )


def _summarize_slice_placement(slice_placement_summary: Any) -> str:
    if not slice_placement_summary:
        return "slice_placement=none"
    if not isinstance(slice_placement_summary, dict):
        return "slice_placement=unknown"
    if "error" in slice_placement_summary:
        return f"slice_placement_error={slice_placement_summary['error']}"
    parts: list[str] = []
    for op_name, counts in sorted(slice_placement_summary.items()):
        if isinstance(counts, dict):
            parts.append(
                f"{op_name}(total={int(counts.get('total', 0) or 0)},"
                f"ne={int(counts.get('ne_preferred', 0) or 0)})"
            )
    return "slice_placement=" + (",".join(parts) if parts else "none")


def _state_layout_hint(
    *,
    state_layout: str,
    read_state_op_count: int | None,
    large_read_state_ops: Any,
    slice_placement_summary: Any,
) -> str:
    return (
        f"state_layout={normalize_state_layout(state_layout)}; "
        f"read_state_op_count={read_state_op_count}; "
        f"{_summarize_large_read_state_ops(large_read_state_ops)}; "
        f"{_summarize_slice_placement(slice_placement_summary)}"
    )


def evaluate_stateful_status(
    *,
    conversion_ok: bool,
    write_state_ok: bool,
    rel_diff: float | None,
    max_rel_diff: float,
    normalized_compute_plan_counts: dict[str, Any],
    require_ne_placement: bool,
    steady_predict_ms: float | None,
    first_predict_ms: float | None,
    write_state_ms: float | None,
    read_state_op_count: int | None,
    state_layout: str = STATE_LAYOUT_PACKED,
    num_layers: int | None = None,
    read_state_error: Any = None,
    read_state_max_abs_diff: Any = None,
    large_read_state_ops: list[dict[str, Any]] | None = None,
    read_state_counts_by_state: dict[str, Any] | None = None,
    expected_read_state_names: tuple[str, ...] | None = None,
    slice_placement_summary: Any = None,
    readback_tolerance: float = STATE_READBACK_TOLERANCE,
) -> tuple[str, list[str]]:
    normalized_state_layout = normalize_state_layout(state_layout)
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if not conversion_ok:
        fail_reasons.append("Core ML conversion failed")
    if not write_state_ok:
        fail_reasons.append("np.float32 write_state failed")
    if rel_diff is None:
        fail_reasons.append("rel_diff is unavailable")
    elif rel_diff > max_rel_diff:
        fail_reasons.append(f"rel_diff {rel_diff:.6g} exceeds max_rel_diff {max_rel_diff:.6g}")

    if require_ne_placement:
        for category in REQUIRED_NORMALIZED_NE_OPS:
            counts = normalized_compute_plan_counts.get(category)
            ne_preferred = 0
            if isinstance(counts, dict):
                ne_preferred = int(counts.get("ne_preferred", 0) or 0)
            if ne_preferred <= 0:
                fail_reasons.append(f"{category} has no NE-preferred operations")
    else:
        warn_reasons.append("NE placement requirement disabled by --no-require-ne-placement")

    if fail_reasons:
        return "FAIL", fail_reasons

    if steady_predict_ms is None:
        warn_reasons.append("steady_predict_ms is unavailable")
    elif steady_predict_ms >= NO_STATE_BASELINE_MS:
        warn_reasons.append(
            f"steady_predict_ms {steady_predict_ms:.6g} is not faster than "
            f"no-state baseline {NO_STATE_BASELINE_MS:.3f}; "
            + _state_layout_hint(
                state_layout=normalized_state_layout,
                read_state_op_count=read_state_op_count,
                large_read_state_ops=large_read_state_ops,
                slice_placement_summary=slice_placement_summary,
            )
        )
    if first_predict_ms is not None and steady_predict_ms is not None:
        if first_predict_ms > max(NO_STATE_BASELINE_MS * 2.0, steady_predict_ms * 3.0):
            warn_reasons.append(
                f"first_predict_ms {first_predict_ms:.6g} is high versus steady_predict_ms "
                f"{steady_predict_ms:.6g}"
            )
    if write_state_ms is not None and write_state_ms > 250.0:
        warn_reasons.append(f"write_state_ms {write_state_ms:.6g} is high")
    if read_state_op_count is None:
        warn_reasons.append("read_state_op_count is unavailable")
    elif read_state_op_count <= 0:
        warn_reasons.append("read_state_op_count is zero")
    if _has_read_state_error(read_state_error):
        warn_reasons.append(f"read_state_error present after write_state: {read_state_error}")

    max_readback_diff = _max_read_state_abs_diff(read_state_max_abs_diff)
    if max_readback_diff is not None and max_readback_diff > readback_tolerance:
        warn_reasons.append(
            f"read_state_max_abs_diff {max_readback_diff:.6g} exceeds tolerance "
            f"{readback_tolerance:.6g}"
        )

    large_read_state_ops = large_read_state_ops or []
    giant_read_state_ops = [
        op
        for op in large_read_state_ops
        if _read_state_op_bytes_fp16(op) >= GIANT_READ_STATE_BYTES_FP16
    ]
    if normalized_state_layout == STATE_LAYOUT_PACKED and large_read_state_ops:
        warn_reasons.append(
            "packed state layout produced large read_state ops: "
            + _state_layout_hint(
                state_layout=normalized_state_layout,
                read_state_op_count=read_state_op_count,
                large_read_state_ops=large_read_state_ops,
                slice_placement_summary=slice_placement_summary,
            )
        )
    elif normalized_state_layout == STATE_LAYOUT_PER_LAYER and giant_read_state_ops:
        warn_reasons.append(
            "per-layer state layout produced giant read_state ops: "
            + _state_layout_hint(
                state_layout=normalized_state_layout,
                read_state_op_count=read_state_op_count,
                large_read_state_ops=giant_read_state_ops,
                slice_placement_summary=slice_placement_summary,
            )
        )

    if read_state_counts_by_state:
        if expected_read_state_names is None:
            expected_read_state_names = expected_state_names(
                normalized_state_layout,
                num_layers=num_layers,
            )
        missing_states = [
            state_name
            for state_name in expected_read_state_names
            if int(read_state_counts_by_state.get(state_name, 0) or 0) <= 0
        ]
        if missing_states:
            warn_reasons.append(
                "read_state_counts_by_state missing expected states: " + ",".join(missing_states)
            )

    return ("WARN" if warn_reasons else "PASS"), warn_reasons


def compute_output_diff_metrics(
    coreml_output: np.ndarray,
    pytorch_output: np.ndarray,
) -> tuple[float, float]:
    coreml_output = np.asarray(coreml_output)
    pytorch_output = np.asarray(pytorch_output)
    if coreml_output.shape != pytorch_output.shape:
        raise ValueError(
            f"Core ML output shape {tuple(coreml_output.shape)} does not match "
            f"PyTorch output shape {tuple(pytorch_output.shape)}"
        )
    if not np.all(np.isfinite(coreml_output)):
        raise ValueError("Core ML output contains non-finite values")
    if not np.all(np.isfinite(pytorch_output)):
        raise ValueError("PyTorch output contains non-finite values")

    diff = coreml_output.astype(np.float32) - pytorch_output.astype(np.float32)
    max_abs_diff = float(np.max(np.abs(diff)))
    denom = max(float(np.max(np.abs(pytorch_output.astype(np.float32)))), 1e-8)
    rel_diff = float(max_abs_diff / denom)
    return max_abs_diff, rel_diff


def sync_device(device: torch.device) -> None:
    real_bench.sync_device(device)


def benchmark_torch_cached_step(
    model: TextToLatentRFDiT,
    inputs: tuple[torch.Tensor, ...],
    context_kv_cache: list[tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    x_t, t, text_state, text_mask, speaker_state, speaker_mask, latent_mask = inputs
    with torch.inference_mode():
        for _ in range(warmup):
            model.forward_with_encoded_conditions(
                x_t=x_t,
                t=t,
                text_state=text_state,
                text_mask=text_mask,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                latent_mask=latent_mask,
                context_kv_cache=context_kv_cache,
            )
        sync_device(device)
        start = time.perf_counter()
        output = None
        for _ in range(iterations):
            output = model.forward_with_encoded_conditions(
                x_t=x_t,
                t=t,
                text_state=text_state,
                text_mask=text_mask,
                speaker_state=speaker_state,
                speaker_mask=speaker_mask,
                latent_mask=latent_mask,
                context_kv_cache=context_kv_cache,
            )
        sync_device(device)
    if output is None:
        raise RuntimeError("No PyTorch cached benchmark iterations ran.")
    avg_ms = (time.perf_counter() - start) * 1000.0 / float(iterations)
    return output.detach().cpu().numpy(), avg_ms


class RealCoreMLStatefulDenoiserStep(nn.Module):
    """
    Benchmark-local stateful one-step wrapper.

    The production TextToLatentRFDiT modules are reused as-is. Only the conversion
    blockers in the denoiser step are rewritten locally: complex RoPE, SDPA, bool
    masks, and context KV normal inputs. Text+speaker context KV comes from MLState.
    """

    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        sequence_length: int,
        num_layers: int,
        c_ctx_bucket: int,
        state_layout: str = STATE_LAYOUT_PACKED,
    ):
        super().__init__()
        normalized_state_layout = normalize_state_layout(state_layout)
        if model.cfg.use_caption_condition:
            raise NotImplementedError(
                "P1a cond-only benchmark supports the speaker-conditioned default checkpoint only."
            )
        if not model.cfg.use_speaker_condition:
            raise ValueError("DEFAULT_CHECKPOINT is expected to use speaker conditioning.")
        if int(model.cfg.num_layers) != int(num_layers):
            raise ValueError(f"num_layers mismatch: cfg={model.cfg.num_layers} packed={num_layers}")

        self.cfg = model.cfg
        self.sequence_length = int(sequence_length)
        self.model_dim = int(model.cfg.model_dim)
        self.heads = int(model.cfg.num_heads)
        self.head_dim = self.model_dim // self.heads
        self.norm_eps = float(model.cfg.norm_eps)
        self.num_layers = int(num_layers)
        self.c_ctx_bucket = int(c_ctx_bucket)
        self.state_layout = normalized_state_layout

        self.cond_module = model.cond_module
        self.in_proj = model.in_proj
        self.blocks = model.blocks
        self.out_norm = model.out_norm
        self.out_proj = model.out_proj

        timestep_half = int(model.cfg.timestep_embed_dim) // 2
        timestep_freqs = 1000.0 * torch.exp(
            -math.log(10000.0) * torch.arange(timestep_half, dtype=torch.float32) / timestep_half
        )
        self.register_buffer("timestep_freqs", timestep_freqs, persistent=False)

        rope_cos, rope_sin = real_bench.build_rope_cache(
            sequence_length=self.sequence_length,
            head_dim=self.head_dim,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        if self.state_layout == STATE_LAYOUT_PACKED:
            state_shape = (self.num_layers, 1, self.c_ctx_bucket, self.heads, self.head_dim)
            self.register_buffer(
                "context_k_state",
                torch.zeros(state_shape, dtype=torch.float16),
            )
            self.register_buffer(
                "context_v_state",
                torch.zeros(state_shape, dtype=torch.float16),
            )
        else:
            state_shape = (1, self.c_ctx_bucket, self.heads, self.head_dim)
            for layer_index in range(self.num_layers):
                self.register_buffer(
                    f"context_k_l{layer_index:02d}",
                    torch.zeros(state_shape, dtype=torch.float16),
                )
                self.register_buffer(
                    f"context_v_l{layer_index:02d}",
                    torch.zeros(state_shape, dtype=torch.float16),
                )
        self.register_buffer(
            "valid_mask_state",
            torch.zeros((1, self.c_ctx_bucket), dtype=torch.float16),
        )
        self._assert_state_buffer_dtypes()

    def state_buffer_names(self) -> tuple[str, ...]:
        return expected_state_names(self.state_layout, num_layers=self.num_layers)

    def iter_state_buffers(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name in self.state_buffer_names():
            yield name, getattr(self, name)

    def _assert_state_buffer_dtypes(self) -> None:
        for name in self.state_buffer_names():
            value = getattr(self, name)
            if value.dtype != torch.float16:
                raise TypeError(f"{name} must be registered as torch.float16, got {value.dtype}")

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.timestep_freqs[None, :]
        return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)

    def low_rank_adaln(
        self,
        adaln: nn.Module,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = cond_embed.chunk(3, dim=-1)
        shift = adaln.shift_up(adaln.shift_down(F.silu(shift))) + shift
        scale = adaln.scale_up(adaln.scale_down(F.silu(scale))) + scale
        gate = adaln.gate_up(adaln.gate_down(F.silu(gate))) + gate
        y = real_bench.rms_norm_unit(x, adaln.eps)
        y = y * (1.0 + scale) + shift
        return y, torch.tanh(gate)

    def joint_attention(
        self,
        attention: nn.Module,
        x: torch.Tensor,
        latent_mask_f: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        bsz = x.shape[0]
        seq_len = x.shape[1]
        q = attention.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_self = attention.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v_self = attention.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)

        q = real_bench.rms_norm(q, attention.q_norm.weight, attention.q_norm.eps)
        k_self = real_bench.rms_norm(k_self, attention.k_norm.weight, attention.k_norm.eps)
        q = real_bench.apply_real_rope_half_heads(q, self.rope_cos, self.rope_sin)
        k_self = real_bench.apply_real_rope_half_heads(k_self, self.rope_cos, self.rope_sin)

        if self.state_layout == STATE_LAYOUT_PACKED:
            context_k = self.context_k_state[layer_index].to(dtype=k_self.dtype)
            context_v = self.context_v_state[layer_index].to(dtype=v_self.dtype)
        else:
            context_k = getattr(self, f"context_k_l{layer_index:02d}").to(dtype=k_self.dtype)
            context_v = getattr(self, f"context_v_l{layer_index:02d}").to(dtype=v_self.dtype)
        valid_mask = self.valid_mask_state.to(dtype=latent_mask_f.dtype)

        k = torch.cat((k_self, context_k), dim=1).transpose(1, 2)
        v = torch.cat((v_self, context_v), dim=1).transpose(1, 2)
        q = q.transpose(1, 2)
        key_mask_f = torch.cat((latent_mask_f, valid_mask), dim=1)
        y = real_bench.explicit_scaled_dot_product_attention(q, k, v, key_mask_f)
        y = y.transpose(1, 2).reshape(bsz, seq_len, self.model_dim)
        y = y * torch.sigmoid(attention.gate(x))
        return attention.wo(y)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_mask_f: torch.Tensor,
    ) -> torch.Tensor:
        t_embed = self.timestep_embedding(t).to(dtype=x_t.dtype)
        cond_embed = self.cond_module(t_embed)[:, None, :]

        x = self.in_proj(x_t)
        for layer_index, block in enumerate(self.blocks):
            h, attention_gate = self.low_rank_adaln(block.attention_adaln, x, cond_embed)
            y = self.joint_attention(
                attention=block.attention,
                x=h,
                latent_mask_f=latent_mask_f,
                layer_index=layer_index,
            )
            x = x + attention_gate * y

            h_mlp, mlp_gate = self.low_rank_adaln(block.mlp_adaln, x, cond_embed)
            x = x + mlp_gate * block.mlp(h_mlp)

        x = real_bench.rms_norm(x, self.out_norm.weight, self.out_norm.eps)
        return self.out_proj(x)


def quiet_convert(ct: Any, *args: Any, verbose: bool = False, **kwargs: Any) -> Any:
    if verbose:
        return ct.convert(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return ct.convert(*args, **kwargs)


def io_numpy_dtype(io_dtype: str) -> Any:
    if io_dtype == "float32":
        return np.float32
    if io_dtype == "float16":
        return np.float16
    raise ValueError(f"Unsupported io dtype: {io_dtype!r}")


def cpu_stateful_inputs(
    inputs: tuple[torch.Tensor, ...],
    *,
    io_dtype: str,
) -> tuple[tuple[torch.Tensor, ...], dict[str, np.ndarray]]:
    x_t, t, _text_state, _text_mask, _speaker_state, _speaker_mask, latent_mask = inputs
    trace_inputs = (
        x_t.detach().cpu().to(dtype=torch.float32),
        t.detach().cpu().to(dtype=torch.float32),
        latent_mask.detach().cpu().to(dtype=torch.float32),
    )
    dtype = io_numpy_dtype(io_dtype)
    predict_inputs = {
        "x_t": trace_inputs[0].numpy().astype(dtype, copy=False),
        "t": trace_inputs[1].numpy().astype(dtype, copy=False),
        "latent_mask_f": trace_inputs[2].numpy().astype(dtype, copy=False),
    }
    return trace_inputs, predict_inputs


def convert_wrapper(
    ct: Any,
    wrapper: RealCoreMLStatefulDenoiserStep,
    trace_inputs: tuple[torch.Tensor, ...],
    *,
    artifact_dir: Path,
    compute_precision: Any | None,
    io_dtype: str,
    verbose: bool,
) -> tuple[Any, float]:
    input_dtype = io_numpy_dtype(io_dtype)
    input_types = [
        ct.TensorType(name="x_t", shape=trace_inputs[0].shape, dtype=input_dtype),
        ct.TensorType(name="t", shape=trace_inputs[1].shape, dtype=input_dtype),
        ct.TensorType(name="latent_mask_f", shape=trace_inputs[2].shape, dtype=input_dtype),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=tuple(buffer.shape), dtype=np.float16),
            name=name,
        )
        for name, buffer in wrapper.iter_state_buffers()
    ]
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper.eval(), trace_inputs, check_trace=False)

    compute_unit = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    if compute_unit is None:
        raise RuntimeError("coremltools does not expose ComputeUnit.CPU_AND_NE.")

    package_path = artifact_dir / "stateful_step.mlpackage"
    print(f"[artifacts] mlpackage: {package_path}", flush=True)
    convert_kwargs = {
        "inputs": input_types,
        "states": states,
        "convert_to": "mlprogram",
        "minimum_deployment_target": ct.target.iOS18,
        "compute_units": compute_unit,
        "package_dir": str(package_path),
    }
    if compute_precision is not None:
        convert_kwargs["compute_precision"] = compute_precision

    start = time.perf_counter()
    mlmodel = quiet_convert(ct, traced, **convert_kwargs, verbose=verbose)
    return mlmodel, time.perf_counter() - start


def iter_mil_operations(block: Any) -> Iterator[Any]:
    for operation in block.operations:
        yield operation
        for nested_block in getattr(operation, "blocks", []):
            yield from iter_mil_operations(nested_block)


def _shape_to_int_tuple(shape: Any) -> tuple[int, ...] | None:
    try:
        values = tuple(int(dim) for dim in shape)
    except Exception:
        return None
    if any(dim < 0 for dim in values):
        return None
    return values


def mil_state_op_summary(mlmodel: Any) -> dict[str, Any]:
    try:
        program = mlmodel._mil_program
        main = program.functions["main"]
    except Exception as exc:
        return {
            "read_state_op_count": None,
            "read_state_counts_by_state": {},
            "large_read_state_ops": [],
            "max_read_state_bytes_fp16": None,
            "error": f"{type(exc).__name__}: {first_line(exc)}",
        }

    read_state_count = 0
    counts_by_state: Counter[str] = Counter()
    large_read_state_ops: list[dict[str, Any]] = []
    max_read_state_bytes_fp16 = 0
    for op in iter_mil_operations(main):
        if getattr(op, "op_type", None) != "read_state":
            continue
        read_state_count += 1
        input_var = getattr(op, "inputs", {}).get("input")
        state_name = getattr(input_var, "name", "unknown")
        if state_name.endswith("_workaround"):
            state_name = state_name[: -len("_workaround")]
        counts_by_state[str(state_name)] += 1
        outputs = getattr(op, "outputs", [])
        if outputs:
            shape = _shape_to_int_tuple(getattr(outputs[0], "shape", ()))
            if shape is not None:
                numel = math.prod(shape)
                bytes_fp16 = int(numel * 2)
                max_read_state_bytes_fp16 = max(max_read_state_bytes_fp16, bytes_fp16)
                if bytes_fp16 >= LARGE_READ_STATE_BYTES_FP16:
                    large_read_state_ops.append(
                        {
                            "name": str(getattr(op, "name", "")),
                            "state": str(state_name),
                            "shape": list(shape),
                            "bytes_fp16": bytes_fp16,
                        }
                    )

    return {
        "read_state_op_count": int(read_state_count),
        "read_state_counts_by_state": dict(counts_by_state),
        "large_read_state_ops": large_read_state_ops,
        "max_read_state_bytes_fp16": int(max_read_state_bytes_fp16),
    }


def iter_mlprogram_operations(block: Any) -> Iterator[Any]:
    for operation in block.operations:
        yield operation
        for nested_block in operation.blocks:
            yield from iter_mlprogram_operations(nested_block)


def compute_plan_report(ct: Any, mlmodel: Any, compute_unit: Any) -> dict[str, Any]:
    try:
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice
        from coremltools.models.compute_plan import MLComputePlan
    except Exception as exc:
        error = f"{type(exc).__name__}: {first_line(exc)}"
        return {
            "raw_compute_plan_counts": {"error": error},
            "normalized_compute_plan_counts": normalize_compute_plan_counts({}),
            "slice_placement_summary": {"error": error},
        }

    try:
        plan = MLComputePlan.load_from_path(
            mlmodel.get_compiled_model_path(),
            compute_units=compute_unit,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {first_line(exc)}"
        return {
            "raw_compute_plan_counts": {"error": error},
            "normalized_compute_plan_counts": normalize_compute_plan_counts({}),
            "slice_placement_summary": {"error": error},
        }

    program = plan.model_structure.program
    if program is None:
        error = "compiled model is not an ML Program"
        return {
            "raw_compute_plan_counts": {"error": error},
            "normalized_compute_plan_counts": normalize_compute_plan_counts({}),
            "slice_placement_summary": {"error": error},
        }

    total_by_op: Counter[str] = Counter()
    ne_by_op: Counter[str] = Counter()
    devices_by_op: dict[str, Counter[str]] = {}
    slice_summary: dict[str, Any] = {}

    main_function = program.functions["main"]
    for operation in iter_mlprogram_operations(main_function.block):
        op_name = str(operation.operator_name)
        total_by_op[op_name] += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        device = None if usage is None else usage.preferred_compute_device
        device_name = "Unknown" if device is None else type(device).__name__
        devices_by_op.setdefault(op_name, Counter())[device_name] += 1
        if isinstance(device, MLNeuralEngineComputeDevice):
            ne_by_op[op_name] += 1

    raw: dict[str, Any] = {}
    for op_name in sorted(total_by_op):
        raw[op_name] = {
            "total": int(total_by_op[op_name]),
            "ne_preferred": int(ne_by_op[op_name]),
            "preferred_devices": dict(devices_by_op.get(op_name, Counter())),
        }
        suffix = normalize_operator_category(op_name)
        if suffix in {"slice_by_index", "slice", "gather"}:
            slice_summary[op_name] = raw[op_name]

    return {
        "raw_compute_plan_counts": raw,
        "normalized_compute_plan_counts": normalize_compute_plan_counts(raw),
        "slice_placement_summary": slice_summary,
    }


def make_and_write_state(
    mlmodel: Any,
    packed: PackedContextState,
    *,
    state_layout: str = STATE_LAYOUT_PACKED,
) -> tuple[Any, dict[str, Any]]:
    state_payloads = state_write_payloads_from_packed(packed, state_layout=state_layout)

    start = time.perf_counter()
    state = mlmodel.make_state()
    make_state_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    for name, payload in state_payloads.items():
        state.write_state(name=name, value=payload)
    write_state_ms = (time.perf_counter() - start) * 1000.0

    read_state_dtype: dict[str, str] = {}
    read_state_max_abs_diff: dict[str, float] = {}
    read_state_shape: dict[str, list[int]] = {}
    read_state_error: dict[str, str] = {}
    for name, payload in state_payloads.items():
        try:
            read_value = np.asarray(state.read_state(name=name))
            read_state_dtype[name] = str(read_value.dtype)
            read_state_shape[name] = [int(dim) for dim in read_value.shape]
            read_state_max_abs_diff[name] = float(
                np.max(np.abs(read_value.astype(np.float32) - payload.astype(np.float32)))
            )
        except Exception as exc:
            read_state_error[name] = f"{type(exc).__name__}: {first_line(exc)}"

    return state, {
        "make_state_ms": float(make_state_ms),
        "write_state_ms": float(write_state_ms),
        "write_state_payload_dtype": "float32",
        "read_state_dtype": read_state_dtype or None,
        "read_state_shape": read_state_shape or None,
        "read_state_max_abs_diff": read_state_max_abs_diff or None,
        "read_state_error": read_state_error or None,
    }


def benchmark_coreml_stateful_predict(
    mlmodel: Any,
    predict_inputs: dict[str, np.ndarray],
    state: Any,
    *,
    warmup: int,
    iterations: int,
) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    prediction = mlmodel.predict(predict_inputs, state=state)
    first_predict_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        prediction = mlmodel.predict(predict_inputs, state=state)
    start = time.perf_counter()
    for _ in range(iterations):
        prediction = mlmodel.predict(predict_inputs, state=state)
    steady_predict_ms = (time.perf_counter() - start) * 1000.0 / float(iterations)
    if prediction is None:
        raise RuntimeError("No Core ML stateful prediction iterations ran.")
    return (
        np.asarray(next(iter(prediction.values()))),
        float(first_predict_ms),
        float(steady_predict_ms),
    )


@contextlib.contextmanager
def temporary_artifact_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="coreml-stateful-step-benchmark-") as tmpdir:
        yield Path(tmpdir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one real Irodori-TTS denoiser step with text+speaker context KV "
            "stored in Core ML MLState. Generated artifacts stay in a temporary directory."
        )
    )
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT, help="Text to tokenize.")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--speaker-context-bucket", type=int, default=160)
    parser.add_argument(
        "--compute-precision",
        choices=("default", "float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--io-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Normal prediction input dtype. float32 is the P1a baseline; float16 is experimental.",
    )
    parser.add_argument(
        "--state-layout",
        choices=("packed", "per-layer"),
        default="packed",
        help="MLState layout for text+speaker context KV.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--ref-wav", type=Path, default=DEFAULT_REF_WAV)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument(
        "--no-require-ne-placement",
        action="store_true",
        help="Do not fail status when normalized linear/matmul/softmax lack NE-preferred placement.",
    )
    parser.add_argument("--max-rel-diff", type=float, default=0.05)
    parser.add_argument("--mode", choices=(MODE_COND_ONLY, "independent3"), default=MODE_COND_ONLY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-convert", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    normalize_state_layout(args.state_layout)
    if args.mode != MODE_COND_ONLY:
        raise ValueError("--mode independent3 is not implemented for P1a; use --mode cond-only.")
    if args.seconds <= 0:
        raise ValueError(f"--seconds must be > 0, got {args.seconds}")
    if args.sequence_length is not None and args.sequence_length <= 0:
        raise ValueError(f"--sequence-length must be > 0, got {args.sequence_length}")
    if args.speaker_context_bucket < 0:
        raise ValueError(
            f"--speaker-context-bucket must be >= 0, got {args.speaker_context_bucket}"
        )
    if args.iterations <= 0:
        raise ValueError(f"--iterations must be > 0, got {args.iterations}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    if args.max_rel_diff < 0:
        raise ValueError(f"--max-rel-diff must be >= 0, got {args.max_rel_diff}")


def _base_failure_summary(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    require_ne_placement = not bool(args.no_require_ne_placement)
    state_layout = state_layout_arg(args)
    return {
        "sequence_length": None,
        "text_len": None,
        "speaker_context_len": None,
        "speaker_context_bucket": int(args.speaker_context_bucket),
        "c_ctx_bucket": None,
        "state_kv_bytes": None,
        "state_layout": state_layout,
        "io_dtype": str(args.io_dtype),
        "convert_seconds": None,
        "make_state_ms": None,
        "write_state_ms": None,
        "first_predict_ms": None,
        "steady_predict_ms": None,
        "pytorch_mps_cached_avg_ms": None,
        "speedup_vs_pytorch_cached": None,
        "max_abs_diff": None,
        "rel_diff": None,
        "raw_compute_plan_counts": {},
        "normalized_compute_plan_counts": normalize_compute_plan_counts({}),
        "read_state_op_count": None,
        "read_state_counts_by_state": {},
        "large_read_state_ops": [],
        "max_read_state_bytes_fp16": None,
        "slice_placement_summary": {},
        "status": "FAIL",
        "status_reasons": [reason],
        "mode": str(args.mode),
        "compute_precision": str(args.compute_precision),
        "require_ne_placement": require_ne_placement,
    }


def _available_failure_summary(
    args: argparse.Namespace,
    reason: str,
    *,
    metadata: dict[str, Any] | None = None,
    packed: PackedContextState | None = None,
    state_kv_bytes: int | None = None,
    compute_precision_name: str | None = None,
    require_ne_placement: bool | None = None,
    convert_seconds: float | None = None,
    pytorch_mps_cached_avg_ms: float | None = None,
    raw_compute_plan_counts: dict[str, Any] | None = None,
    normalized_compute_plan_counts: dict[str, Any] | None = None,
    read_state_op_count: int | None = None,
    read_state_counts_by_state: dict[str, Any] | None = None,
    large_read_state_ops: list[dict[str, Any]] | None = None,
    max_read_state_bytes_fp16: int | None = None,
    slice_placement_summary: dict[str, Any] | None = None,
    state_metrics: dict[str, Any] | None = None,
    first_predict_ms: float | None = None,
    steady_predict_ms: float | None = None,
    max_abs_diff: float | None = None,
    rel_diff: float | None = None,
) -> dict[str, Any]:
    summary = _base_failure_summary(args, reason)
    state_layout = state_layout_arg(args)
    if metadata is not None:
        summary.update(
            {
                "sequence_length": int(metadata["sequence_length"]),
                "seconds_derived_sequence_length": int(metadata["seconds_derived_sequence_length"]),
                "sequence_length_source": str(metadata["sequence_length_source"]),
                "text_valid_tokens": int(metadata["text_valid_tokens"]),
                "ref_len": int(metadata["ref_len"]),
            }
        )
    if packed is not None:
        summary.update(
            {
                "text_len": int(packed.text_len),
                "speaker_context_len": int(packed.speaker_context_len),
                "speaker_context_bucket": int(packed.speaker_context_bucket),
                "c_ctx_bucket": int(packed.c_ctx_bucket),
                "state_layout_shape": state_payload_shapes(packed, state_layout=state_layout),
            }
        )
    if state_kv_bytes is not None:
        summary["state_kv_bytes"] = int(state_kv_bytes)
    if compute_precision_name is not None:
        summary["compute_precision"] = str(compute_precision_name)
    if require_ne_placement is not None:
        summary["require_ne_placement"] = bool(require_ne_placement)
    if convert_seconds is not None:
        summary["convert_seconds"] = float(convert_seconds)
    if pytorch_mps_cached_avg_ms is not None:
        summary["pytorch_mps_cached_avg_ms"] = float(pytorch_mps_cached_avg_ms)
    if raw_compute_plan_counts is not None:
        summary["raw_compute_plan_counts"] = raw_compute_plan_counts
    if normalized_compute_plan_counts is not None:
        summary["normalized_compute_plan_counts"] = normalized_compute_plan_counts
    if read_state_op_count is not None:
        summary["read_state_op_count"] = int(read_state_op_count)
    if read_state_counts_by_state is not None:
        summary["read_state_counts_by_state"] = read_state_counts_by_state
    if large_read_state_ops is not None:
        summary["large_read_state_ops"] = large_read_state_ops
    if max_read_state_bytes_fp16 is not None:
        summary["max_read_state_bytes_fp16"] = int(max_read_state_bytes_fp16)
    if slice_placement_summary is not None:
        summary["slice_placement_summary"] = slice_placement_summary
    if state_metrics is not None:
        for key in (
            "make_state_ms",
            "write_state_ms",
            "write_state_payload_dtype",
            "read_state_dtype",
            "read_state_shape",
            "read_state_max_abs_diff",
            "read_state_error",
        ):
            if key in state_metrics:
                summary[key] = state_metrics.get(key)
    if first_predict_ms is not None:
        summary["first_predict_ms"] = float(first_predict_ms)
    if steady_predict_ms is not None:
        summary["steady_predict_ms"] = float(steady_predict_ms)
    if max_abs_diff is not None:
        summary["max_abs_diff"] = float(max_abs_diff)
    if rel_diff is not None:
        summary["rel_diff"] = float(rel_diff)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    state_layout = state_layout_arg(args)
    require_ne_placement = not bool(args.no_require_ne_placement)

    ct = real_bench.import_coremltools()
    compute_precision, compute_precision_name = real_bench.resolve_compute_precision(
        ct,
        args.compute_precision,
    )
    if platform.system() != "Darwin":
        raise RuntimeError("Core ML prediction and PyTorch MPS baseline require macOS.")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "PyTorch MPS baseline requested but torch.backends.mps.is_available() is False."
        )

    mps_device = torch.device("mps")
    checkpoint_path = real_bench.resolve_checkpoint_path(args.checkpoint)
    print(f"[mode] {MODE_COND_ONLY} state_layout={state_layout}", flush=True)
    print(
        f"[convert] compute_precision={compute_precision_name} io_dtype={args.io_dtype}", flush=True
    )
    print(f"[coremltools] {ct.__version__}", flush=True)
    print(f"[torch] {torch.__version__}", flush=True)

    print("[load] actual checkpoint/model weights on MPS", flush=True)
    model, model_cfg, train_cfg = real_bench.load_actual_model(checkpoint_path, mps_device)

    print(
        "[prepare] tokenizer, codec-derived length, rem.wav reference, encoded conditions",
        flush=True,
    )
    inputs_mps, metadata = real_bench.prepare_real_inputs(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        checkpoint=args.checkpoint,
        codec_repo=args.codec_repo,
        ref_wav=args.ref_wav,
        text=args.text,
        seconds=float(args.seconds),
        sequence_length_override=args.sequence_length,
        seed=int(args.seed),
        device=mps_device,
    )
    print(
        "[shape] sequence_length={sequence_length} source={sequence_length_source} "
        "seconds_derived={seconds_derived_sequence_length} text_len={text_len} "
        "ref_len={ref_len} speaker_context_len={speaker_context_len}".format(**metadata),
        flush=True,
    )

    _x_t, _t, text_state, text_mask, speaker_state, speaker_mask, _latent_mask = inputs_mps
    print("[cache] model.build_context_kv_cache(text_state, speaker_state)", flush=True)
    with torch.inference_mode():
        context_kv_cache = model.build_context_kv_cache(text_state, speaker_state)
    packed = pack_context_kv_state(
        context_kv_cache,
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        speaker_context_bucket=int(args.speaker_context_bucket),
    )
    num_layers = int(len(context_kv_cache))
    num_heads = int(context_kv_cache[0][0].shape[2])
    head_dim = int(context_kv_cache[0][0].shape[3])
    state_kv_bytes = compute_state_kv_bytes(
        num_layers=num_layers,
        c_ctx_bucket=packed.c_ctx_bucket,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    print(
        f"[state] L={num_layers} H={num_heads} D={head_dim} "
        f"C_ctx_bucket={packed.c_ctx_bucket} state_kv_bytes={state_kv_bytes}",
        flush=True,
    )

    print(
        "[benchmark] PyTorch MPS forward_with_encoded_conditions with context_kv_cache", flush=True
    )
    pytorch_output, pytorch_cached_avg_ms = benchmark_torch_cached_step(
        model,
        inputs_mps,
        context_kv_cache,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
        device=mps_device,
    )
    trace_inputs, predict_inputs = cpu_stateful_inputs(inputs_mps, io_dtype=str(args.io_dtype))

    print("[convert] moving denoiser weights to CPU for TorchScript/Core ML conversion", flush=True)
    model = model.to(device="cpu")
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    wrapper = RealCoreMLStatefulDenoiserStep(
        model,
        sequence_length=int(metadata["sequence_length"]),
        num_layers=num_layers,
        c_ctx_bucket=packed.c_ctx_bucket,
        state_layout=state_layout,
    ).eval()

    convert_seconds: float | None = None
    raw_compute_plan_counts: dict[str, Any] = {}
    normalized_compute_plan_counts: dict[str, Any] = normalize_compute_plan_counts({})
    slice_placement_summary: dict[str, Any] = {}
    read_state_op_count: int | None = None
    read_state_counts_by_state: dict[str, Any] = {}
    large_read_state_ops: list[dict[str, Any]] = []
    max_read_state_bytes_fp16: int | None = None
    state_metrics: dict[str, Any] = {
        "make_state_ms": None,
        "write_state_ms": None,
        "write_state_payload_dtype": None,
        "read_state_dtype": None,
    }

    with temporary_artifact_dir() as artifact_dir:
        print(f"[artifacts] temporary directory: {artifact_dir}", flush=True)
        try:
            mlmodel, convert_seconds = convert_wrapper(
                ct,
                wrapper,
                trace_inputs,
                artifact_dir=artifact_dir,
                compute_precision=compute_precision,
                io_dtype=str(args.io_dtype),
                verbose=bool(args.verbose_convert),
            )
        except Exception as exc:
            reason = f"Core ML conversion failed: {type(exc).__name__}: {first_line(exc)}"
            summary = _base_failure_summary(args, reason)
            summary.update(
                {
                    "sequence_length": int(metadata["sequence_length"]),
                    "text_len": int(metadata["text_len"]),
                    "speaker_context_len": int(packed.speaker_context_len),
                    "c_ctx_bucket": int(packed.c_ctx_bucket),
                    "state_kv_bytes": int(state_kv_bytes),
                    "state_layout_shape": state_payload_shapes(packed, state_layout=state_layout),
                    "pytorch_mps_cached_avg_ms": float(pytorch_cached_avg_ms),
                    "compute_precision": compute_precision_name,
                    "require_ne_placement": require_ne_placement,
                }
            )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
            return 1
        print(
            f"[convert] Core ML stateful mlprogram CPU_AND_NE conversion: {convert_seconds:.3f} s",
            flush=True,
        )

        mil_summary = mil_state_op_summary(mlmodel)
        read_state_op_count = mil_summary.get("read_state_op_count")
        read_state_counts_by_state = mil_summary.get("read_state_counts_by_state", {})
        large_read_state_ops = mil_summary.get("large_read_state_ops", [])
        max_read_state_bytes_fp16 = mil_summary.get("max_read_state_bytes_fp16")

        compute_unit = ct.ComputeUnit.CPU_AND_NE
        plan_report = compute_plan_report(ct, mlmodel, compute_unit)
        raw_compute_plan_counts = plan_report["raw_compute_plan_counts"]
        normalized_compute_plan_counts = plan_report["normalized_compute_plan_counts"]
        slice_placement_summary = plan_report["slice_placement_summary"]
        print(
            "[compute_plan] "
            + json.dumps(
                {
                    "raw_compute_plan_counts": raw_compute_plan_counts,
                    "normalized_compute_plan_counts": normalized_compute_plan_counts,
                    "read_state_op_count": read_state_op_count,
                    "read_state_counts_by_state": read_state_counts_by_state,
                    "large_read_state_ops": large_read_state_ops,
                    "max_read_state_bytes_fp16": max_read_state_bytes_fp16,
                    "slice_placement_summary": slice_placement_summary,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

        try:
            print("[state] make_state + np.float32 write_state", flush=True)
            state, state_metrics = make_and_write_state(
                mlmodel,
                packed,
                state_layout=state_layout,
            )
        except Exception as exc:
            reason = f"np.float32 write_state failed: {type(exc).__name__}: {first_line(exc)}"
            summary = _base_failure_summary(args, reason)
            summary.update(
                {
                    "sequence_length": int(metadata["sequence_length"]),
                    "text_len": int(metadata["text_len"]),
                    "speaker_context_len": int(packed.speaker_context_len),
                    "speaker_context_bucket": int(packed.speaker_context_bucket),
                    "c_ctx_bucket": int(packed.c_ctx_bucket),
                    "state_kv_bytes": int(state_kv_bytes),
                    "state_layout_shape": state_payload_shapes(packed, state_layout=state_layout),
                    "convert_seconds": float(convert_seconds),
                    "pytorch_mps_cached_avg_ms": float(pytorch_cached_avg_ms),
                    "raw_compute_plan_counts": raw_compute_plan_counts,
                    "normalized_compute_plan_counts": normalized_compute_plan_counts,
                    "read_state_op_count": read_state_op_count,
                    "read_state_counts_by_state": read_state_counts_by_state,
                    "large_read_state_ops": large_read_state_ops,
                    "max_read_state_bytes_fp16": max_read_state_bytes_fp16,
                    "slice_placement_summary": slice_placement_summary,
                    "compute_precision": compute_precision_name,
                    "require_ne_placement": require_ne_placement,
                }
            )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
            return 1
        print(
            "[state] "
            + json.dumps(
                {
                    key: state_metrics.get(key)
                    for key in (
                        "make_state_ms",
                        "write_state_ms",
                        "write_state_payload_dtype",
                        "read_state_dtype",
                        "read_state_error",
                    )
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

        print("[benchmark] Core ML stateful first predict + steady predict", flush=True)
        try:
            coreml_output, first_predict_ms, steady_predict_ms = benchmark_coreml_stateful_predict(
                mlmodel,
                predict_inputs,
                state,
                warmup=int(args.warmup),
                iterations=int(args.iterations),
            )
        except Exception as exc:
            reason = f"Core ML predict failed: {type(exc).__name__}: {first_line(exc)}"
            summary = _available_failure_summary(
                args,
                reason,
                metadata=metadata,
                packed=packed,
                state_kv_bytes=state_kv_bytes,
                compute_precision_name=compute_precision_name,
                require_ne_placement=require_ne_placement,
                convert_seconds=convert_seconds,
                pytorch_mps_cached_avg_ms=pytorch_cached_avg_ms,
                raw_compute_plan_counts=raw_compute_plan_counts,
                normalized_compute_plan_counts=normalized_compute_plan_counts,
                read_state_op_count=read_state_op_count,
                read_state_counts_by_state=read_state_counts_by_state,
                large_read_state_ops=large_read_state_ops,
                max_read_state_bytes_fp16=max_read_state_bytes_fp16,
                slice_placement_summary=slice_placement_summary,
                state_metrics=state_metrics,
            )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
            return 1

    try:
        max_abs_diff, rel_diff = compute_output_diff_metrics(coreml_output, pytorch_output)
    except Exception as exc:
        reason = f"Core ML output validation failed: {type(exc).__name__}: {first_line(exc)}"
        summary = _available_failure_summary(
            args,
            reason,
            metadata=metadata,
            packed=packed,
            state_kv_bytes=state_kv_bytes,
            compute_precision_name=compute_precision_name,
            require_ne_placement=require_ne_placement,
            convert_seconds=convert_seconds,
            pytorch_mps_cached_avg_ms=pytorch_cached_avg_ms,
            raw_compute_plan_counts=raw_compute_plan_counts,
            normalized_compute_plan_counts=normalized_compute_plan_counts,
            read_state_op_count=read_state_op_count,
            read_state_counts_by_state=read_state_counts_by_state,
            large_read_state_ops=large_read_state_ops,
            max_read_state_bytes_fp16=max_read_state_bytes_fp16,
            slice_placement_summary=slice_placement_summary,
            state_metrics=state_metrics,
            first_predict_ms=first_predict_ms,
            steady_predict_ms=steady_predict_ms,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 1
    speedup = float(pytorch_cached_avg_ms / steady_predict_ms) if steady_predict_ms > 0 else None

    status, status_reasons = evaluate_stateful_status(
        conversion_ok=True,
        write_state_ok=True,
        rel_diff=rel_diff,
        max_rel_diff=float(args.max_rel_diff),
        normalized_compute_plan_counts=normalized_compute_plan_counts,
        require_ne_placement=require_ne_placement,
        steady_predict_ms=steady_predict_ms,
        first_predict_ms=first_predict_ms,
        write_state_ms=state_metrics.get("write_state_ms"),
        read_state_op_count=read_state_op_count,
        state_layout=state_layout,
        num_layers=num_layers,
        read_state_error=state_metrics.get("read_state_error"),
        read_state_max_abs_diff=state_metrics.get("read_state_max_abs_diff"),
        large_read_state_ops=large_read_state_ops,
        read_state_counts_by_state=read_state_counts_by_state,
        slice_placement_summary=slice_placement_summary,
    )
    print(
        "[status] "
        + json.dumps(
            {"status": status, "status_reasons": status_reasons},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    summary = {
        "sequence_length": int(metadata["sequence_length"]),
        "text_len": int(packed.text_len),
        "speaker_context_len": int(packed.speaker_context_len),
        "speaker_context_bucket": int(packed.speaker_context_bucket),
        "c_ctx_bucket": int(packed.c_ctx_bucket),
        "state_kv_bytes": int(state_kv_bytes),
        "state_layout": state_layout,
        "state_layout_shape": state_payload_shapes(packed, state_layout=state_layout),
        "io_dtype": str(args.io_dtype),
        "convert_seconds": float(convert_seconds),
        "make_state_ms": state_metrics.get("make_state_ms"),
        "write_state_ms": state_metrics.get("write_state_ms"),
        "write_state_payload_dtype": state_metrics.get("write_state_payload_dtype"),
        "read_state_dtype": state_metrics.get("read_state_dtype"),
        "read_state_shape": state_metrics.get("read_state_shape"),
        "read_state_max_abs_diff": state_metrics.get("read_state_max_abs_diff"),
        "read_state_error": state_metrics.get("read_state_error"),
        "first_predict_ms": float(first_predict_ms),
        "steady_predict_ms": float(steady_predict_ms),
        "pytorch_mps_cached_avg_ms": float(pytorch_cached_avg_ms),
        "speedup_vs_pytorch_cached": speedup,
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel_diff,
        "raw_compute_plan_counts": raw_compute_plan_counts,
        "normalized_compute_plan_counts": normalized_compute_plan_counts,
        "read_state_op_count": read_state_op_count,
        "read_state_counts_by_state": read_state_counts_by_state,
        "large_read_state_ops": large_read_state_ops,
        "max_read_state_bytes_fp16": max_read_state_bytes_fp16,
        "slice_placement_summary": slice_placement_summary,
        "status": status,
        "status_reasons": status_reasons,
        "mode": MODE_COND_ONLY,
        "compute_precision": compute_precision_name,
        "require_ne_placement": require_ne_placement,
        "max_rel_diff": float(args.max_rel_diff),
        "seconds_derived_sequence_length": int(metadata["seconds_derived_sequence_length"]),
        "sequence_length_source": str(metadata["sequence_length_source"]),
        "text_valid_tokens": int(metadata["text_valid_tokens"]),
        "ref_len": int(metadata["ref_len"]),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
