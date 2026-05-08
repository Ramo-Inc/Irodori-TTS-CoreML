from __future__ import annotations

import contextlib
import io
import math
import platform
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .coreml_cache import (
    BRANCH_LAYOUT_ALTERNATING_SPEAKER2,
    BRANCH_LAYOUT_ALTERNATING_TEXT2,
    BRANCH_LAYOUT_COND1,
    BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    BRANCH_LAYOUT_JOINT2,
    STATE_LAYOUT_PER_LAYER,
    branch_count_for_layout,
    expected_per_layer_state_names,
)
from .model import TextToLatentRFDiT

__all__ = (
    "BRANCH_LAYOUT_ALTERNATING_SPEAKER2",
    "BRANCH_LAYOUT_ALTERNATING_TEXT2",
    "BRANCH_LAYOUT_COND1",
    "BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3",
    "BRANCH_LAYOUT_JOINT2",
    "STATE_LAYOUT_PER_LAYER",
    "CoreMLPreparedState",
    "CoreMLStatefulDenoiserBackend",
    "CoreMLStatefulUnavailableError",
    "StatefulContextPayload",
    "pack_context_kv_state",
    "state_write_payloads",
)


class CoreMLStatefulUnavailableError(RuntimeError):
    """Raised when the CoreML stateful denoiser fast path cannot be used."""


@dataclass(frozen=True)
class StatefulContextPayload:
    branch_layout: str
    state_layout: str
    state_payloads: Mapping[str, np.ndarray]
    text_len: int
    speaker_context_len: int
    speaker_context_bucket: int
    c_ctx_bucket: int
    batch_size: int


@dataclass
class CoreMLPreparedState:
    state: Any
    branch_layout: str
    lock: threading.Lock


def state_write_payloads(payload: StatefulContextPayload) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.astype(np.float32, copy=False))
        for name, value in payload.state_payloads.items()
    }


def _reset_freqs_cis_caches(module: nn.Module) -> None:
    with torch.no_grad():
        for child in module.modules():
            if not hasattr(child, "_freqs_cis_cache"):
                continue
            existing = child._freqs_cis_cache
            device = existing.device if isinstance(existing, torch.Tensor) else torch.device("cpu")
            child._freqs_cis_cache = torch.empty(
                (0,),
                device=device,
                dtype=torch.complex64,
            )


def _as_bool_cpu(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [B, T], got {tuple(mask.shape)}")
    return mask.detach().cpu().to(dtype=torch.bool)


def _validate_layer_context_tuple(
    layer_index: int,
    context_kv: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(context_kv) != 4:
        raise ValueError(
            "CoreML stateful packing expects 4 context tensors per layer "
            f"(k_text, v_text, k_speaker, v_speaker), got {len(context_kv)} "
            f"at layer {layer_index}"
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
    if k_text.shape[0] != k_speaker.shape[0]:
        raise ValueError(
            f"text/speaker branch batch mismatch at layer {layer_index}: {shapes[0]} vs {shapes[2]}"
        )
    if k_text.shape[2:] != k_speaker.shape[2:]:
        raise ValueError(
            f"text/speaker head shape mismatch at layer {layer_index}: {shapes[0]} vs {shapes[2]}"
        )
    return k_text, v_text, k_speaker, v_speaker


def _build_valid_mask_state(
    *,
    text_mask: torch.Tensor,
    speaker_mask: torch.Tensor,
    speaker_context_bucket: int,
    batch_size: int,
) -> np.ndarray:
    if speaker_context_bucket < 0:
        raise ValueError(f"speaker_context_bucket must be >= 0, got {speaker_context_bucket}")

    text_mask_cpu = _as_bool_cpu(text_mask, name="text_mask")
    speaker_mask_cpu = _as_bool_cpu(speaker_mask, name="speaker_mask")
    if text_mask_cpu.shape[0] != batch_size or speaker_mask_cpu.shape[0] != batch_size:
        raise ValueError(
            "mask branch batch does not match context batch: "
            f"text={tuple(text_mask_cpu.shape)} speaker={tuple(speaker_mask_cpu.shape)} "
            f"context_batch={batch_size}"
        )

    text_len = int(text_mask_cpu.shape[1])
    speaker_len = int(speaker_mask_cpu.shape[1])
    if speaker_len > speaker_context_bucket:
        raise ValueError(
            f"speaker_context_len {speaker_len} exceeds speaker_context_bucket "
            f"{speaker_context_bucket}"
        )

    c_ctx_bucket = text_len + int(speaker_context_bucket)
    valid = np.zeros((batch_size, c_ctx_bucket), dtype=np.float16)
    valid[:, :text_len] = text_mask_cpu.numpy().astype(np.float16)
    speaker_end = text_len + speaker_len
    valid[:, text_len:speaker_end] = speaker_mask_cpu.numpy().astype(np.float16)
    return valid


def pack_context_kv_state(
    context_kv_cache: list[tuple[torch.Tensor, ...]] | tuple[tuple[torch.Tensor, ...], ...],
    *,
    text_mask: torch.Tensor,
    speaker_mask: torch.Tensor,
    speaker_context_bucket: int,
    branch_layout: str = BRANCH_LAYOUT_COND1,
    state_layout: str = STATE_LAYOUT_PER_LAYER,
) -> StatefulContextPayload:
    if state_layout != STATE_LAYOUT_PER_LAYER:
        raise ValueError(f"unsupported state_layout for CoreML fast path: {state_layout}")
    if len(context_kv_cache) <= 0:
        raise ValueError("context_kv_cache must contain at least one layer")

    expected_batch = branch_count_for_layout(branch_layout)
    first = _validate_layer_context_tuple(0, tuple(context_kv_cache[0]))
    batch_size = int(first[0].shape[0])
    if batch_size != expected_batch:
        raise ValueError(
            f"branch batch {batch_size} does not match {branch_layout} "
            f"expected batch {expected_batch}"
        )

    text_len = int(first[0].shape[1])
    speaker_len = int(first[2].shape[1])
    num_heads = int(first[0].shape[2])
    head_dim = int(first[0].shape[3])
    if speaker_len > speaker_context_bucket:
        raise ValueError(
            f"speaker_context_len {speaker_len} exceeds speaker_context_bucket "
            f"{speaker_context_bucket}"
        )
    if int(text_mask.shape[1]) != text_len:
        raise ValueError(
            f"text_mask length {int(text_mask.shape[1])} does not match text_len {text_len}"
        )
    if int(speaker_mask.shape[1]) != speaker_len:
        raise ValueError(
            "speaker_mask length "
            f"{int(speaker_mask.shape[1])} does not match speaker_context_len {speaker_len}"
        )

    c_ctx_bucket = text_len + int(speaker_context_bucket)
    state_payloads: dict[str, np.ndarray] = {}
    for layer_index, layer_kv in enumerate(context_kv_cache):
        k_text, v_text, k_speaker, v_speaker = _validate_layer_context_tuple(
            layer_index,
            tuple(layer_kv),
        )
        expected_text_shape = (batch_size, text_len, num_heads, head_dim)
        expected_speaker_shape = (batch_size, speaker_len, num_heads, head_dim)
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

        k_state = np.zeros((batch_size, c_ctx_bucket, num_heads, head_dim), dtype=np.float16)
        v_state = np.zeros_like(k_state)
        k_state[:, :text_len, :, :] = k_text.detach().cpu().to(dtype=torch.float16).numpy()
        v_state[:, :text_len, :, :] = v_text.detach().cpu().to(dtype=torch.float16).numpy()
        speaker_end = text_len + speaker_len
        k_state[:, text_len:speaker_end, :, :] = (
            k_speaker.detach().cpu().to(dtype=torch.float16).numpy()
        )
        v_state[:, text_len:speaker_end, :, :] = (
            v_speaker.detach().cpu().to(dtype=torch.float16).numpy()
        )
        state_payloads[f"context_k_l{layer_index:02d}"] = k_state
        state_payloads[f"context_v_l{layer_index:02d}"] = v_state

    state_payloads["valid_mask_state"] = _build_valid_mask_state(
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        speaker_context_bucket=int(speaker_context_bucket),
        batch_size=batch_size,
    )
    return StatefulContextPayload(
        branch_layout=branch_layout,
        state_layout=state_layout,
        state_payloads=state_payloads,
        text_len=text_len,
        speaker_context_len=speaker_len,
        speaker_context_bucket=int(speaker_context_bucket),
        c_ctx_bucket=c_ctx_bucket,
        batch_size=batch_size,
    )


def _build_rope_cache(
    *,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    positions = torch.arange(sequence_length, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions * inv_freq[None, :]
    shape = (1, sequence_length, 1, head_dim // 2)
    return torch.cos(freqs).reshape(shape).to(dtype), torch.sin(freqs).reshape(shape).to(dtype)


def _apply_real_rope(
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    head_dim = x.shape[-1]
    x_pairs = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    y0 = x0 * rope_cos - x1 * rope_sin
    y1 = x0 * rope_sin + x1 * rope_cos
    return torch.stack((y0, y1), dim=-1).reshape_as(x)


def _apply_real_rope_half_heads(
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    x_rot, x_passthrough = x.chunk(2, dim=-2)
    x_rot = _apply_real_rope(x_rot, rope_cos, rope_sin)
    return torch.cat((x_rot, x_passthrough), dim=-2)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    y = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + eps)
    return y * weight


def _rms_norm_unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    return x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + eps)


def _explicit_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask_f: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    additive_mask = (1.0 - key_mask_f).reshape(key_mask_f.shape[0], 1, 1, key_mask_f.shape[1])
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    probs = torch.softmax(scores + additive_mask * -10000.0, dim=-1)
    return torch.matmul(probs, v)


class _CoreMLStatefulDenoiserStep(nn.Module):
    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        sequence_length: int,
        branch_count: int,
        c_ctx_bucket: int,
        state_layout: str,
    ) -> None:
        super().__init__()
        if state_layout != STATE_LAYOUT_PER_LAYER:
            raise CoreMLStatefulUnavailableError(f"unsupported CoreML state layout: {state_layout}")
        if model.cfg.use_caption_condition:
            raise CoreMLStatefulUnavailableError(
                "caption-conditioned checkpoints are not supported by the CoreML stateful fast path"
            )
        if not model.cfg.use_speaker_condition:
            raise CoreMLStatefulUnavailableError(
                "speaker-conditioned checkpoints are required by the CoreML stateful fast path"
            )

        self.cfg = model.cfg
        self.sequence_length = int(sequence_length)
        self.branch_count = int(branch_count)
        self.model_dim = int(model.cfg.model_dim)
        self.heads = int(model.cfg.num_heads)
        self.head_dim = self.model_dim // self.heads
        self.norm_eps = float(model.cfg.norm_eps)
        self.num_layers = int(model.cfg.num_layers)
        self.c_ctx_bucket = int(c_ctx_bucket)
        self.state_layout = state_layout

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

        rope_cos, rope_sin = _build_rope_cache(
            sequence_length=self.sequence_length,
            head_dim=self.head_dim,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        state_shape = (self.branch_count, self.c_ctx_bucket, self.heads, self.head_dim)
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
            torch.zeros((self.branch_count, self.c_ctx_bucket), dtype=torch.float16),
        )

    def state_buffer_names(self) -> tuple[str, ...]:
        return expected_per_layer_state_names(num_layers=self.num_layers)

    def iter_state_buffers(self):
        for name in self.state_buffer_names():
            yield name, getattr(self, name)

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
        y = _rms_norm_unit(x, adaln.eps)
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

        q = _rms_norm(q, attention.q_norm.weight, attention.q_norm.eps)
        k_self = _rms_norm(k_self, attention.k_norm.weight, attention.k_norm.eps)
        q = _apply_real_rope_half_heads(q, self.rope_cos, self.rope_sin)
        k_self = _apply_real_rope_half_heads(k_self, self.rope_cos, self.rope_sin)

        context_k = getattr(self, f"context_k_l{layer_index:02d}").to(dtype=k_self.dtype)
        context_v = getattr(self, f"context_v_l{layer_index:02d}").to(dtype=v_self.dtype)
        valid_mask = self.valid_mask_state.to(dtype=latent_mask_f.dtype)

        k = torch.cat((k_self, context_k), dim=1).transpose(1, 2)
        v = torch.cat((v_self, context_v), dim=1).transpose(1, 2)
        q = q.transpose(1, 2)
        key_mask_f = torch.cat((latent_mask_f, valid_mask), dim=1)
        y = _explicit_scaled_dot_product_attention(q, k, v, key_mask_f)
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

        x = _rms_norm(x, self.out_norm.weight, self.out_norm.eps)
        return self.out_proj(x)


class CoreMLStatefulDenoiserBackend:
    def __init__(
        self,
        model: TextToLatentRFDiT,
        *,
        sequence_length: int,
        c_ctx_bucket: int,
        branch_layout: str,
        state_layout: str = STATE_LAYOUT_PER_LAYER,
        io_dtype: str = "float32",
    ) -> None:
        self.model = model
        self.sequence_length = int(sequence_length)
        self.c_ctx_bucket = int(c_ctx_bucket)
        self.branch_layout = branch_layout
        self.branch_count = branch_count_for_layout(branch_layout)
        self.state_layout = state_layout
        self.io_dtype = io_dtype
        self._mlmodel: Any | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        self._metrics_lock = threading.Lock()
        self._prepare_state_count = 0
        self._predict_count = 0
        self._make_state_ms_total = 0.0
        self._write_state_ms_total = 0.0
        self._predict_ms_total = 0.0
        self._first_predict_ms: float | None = None

    def _metrics_lock_for_snapshot(self) -> threading.Lock:
        lock = getattr(self, "_metrics_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._metrics_lock = lock
        return lock

    def precompile(self) -> None:
        """Ensure the underlying MLModel is converted/loaded without running predict."""
        self._ensure_mlmodel()

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock_for_snapshot():
            prepare_count = self._prepare_state_count
            predict_count = self._predict_count
            first_predict_ms = self._first_predict_ms
            make_state_ms_total = self._make_state_ms_total
            write_state_ms_total = self._write_state_ms_total
            predict_ms_total = self._predict_ms_total
        steady_count = max(0, predict_count - 1) if first_predict_ms is not None else 0
        steady_total_ms = (
            (predict_ms_total - first_predict_ms)
            if first_predict_ms is not None and predict_count > 1
            else 0.0
        )
        avg_steady_predict_ms = steady_total_ms / steady_count if steady_count > 0 else None
        snapshot: dict[str, Any] = {
            "branch_layout": self.branch_layout,
            "state_layout": self.state_layout,
            "sequence_length": self.sequence_length,
            "c_ctx_bucket": self.c_ctx_bucket,
            "loaded": self._mlmodel is not None,
            "prepare_state_count": prepare_count,
            "predict_count": predict_count,
            "total_make_state_ms": make_state_ms_total,
            "total_write_state_ms": write_state_ms_total,
            "total_predict_ms": predict_ms_total,
            "avg_make_state_ms": (
                make_state_ms_total / prepare_count if prepare_count > 0 else 0.0
            ),
            "avg_write_state_ms": (
                write_state_ms_total / prepare_count if prepare_count > 0 else 0.0
            ),
            "avg_predict_ms": (predict_ms_total / predict_count if predict_count > 0 else 0.0),
            "first_predict_ms": first_predict_ms,
            "steady_predict_count": steady_count,
            "total_steady_predict_ms": steady_total_ms,
            "avg_steady_predict_ms": avg_steady_predict_ms,
            "ne_placement_available": False,
            "ne_placement_summary": None,
        }
        return snapshot

    def prepare_state(self, payload: StatefulContextPayload) -> CoreMLPreparedState:
        if payload.branch_layout != self.branch_layout:
            raise CoreMLStatefulUnavailableError(
                f"payload branch_layout={payload.branch_layout} does not match backend "
                f"branch_layout={self.branch_layout}"
            )
        if payload.state_layout != self.state_layout:
            raise CoreMLStatefulUnavailableError(
                f"payload state_layout={payload.state_layout} does not match backend "
                f"state_layout={self.state_layout}"
            )

        mlmodel = self._ensure_mlmodel()
        try:
            make_t0 = time.perf_counter()
            state = mlmodel.make_state()
            make_ms = (time.perf_counter() - make_t0) * 1000.0
            write_t0 = time.perf_counter()
            for name, value in state_write_payloads(payload).items():
                state.write_state(name=name, value=value)
            write_ms = (time.perf_counter() - write_t0) * 1000.0
        except Exception as exc:
            raise CoreMLStatefulUnavailableError(
                f"CoreML state initialization failed: {type(exc).__name__}: {_first_line(exc)}"
            ) from exc
        with self._metrics_lock_for_snapshot():
            self._prepare_state_count += 1
            self._make_state_ms_total += make_ms
            self._write_state_ms_total += write_ms
        return CoreMLPreparedState(
            state=state,
            branch_layout=self.branch_layout,
            lock=threading.Lock(),
        )

    def predict_step(
        self,
        prepared_state: CoreMLPreparedState,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prepared_state.branch_layout != self.branch_layout:
            raise CoreMLStatefulUnavailableError(
                f"prepared state branch_layout={prepared_state.branch_layout} does not match "
                f"backend branch_layout={self.branch_layout}"
            )
        mlmodel = self._ensure_mlmodel()
        predict_inputs = {
            "x_t": _torch_to_numpy_input(x_t, self.io_dtype),
            "t": _torch_to_numpy_input(t, self.io_dtype),
            "latent_mask_f": _torch_to_numpy_input(latent_mask.to(dtype=x_t.dtype), self.io_dtype),
        }
        try:
            with self._predict_lock, prepared_state.lock:
                predict_t0 = time.perf_counter()
                prediction = mlmodel.predict(predict_inputs, state=prepared_state.state)
                predict_ms = (time.perf_counter() - predict_t0) * 1000.0
        except Exception as exc:
            raise CoreMLStatefulUnavailableError(
                f"CoreML stateful predict failed: {type(exc).__name__}: {_first_line(exc)}"
            ) from exc
        if not isinstance(prediction, Mapping) or not prediction:
            raise CoreMLStatefulUnavailableError("CoreML stateful predict returned no outputs")
        with self._metrics_lock_for_snapshot():
            self._predict_count += 1
            self._predict_ms_total += predict_ms
            if self._first_predict_ms is None:
                self._first_predict_ms = predict_ms
        output = np.asarray(next(iter(prediction.values())))
        return torch.as_tensor(output, device=x_t.device, dtype=x_t.dtype)

    def _ensure_mlmodel(self) -> Any:
        with self._load_lock:
            if self._mlmodel is not None:
                return self._mlmodel
            self._mlmodel = self._convert_mlmodel()
            return self._mlmodel

    def _convert_mlmodel(self) -> Any:
        if platform.system() != "Darwin":
            raise CoreMLStatefulUnavailableError(
                "CoreML stateful denoiser requires macOS CoreML runtime support"
            )
        ct = _import_coremltools()
        compute_unit = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
        if compute_unit is None:
            raise CoreMLStatefulUnavailableError(
                "coremltools does not expose ComputeUnit.CPU_AND_NE"
            )
        if not hasattr(ct, "target") or not hasattr(ct.target, "iOS18"):
            raise CoreMLStatefulUnavailableError("coremltools iOS18 target support is required")

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        self._temporary_dir = tempfile.TemporaryDirectory(prefix="irodori-coreml-stateful-")
        package_path = Path(self._temporary_dir.name) / f"{self.branch_layout}.mlpackage"

        try:
            _reset_freqs_cis_caches(self.model)
            self.model.to(device="cpu", dtype=torch.float32)
            wrapper = _CoreMLStatefulDenoiserStep(
                self.model,
                sequence_length=self.sequence_length,
                branch_count=self.branch_count,
                c_ctx_bucket=self.c_ctx_bucket,
                state_layout=self.state_layout,
            ).eval()
            trace_inputs = _trace_inputs_for_wrapper(wrapper)
            traced = torch.jit.trace(wrapper, trace_inputs, check_trace=False)
            input_dtype = _numpy_dtype(self.io_dtype)
            input_types = [
                ct.TensorType(name="x_t", shape=trace_inputs[0].shape, dtype=input_dtype),
                ct.TensorType(name="t", shape=trace_inputs[1].shape, dtype=input_dtype),
                ct.TensorType(
                    name="latent_mask_f",
                    shape=trace_inputs[2].shape,
                    dtype=input_dtype,
                ),
            ]
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=tuple(buffer.shape), dtype=np.float16),
                    name=name,
                )
                for name, buffer in wrapper.iter_state_buffers()
            ]
            return _convert_and_reload_mlmodel(
                ct,
                traced,
                package_path=package_path,
                compute_unit=compute_unit,
                convert_kwargs={
                    "inputs": input_types,
                    "states": states,
                    "convert_to": "mlprogram",
                    "minimum_deployment_target": ct.target.iOS18,
                    "compute_units": compute_unit,
                },
            )
        except CoreMLStatefulUnavailableError:
            raise
        except Exception as exc:
            raise CoreMLStatefulUnavailableError(
                f"CoreML stateful conversion failed: {type(exc).__name__}: {_first_line(exc)}"
            ) from exc
        finally:
            self.model.to(device=model_device, dtype=model_dtype)
            _reset_freqs_cis_caches(self.model)


def _convert_and_reload_mlmodel(
    ct: Any,
    traced: Any,
    *,
    package_path: Path,
    compute_unit: Any,
    convert_kwargs: Mapping[str, Any],
) -> Any:
    convert_kwargs_with_package = dict(convert_kwargs)
    convert_kwargs_with_package["package_dir"] = str(package_path)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        ct.convert(traced, **convert_kwargs_with_package)
    return ct.models.MLModel(str(package_path), compute_units=compute_unit)


def _trace_inputs_for_wrapper(
    wrapper: _CoreMLStatefulDenoiserStep,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_t = torch.zeros(
        (
            wrapper.branch_count,
            wrapper.sequence_length,
            int(wrapper.cfg.patched_latent_dim),
        ),
        dtype=torch.float32,
    )
    t = torch.full((wrapper.branch_count,), 0.999, dtype=torch.float32)
    latent_mask = torch.ones(
        (wrapper.branch_count, wrapper.sequence_length),
        dtype=torch.float32,
    )
    return x_t, t, latent_mask


def _torch_to_numpy_input(value: torch.Tensor, io_dtype: str) -> np.ndarray:
    return np.ascontiguousarray(value.detach().cpu().numpy().astype(_numpy_dtype(io_dtype)))


def _numpy_dtype(io_dtype: str) -> Any:
    normalized = str(io_dtype).strip().lower()
    if normalized == "float32":
        return np.float32
    if normalized == "float16":
        return np.float16
    raise ValueError(f"unsupported CoreML io dtype: {io_dtype!r}")


def _import_coremltools() -> Any:
    try:
        import coremltools as ct
    except Exception as exc:
        raise CoreMLStatefulUnavailableError(
            "coremltools is required for the CoreML stateful denoiser fast path"
        ) from exc
    return ct


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else repr(exc)
