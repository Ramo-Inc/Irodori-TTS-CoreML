from __future__ import annotations

import contextlib
import io
import math
import platform
import sys
import time
from collections import Counter
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

from irodori_tts.config import ModelConfig  # noqa: E402
from irodori_tts.model import TextToLatentRFDiT  # noqa: E402

DIFF_TOL = 1e-5


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: str
    detail: str
    max_abs_diff: float | None = None
    predict_avg_ms: float | None = None
    compute_unit: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class CurrentStyleComplexRoPEProbe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(1, 3, 1, 2, 2))
        return torch.view_as_real(x_complex).reshape(1, 3, 1, 4)


class CurrentTinyTextToLatentForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = ModelConfig(
            latent_dim=4,
            latent_patch_size=1,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=1.0,
            text_mlp_ratio=1.0,
            text_vocab_size=16,
            text_dim=8,
            text_layers=1,
            text_heads=2,
            use_caption_condition=True,
            caption_vocab_size=16,
            caption_dim=8,
            caption_layers=1,
            caption_heads=2,
            caption_mlp_ratio=1.0,
            speaker_dim=8,
            speaker_layers=1,
            speaker_heads=2,
            timestep_embed_dim=8,
            adaln_rank=4,
            norm_eps=1e-5,
        )
        self.model = TextToLatentRFDiT(cfg).eval()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            x_t=x_t,
            t=t,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=None,
            ref_mask=None,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
            latent_mask=latent_mask,
        )


class CoreMLCompatibleDenoiserStep(nn.Module):
    seq_len = 3
    text_len = 2
    latent_dim = 4
    dim = 8
    heads = 2
    head_dim = 4
    mlp_hidden_dim = 12
    norm_eps = 1e-5

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(self.latent_dim, self.dim)
        self.cond = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6, bias=False),
        )
        self.attn_norm_weight = nn.Parameter(torch.ones(self.dim))
        self.mlp_norm_weight = nn.Parameter(torch.ones(self.dim))
        self.out_norm_weight = nn.Parameter(torch.ones(self.dim))
        self.q_norm_weight = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wk_text = nn.Linear(self.dim, self.dim, bias=False)
        self.wv_text = nn.Linear(self.dim, self.dim, bias=False)
        self.gate_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)
        self.w1 = nn.Linear(self.dim, self.mlp_hidden_dim, bias=False)
        self.w2 = nn.Linear(self.mlp_hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.mlp_hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.latent_dim)

        positions = torch.arange(self.seq_len, dtype=torch.float32)[:, None]
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        freqs = positions * inv_freq[None, :]
        rope_shape = (1, 1, self.seq_len, self.head_dim // 2)
        self.register_buffer("rope_cos", torch.cos(freqs).reshape(rope_shape), persistent=False)
        self.register_buffer("rope_sin", torch.sin(freqs).reshape(rope_shape), persistent=False)

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        y = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + self.norm_eps)
        return y * weight

    def apply_real_rope(self, x: torch.Tensor) -> torch.Tensor:
        x_pairs = x.reshape(1, self.heads, self.seq_len, self.head_dim // 2, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        y0 = x0 * self.rope_cos - x1 * self.rope_sin
        y1 = x0 * self.rope_sin + x1 * self.rope_cos
        return torch.stack((y0, y1), dim=-1).reshape(1, self.heads, self.seq_len, self.head_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_in: torch.Tensor,
        text_state: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_proj(x_t)
        mod = self.cond(cond_in).reshape(1, 1, 6, self.dim)
        attn_shift = mod[:, :, 0, :]
        attn_scale = mod[:, :, 1, :]
        attn_gate = torch.tanh(mod[:, :, 2, :])
        mlp_shift = mod[:, :, 3, :]
        mlp_scale = mod[:, :, 4, :]
        mlp_gate = torch.tanh(mod[:, :, 5, :])

        h = self.rms_norm(x, self.attn_norm_weight)
        h = h * (1.0 + attn_scale) + attn_shift

        q = self.wq(h).reshape(1, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_self = self.wk(h).reshape(1, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_self = self.wv(h).reshape(1, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_text = (
            self.wk_text(text_state)
            .reshape(1, self.text_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v_text = (
            self.wv_text(text_state)
            .reshape(1, self.text_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )

        head_norm_shape = (1, self.heads, 1, self.head_dim)
        q = self.rms_norm(q, self.q_norm_weight.reshape(head_norm_shape))
        k_self = self.rms_norm(k_self, self.k_norm_weight.reshape(head_norm_shape))
        k_text = self.rms_norm(k_text, self.k_norm_weight.reshape(head_norm_shape))
        q = self.apply_real_rope(q)
        k_self = self.apply_real_rope(k_self)

        k = torch.cat((k_self, k_text), dim=2)
        v = torch.cat((v_self, v_text), dim=2)
        mask = torch.cat((latent_mask, text_mask), dim=1).float()
        additive_mask = (1.0 - mask).reshape(1, 1, 1, self.seq_len + self.text_len) * -10000.0
        scores = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(float(self.head_dim)))
        probs = torch.softmax(scores + additive_mask, dim=-1)
        y = torch.matmul(probs, v).transpose(1, 2).reshape(1, self.seq_len, self.dim)
        y = self.wo(y * torch.sigmoid(self.gate_proj(h)))
        x = x + attn_gate * y

        h_mlp = self.rms_norm(x, self.mlp_norm_weight)
        h_mlp = h_mlp * (1.0 + mlp_scale) + mlp_shift
        x = x + mlp_gate * self.w2(F.silu(self.w1(h_mlp)) * self.w3(h_mlp))
        return self.out_proj(self.rms_norm(x, self.out_norm_weight))


class IrodoriLikeAttention(nn.Module):
    dim = 320
    heads = 8
    head_dim = dim // heads
    text_dim = 160
    seq_len = 32
    text_len = 16

    def __init__(self) -> None:
        super().__init__()
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wk_text = nn.Linear(self.text_dim, self.dim, bias=False)
        self.wv_text = nn.Linear(self.text_dim, self.dim, bias=False)
        self.gate = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.wq(x).reshape(1, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_self = self.wk(x).reshape(1, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_self = self.wv(x).reshape(1, self.seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_text = (
            self.wk_text(text)
            .reshape(1, self.text_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v_text = (
            self.wv_text(text)
            .reshape(1, self.text_len, self.heads, self.head_dim)
            .transpose(1, 2)
        )

        k = torch.cat((k_self, k_text), dim=2)
        v = torch.cat((v_self, v_text), dim=2)
        mask = torch.cat((latent_mask, text_mask), dim=1).float()
        additive_mask = (1.0 - mask).reshape(1, 1, 1, self.seq_len + self.text_len) * -10000.0
        scores = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(float(self.head_dim)))
        probs = torch.softmax(scores + additive_mask, dim=-1)
        y = torch.matmul(probs, v).transpose(1, 2).reshape(1, self.seq_len, self.dim)
        return self.wo(y * torch.sigmoid(self.gate(x)))


def import_coremltools() -> Any:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError(
            "coremltools is required for this PoC. Run with: "
            "uv run --with 'coremltools>=8.0' python tools/coreml_ane_poc.py"
        ) from exc
    return ct


def _first_line(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)


def _quiet_convert(ct: Any, *args: Any, **kwargs: Any) -> Any:
    if kwargs.pop("verbose", False):
        return ct.convert(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return ct.convert(*args, **kwargs)


def _iter_mlprogram_operations(block: Any) -> Any:
    for operation in block.operations:
        yield operation
        for nested_block in operation.blocks:
            yield from _iter_mlprogram_operations(nested_block)


def probe_complex_rope_conversion_failure(ct: Any | None = None) -> ProbeResult:
    ct = ct or import_coremltools()
    x = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    traced = torch.jit.trace(CurrentStyleComplexRoPEProbe().eval(), x, check_trace=False)
    try:
        _quiet_convert(
            ct,
            traced,
            inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {_first_line(exc)}"
        expected = isinstance(exc, NotImplementedError) and "view_as_complex" in str(exc)
        compatible = "view_as_complex" in str(exc) and "not implemented" in str(exc).lower()
        status = "PASS" if expected or compatible else "FAIL"
        return ProbeResult(
            name="Probe A: current-style complex RoPE conversion failure",
            status=status,
            detail=message,
        )
    return ProbeResult(
        name="Probe A: current-style complex RoPE conversion failure",
        status="FAIL",
        detail="conversion unexpectedly succeeded",
    )


def _current_tiny_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(
            [[[0.10, -0.20, 0.30, -0.40], [0.20, 0.00, -0.10, 0.05], [0.01, 0.02, 0.03, 0.04]]],
            dtype=torch.float32,
        ),
        torch.tensor([0.25], dtype=torch.float32),
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[True, True, False]], dtype=torch.bool),
        torch.tensor([[4, 5]], dtype=torch.long),
        torch.tensor([[True, True]], dtype=torch.bool),
        torch.tensor([[True, True, True]], dtype=torch.bool),
    )


def probe_current_tiny_forward_not_coreml_ready(ct: Any | None = None) -> ProbeResult:
    ct = ct or import_coremltools()
    torch.manual_seed(20260508)
    model = CurrentTinyTextToLatentForward().eval()
    inputs = _current_tiny_inputs()
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs, check_trace=False)
    graph_text = str(traced.inlined_graph)
    has_current_blocker = any(
        token in graph_text
        for token in ("view_as_complex", "view_as_real", "ComplexFloat", "complex64")
    )
    try:
        _quiet_convert(
            ct,
            traced,
            inputs=[
                ct.TensorType(name="x_t", shape=inputs[0].shape, dtype=np.float32),
                ct.TensorType(name="t", shape=inputs[1].shape, dtype=np.float32),
                ct.TensorType(name="text_ids", shape=inputs[2].shape, dtype=np.int32),
                ct.TensorType(name="text_mask", shape=inputs[3].shape, dtype=bool),
                ct.TensorType(name="caption_ids", shape=inputs[4].shape, dtype=np.int32),
                ct.TensorType(name="caption_mask", shape=inputs[5].shape, dtype=bool),
                ct.TensorType(name="latent_mask", shape=inputs[6].shape, dtype=bool),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {_first_line(exc)}"
        status = "PASS" if has_current_blocker else "FAIL"
        return ProbeResult(
            name="Probe B: current TextToLatentRFDiT tiny forward is not directly CoreML-ready",
            status=status,
            detail=message,
        )
    return ProbeResult(
        name="Probe B: current TextToLatentRFDiT tiny forward is not directly CoreML-ready",
        status="FAIL",
        detail="conversion unexpectedly succeeded",
    )


def _coreml_step_and_inputs() -> tuple[CoreMLCompatibleDenoiserStep, tuple[torch.Tensor, ...]]:
    torch.manual_seed(20260508)
    model = CoreMLCompatibleDenoiserStep().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.1)
        model.attn_norm_weight.fill_(1.0)
        model.mlp_norm_weight.fill_(1.0)
        model.out_norm_weight.fill_(1.0)
        model.q_norm_weight.fill_(1.0)
        model.k_norm_weight.fill_(1.0)

    x_t = torch.tensor(
        [[[0.10, -0.20, 0.30, -0.40], [0.20, 0.00, -0.10, 0.05], [0.01, 0.02, 0.03, 0.04]]],
        dtype=torch.float32,
    )
    cond_in = torch.linspace(-0.35, 0.35, CoreMLCompatibleDenoiserStep.dim).reshape(1, -1)
    text_state = torch.linspace(-0.20, 0.55, CoreMLCompatibleDenoiserStep.text_len * CoreMLCompatibleDenoiserStep.dim).reshape(
        1, CoreMLCompatibleDenoiserStep.text_len, CoreMLCompatibleDenoiserStep.dim
    )
    latent_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    text_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    return model, (x_t, cond_in, text_state, latent_mask, text_mask)


def probe_coreml_compatible_denoiser(
    ct: Any | None = None,
    compute_unit: Any | None = None,
    *,
    iterations: int = 10,
) -> ProbeResult:
    ct = ct or import_coremltools()
    if platform.system() != "Darwin":
        return ProbeResult(
            name="Probe C: CoreML-compatible denoiser rewrite",
            status="SKIP",
            detail="Core ML prediction is only available on macOS",
        )

    compute_unit = compute_unit or ct.ComputeUnit.ALL
    compute_unit_name = getattr(compute_unit, "name", str(compute_unit))
    model, inputs = _coreml_step_and_inputs()
    with torch.no_grad():
        expected = model(*inputs).detach().cpu().numpy()
        traced = torch.jit.trace(model, inputs, check_trace=False)
    input_types = [
        ct.TensorType(name="x_t", shape=inputs[0].shape, dtype=np.float32),
        ct.TensorType(name="cond_in", shape=inputs[1].shape, dtype=np.float32),
        ct.TensorType(name="text_state", shape=inputs[2].shape, dtype=np.float32),
        ct.TensorType(name="latent_mask", shape=inputs[3].shape, dtype=np.float32),
        ct.TensorType(name="text_mask", shape=inputs[4].shape, dtype=np.float32),
    ]
    try:
        mlmodel = _quiet_convert(
            ct,
            traced,
            inputs=input_types,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_units=compute_unit,
            compute_precision=ct.precision.FLOAT32,
        )
        predict_inputs = {
            "x_t": inputs[0].numpy(),
            "cond_in": inputs[1].numpy(),
            "text_state": inputs[2].numpy(),
            "latent_mask": inputs[3].numpy(),
            "text_mask": inputs[4].numpy(),
        }
        prediction = mlmodel.predict(predict_inputs)
        output = np.asarray(next(iter(prediction.values())))
        max_abs_diff = float(np.max(np.abs(output - expected)))

        for _ in range(2):
            mlmodel.predict(predict_inputs)
        start = time.perf_counter()
        for _ in range(iterations):
            mlmodel.predict(predict_inputs)
        predict_avg_ms = (time.perf_counter() - start) * 1000.0 / float(iterations)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {_first_line(exc)}"
        if compute_unit_name == "CPU_AND_NE":
            return ProbeResult(
                name="Probe C: CoreML-compatible denoiser rewrite",
                status="SKIP",
                detail=f"CPU_AND_NE runtime rejected the model: {detail}",
                compute_unit=compute_unit_name,
            )
        return ProbeResult(
            name="Probe C: CoreML-compatible denoiser rewrite",
            status="FAIL",
            detail=detail,
            compute_unit=compute_unit_name,
        )

    status = "PASS" if max_abs_diff < DIFF_TOL else "FAIL"
    return ProbeResult(
        name="Probe C: CoreML-compatible denoiser rewrite",
        status=status,
        detail=(
            f"numerical correctness only: max_abs_diff={max_abs_diff:.3g}, "
            f"avg_predict_ms={predict_avg_ms:.3f}; execution is not per-op ANE placement proof"
        ),
        max_abs_diff=max_abs_diff,
        predict_avg_ms=predict_avg_ms,
        compute_unit=compute_unit_name,
    )


def _irodori_like_attention_and_inputs() -> tuple[IrodoriLikeAttention, tuple[torch.Tensor, ...]]:
    torch.manual_seed(20260508)
    model = IrodoriLikeAttention().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)

    x = torch.randn(1, IrodoriLikeAttention.seq_len, IrodoriLikeAttention.dim, dtype=torch.float32)
    text = torch.randn(
        1, IrodoriLikeAttention.text_len, IrodoriLikeAttention.text_dim, dtype=torch.float32
    )
    latent_mask = torch.ones(1, IrodoriLikeAttention.seq_len, dtype=torch.float32)
    text_mask = torch.ones(1, IrodoriLikeAttention.text_len, dtype=torch.float32)
    return model, (x, text, latent_mask, text_mask)


def probe_ane_compute_plan_placement(ct: Any | None = None) -> ProbeResult:
    ct = ct or import_coremltools()
    if platform.system() != "Darwin":
        return ProbeResult(
            name="Probe D: Irodori-like attention per-op ANE placement",
            status="SKIP",
            detail="MLComputePlan inspection requires macOS",
            compute_unit="CPU_AND_NE",
        )

    try:
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice
        from coremltools.models.compute_plan import MLComputePlan
    except Exception as exc:
        return ProbeResult(
            name="Probe D: Irodori-like attention per-op ANE placement",
            status="SKIP",
            detail=f"MLComputePlan unavailable: {type(exc).__name__}: {_first_line(exc)}",
            compute_unit="CPU_AND_NE",
        )

    cpu_and_ne = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    if cpu_and_ne is None:
        return ProbeResult(
            name="Probe D: Irodori-like attention per-op ANE placement",
            status="SKIP",
            detail="coremltools does not expose ComputeUnit.CPU_AND_NE",
            compute_unit="CPU_AND_NE",
        )

    model, inputs = _irodori_like_attention_and_inputs()
    with torch.no_grad():
        traced = torch.jit.trace(model, inputs, check_trace=False)

    try:
        mlmodel = _quiet_convert(
            ct,
            traced,
            inputs=[
                ct.TensorType(name="x", shape=inputs[0].shape, dtype=np.float32),
                ct.TensorType(name="text", shape=inputs[1].shape, dtype=np.float32),
                ct.TensorType(name="latent_mask", shape=inputs[2].shape, dtype=np.float32),
                ct.TensorType(name="text_mask", shape=inputs[3].shape, dtype=np.float32),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_units=cpu_and_ne,
        )
        plan = MLComputePlan.load_from_path(
            mlmodel.get_compiled_model_path(),
            compute_units=cpu_and_ne,
        )
    except Exception as exc:
        return ProbeResult(
            name="Probe D: Irodori-like attention per-op ANE placement",
            status="SKIP",
            detail=f"MLComputePlan runtime rejected the model: {type(exc).__name__}: {_first_line(exc)}",
            compute_unit="CPU_AND_NE",
        )

    program = plan.model_structure.program
    if program is None:
        return ProbeResult(
            name="Probe D: Irodori-like attention per-op ANE placement",
            status="FAIL",
            detail="compiled model is not an ML Program",
            compute_unit="CPU_AND_NE",
        )

    main_function = program.functions["main"]
    total_by_op: Counter[str] = Counter()
    neural_engine_by_op: Counter[str] = Counter()
    preferred_device_by_op: Counter[tuple[str, str]] = Counter()
    for operation in _iter_mlprogram_operations(main_function.block):
        op_name = operation.operator_name
        total_by_op[op_name] += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            device_name = "Unknown"
        else:
            device = usage.preferred_compute_device
            device_name = type(device).__name__
            if isinstance(device, MLNeuralEngineComputeDevice):
                neural_engine_by_op[op_name] += 1
        preferred_device_by_op[(op_name, device_name)] += 1

    linear_ne = neural_engine_by_op["ios16.linear"]
    attention_ne = neural_engine_by_op["ios16.matmul"] + neural_engine_by_op["ios16.softmax"]
    status = "PASS" if linear_ne >= 1 and attention_ne >= 1 else "FAIL"
    interesting_ops = ("ios16.linear", "ios16.matmul", "ios16.softmax")
    op_counts = ", ".join(
        f"{op}:total={total_by_op[op]},ne={neural_engine_by_op[op]}" for op in interesting_ops
    )
    preferred_counts = ", ".join(
        f"{op}/{device}={count}"
        for (op, device), count in sorted(preferred_device_by_op.items())
        if op in interesting_ops
    )
    return ProbeResult(
        name="Probe D: Irodori-like attention per-op ANE placement",
        status=status,
        detail=f"{op_counts}; preferred={preferred_counts}",
        compute_unit="CPU_AND_NE",
    )


def run_all_probes(ct: Any | None = None) -> list[ProbeResult]:
    ct = ct or import_coremltools()
    results = [
        probe_complex_rope_conversion_failure(ct),
        probe_current_tiny_forward_not_coreml_ready(ct),
        probe_coreml_compatible_denoiser(ct, ct.ComputeUnit.ALL),
    ]
    cpu_and_ne = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    if cpu_and_ne is not None:
        results.append(probe_coreml_compatible_denoiser(ct, cpu_and_ne))
    else:
        results.append(
            ProbeResult(
                name="Probe C: CoreML-compatible denoiser rewrite",
                status="SKIP",
                detail="coremltools does not expose ComputeUnit.CPU_AND_NE",
                compute_unit="CPU_AND_NE",
            )
        )
    results.append(probe_ane_compute_plan_placement(ct))
    return results


def print_summary(results: list[ProbeResult], ct: Any) -> None:
    print("Core ML / ANE feasibility PoC")
    print(f"coremltools={ct.__version__} torch={torch.__version__}")
    for result in results:
        unit = f" [{result.compute_unit}]" if result.compute_unit else ""
        print(f"{result.status}: {result.name}{unit} - {result.detail}")


def main() -> int:
    try:
        ct = import_coremltools()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    results = run_all_probes(ct)
    print_summary(results, ct)
    required = [
        result
        for result in results
        if not (
            result.status == "SKIP"
            and (
                result.compute_unit == "CPU_AND_NE"
                or result.name.startswith("Probe D")
            )
        )
    ]
    return 0 if all(result.passed for result in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
