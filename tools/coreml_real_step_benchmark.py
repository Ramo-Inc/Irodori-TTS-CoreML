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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.codec import DACVAECodec, patchify_latent  # noqa: E402
from irodori_tts.config import ModelConfig  # noqa: E402
from irodori_tts.inference_runtime import _load_checkpoint_for_inference  # noqa: E402
from irodori_tts.model import TextToLatentRFDiT  # noqa: E402
from irodori_tts.text_normalization import normalize_text  # noqa: E402
from irodori_tts.tokenizer import PretrainedTextTokenizer  # noqa: E402

DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_TEXT = "こんにちは。今日はいい天気ですね。"
DEFAULT_REF_WAV = REPO_ROOT / "rem.wav"
MODE_NO_CONTEXT_KV_CACHE = "no_context_kv_cache"
REQUIRED_NE_OPS = ("ios16.linear", "ios16.matmul", "ios16.softmax")


def import_coremltools() -> Any:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError(
            "coremltools is required for this benchmark. Run with: "
            "uv run --with 'coremltools>=8.0' python tools/coreml_real_step_benchmark.py"
        ) from exc
    return ct


def first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else repr(exc)


def resolve_compute_precision(ct: Any, precision: str) -> tuple[Any | None, str]:
    mode = str(precision).strip().lower()
    if mode == "default":
        return None, "default"
    if mode == "float16":
        return ct.precision.FLOAT16, "float16"
    if mode == "float32":
        return ct.precision.FLOAT32, "float32"
    raise ValueError(f"Unsupported compute precision: {precision!r}")


def resolve_checkpoint_path(checkpoint: str) -> Path:
    raw = str(checkpoint).strip()
    if raw == "":
        raise ValueError("checkpoint must be non-empty.")

    candidate = Path(raw).expanduser()
    local_candidates = [candidate]
    if not candidate.is_absolute():
        local_candidates.append(REPO_ROOT / candidate)
    for local_path in local_candidates:
        if local_path.is_file():
            return local_path

    if candidate.suffix.lower() in {".pt", ".safetensors"}:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    path = hf_hub_download(repo_id=raw, filename="model.safetensors")
    print(f"[checkpoint] hf://{raw} -> {path}", flush=True)
    return Path(path)


def default_text_max_len(train_cfg: dict | None) -> int:
    value = train_cfg.get("max_text_len") if isinstance(train_cfg, dict) else None
    if isinstance(value, int) and value > 0:
        return int(value)
    return 256


def load_actual_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TextToLatentRFDiT, ModelConfig, dict | None]:
    model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(checkpoint_path)
    model_cfg = ModelConfig(**model_cfg_dict)
    model = TextToLatentRFDiT(model_cfg)
    model.load_state_dict(model_state)
    del model_state
    model = model.to(device=device, dtype=torch.float32).eval()
    return model, model_cfg, train_cfg


def build_rope_cache(
    *,
    sequence_length: int,
    head_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = positions * inv_freq[None, :]
    shape = (1, sequence_length, 1, head_dim // 2)
    return torch.cos(freqs).reshape(shape).to(dtype), torch.sin(freqs).reshape(shape).to(dtype)


def apply_real_rope(
    x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
) -> torch.Tensor:
    head_dim = x.shape[-1]
    x_pairs = x.reshape(*x.shape[:-1], head_dim // 2, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    y0 = x0 * rope_cos - x1 * rope_sin
    y1 = x0 * rope_sin + x1 * rope_cos
    return torch.stack((y0, y1), dim=-1).reshape_as(x)


def apply_real_rope_half_heads(
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    x_rot, x_passthrough = x.chunk(2, dim=-2)
    x_rot = apply_real_rope(x_rot, rope_cos, rope_sin)
    return torch.cat((x_rot, x_passthrough), dim=-2)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    y = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + eps)
    return y * weight


def rms_norm_unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    return x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + eps)


def explicit_scaled_dot_product_attention(
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


class RealCoreMLDenoiserStep(nn.Module):
    """
    Core ML conversion wrapper for one real TextToLatentRFDiT denoiser step.

    The production model is not modified. This wrapper reuses the real denoiser
    modules and replaces only the conversion blockers in the denoiser path:
    complex RoPE, scaled_dot_product_attention, and bool attention masks.
    """

    def __init__(self, model: TextToLatentRFDiT, sequence_length: int):
        super().__init__()
        if model.cfg.use_caption_condition:
            raise NotImplementedError(
                "This benchmark targets the speaker-conditioned DEFAULT_CHECKPOINT. "
                "Caption-conditioned checkpoints need a caption-state wrapper variant."
            )
        if not model.cfg.use_speaker_condition:
            raise ValueError("DEFAULT_CHECKPOINT is expected to use speaker conditioning.")

        self.cfg = model.cfg
        self.sequence_length = int(sequence_length)
        self.model_dim = int(model.cfg.model_dim)
        self.heads = int(model.cfg.num_heads)
        self.head_dim = self.model_dim // self.heads
        self.norm_eps = float(model.cfg.norm_eps)

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

        rope_cos, rope_sin = build_rope_cache(
            sequence_length=self.sequence_length,
            head_dim=self.head_dim,
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

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
        y = rms_norm_unit(x, adaln.eps)
        y = y * (1.0 + scale) + shift
        return y, torch.tanh(gate)

    def project_kv(
        self,
        attention: nn.Module,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = text_state.shape[0]
        text_len = text_state.shape[1]
        speaker_len = speaker_state.shape[1]
        k_text = attention.wk_text(text_state).reshape(bsz, text_len, self.heads, self.head_dim)
        v_text = attention.wv_text(text_state).reshape(bsz, text_len, self.heads, self.head_dim)
        k_speaker = attention.wk_speaker(speaker_state).reshape(
            bsz, speaker_len, self.heads, self.head_dim
        )
        v_speaker = attention.wv_speaker(speaker_state).reshape(
            bsz, speaker_len, self.heads, self.head_dim
        )
        k_text = rms_norm(k_text, attention.k_norm.weight, attention.k_norm.eps)
        k_speaker = rms_norm(k_speaker, attention.k_norm.weight, attention.k_norm.eps)
        return k_text, v_text, k_speaker, v_speaker

    def joint_attention(
        self,
        attention: nn.Module,
        x: torch.Tensor,
        text_state: torch.Tensor,
        text_mask_f: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask_f: torch.Tensor,
        latent_mask_f: torch.Tensor,
    ) -> torch.Tensor:
        bsz = x.shape[0]
        seq_len = x.shape[1]
        q = attention.wq(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_self = attention.wk(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        v_self = attention.wv(x).reshape(bsz, seq_len, self.heads, self.head_dim)
        k_text, v_text, k_speaker, v_speaker = self.project_kv(
            attention=attention,
            text_state=text_state,
            speaker_state=speaker_state,
        )

        q = rms_norm(q, attention.q_norm.weight, attention.q_norm.eps)
        k_self = rms_norm(k_self, attention.k_norm.weight, attention.k_norm.eps)
        q = apply_real_rope_half_heads(q, self.rope_cos, self.rope_sin)
        k_self = apply_real_rope_half_heads(k_self, self.rope_cos, self.rope_sin)

        k = torch.cat((k_self, k_text, k_speaker), dim=1).transpose(1, 2)
        v = torch.cat((v_self, v_text, v_speaker), dim=1).transpose(1, 2)
        q = q.transpose(1, 2)
        key_mask_f = torch.cat((latent_mask_f, text_mask_f, speaker_mask_f), dim=1)
        y = explicit_scaled_dot_product_attention(q, k, v, key_mask_f)
        y = y.transpose(1, 2).reshape(bsz, seq_len, self.model_dim)
        y = y * torch.sigmoid(attention.gate(x))
        return attention.wo(y)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask_f: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask_f: torch.Tensor,
        latent_mask_f: torch.Tensor,
    ) -> torch.Tensor:
        t_embed = self.timestep_embedding(t).to(dtype=x_t.dtype)
        cond_embed = self.cond_module(t_embed)[:, None, :]

        x = self.in_proj(x_t)
        for block in self.blocks:
            h, attention_gate = self.low_rank_adaln(block.attention_adaln, x, cond_embed)
            y = self.joint_attention(
                attention=block.attention,
                x=h,
                text_state=text_state,
                text_mask_f=text_mask_f,
                speaker_state=speaker_state,
                speaker_mask_f=speaker_mask_f,
                latent_mask_f=latent_mask_f,
            )
            x = x + attention_gate * y

            h_mlp, mlp_gate = self.low_rank_adaln(block.mlp_adaln, x, cond_embed)
            x = x + mlp_gate * block.mlp(h_mlp)

        x = rms_norm(x, self.out_norm.weight, self.out_norm.eps)
        return self.out_proj(x)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def benchmark_torch_step(
    model: TextToLatentRFDiT,
    inputs: tuple[torch.Tensor, ...],
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
            )
        sync_device(device)
    if output is None:
        raise RuntimeError("No PyTorch benchmark iterations ran.")
    avg_ms = (time.perf_counter() - start) * 1000.0 / float(iterations)
    return output.detach().cpu().numpy(), avg_ms


def benchmark_coreml_predict(
    mlmodel: Any,
    predict_inputs: dict[str, np.ndarray],
    *,
    warmup: int,
    iterations: int,
) -> tuple[np.ndarray, float]:
    prediction = None
    for _ in range(warmup):
        prediction = mlmodel.predict(predict_inputs)
    start = time.perf_counter()
    for _ in range(iterations):
        prediction = mlmodel.predict(predict_inputs)
    avg_ms = (time.perf_counter() - start) * 1000.0 / float(iterations)
    if prediction is None:
        raise RuntimeError("No Core ML prediction iterations ran.")
    return np.asarray(next(iter(prediction.values()))), avg_ms


def quiet_convert(ct: Any, *args: Any, verbose: bool = False, **kwargs: Any) -> Any:
    if verbose:
        return ct.convert(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return ct.convert(*args, **kwargs)


def iter_mlprogram_operations(block: Any) -> Iterator[Any]:
    for operation in block.operations:
        yield operation
        for nested_block in operation.blocks:
            yield from iter_mlprogram_operations(nested_block)


def compute_plan_counts(ct: Any, mlmodel: Any, compute_unit: Any) -> dict[str, Any]:
    try:
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice
        from coremltools.models.compute_plan import MLComputePlan
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {first_line(exc)}"}

    try:
        plan = MLComputePlan.load_from_path(
            mlmodel.get_compiled_model_path(),
            compute_units=compute_unit,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {first_line(exc)}"}

    program = plan.model_structure.program
    if program is None:
        return {"error": "compiled model is not an ML Program"}

    total_by_op: Counter[str] = Counter()
    ne_by_op: Counter[str] = Counter()
    preferred_by_interesting_op: dict[str, Counter[str]] = {
        "ios16.linear": Counter(),
        "ios16.matmul": Counter(),
        "ios16.softmax": Counter(),
    }
    total_ne_preferred_ops = 0

    main_function = program.functions["main"]
    for operation in iter_mlprogram_operations(main_function.block):
        op_name = str(operation.operator_name)
        total_by_op[op_name] += 1
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        device = None if usage is None else usage.preferred_compute_device
        device_name = "Unknown" if device is None else type(device).__name__
        if op_name in preferred_by_interesting_op:
            preferred_by_interesting_op[op_name][device_name] += 1
        if isinstance(device, MLNeuralEngineComputeDevice):
            ne_by_op[op_name] += 1
            total_ne_preferred_ops += 1

    counts: dict[str, Any] = {
        "total_ne_preferred_ops": int(total_ne_preferred_ops),
    }
    for op_name in ("ios16.linear", "ios16.matmul", "ios16.softmax"):
        counts[op_name] = {
            "total": int(total_by_op[op_name]),
            "ne_preferred": int(ne_by_op[op_name]),
            "preferred_devices": dict(preferred_by_interesting_op[op_name]),
        }
    return counts


def evaluate_benchmark_status(
    *,
    ne_preferred_counts: dict[str, Any],
    rel_diff: float,
    max_rel_diff: float,
    require_ne_placement: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if rel_diff > max_rel_diff:
        reasons.append(f"rel_diff {rel_diff:.6g} exceeds max_rel_diff {max_rel_diff:.6g}")

    if require_ne_placement:
        if "error" in ne_preferred_counts:
            reasons.append(f"NE placement check failed: {ne_preferred_counts['error']}")
        else:
            for op_name in REQUIRED_NE_OPS:
                op_counts = ne_preferred_counts.get(op_name)
                ne_preferred = 0
                if isinstance(op_counts, dict):
                    raw_value = op_counts.get("ne_preferred", 0)
                    if isinstance(raw_value, int | float):
                        ne_preferred = int(raw_value)
                if ne_preferred <= 0:
                    reasons.append(f"{op_name} has no NE-preferred operations")

    return ("FAIL" if reasons else "PASS"), reasons


def prepare_real_inputs(
    *,
    model: TextToLatentRFDiT,
    model_cfg: ModelConfig,
    train_cfg: dict | None,
    checkpoint: str,
    codec_repo: str,
    ref_wav: Path,
    text: str,
    seconds: float,
    sequence_length_override: int | None,
    seed: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    del checkpoint
    if not ref_wav.is_file():
        raise FileNotFoundError(f"Reference wav not found: {ref_wav}")

    tokenizer = PretrainedTextTokenizer.from_pretrained(
        repo_id=model_cfg.text_tokenizer_repo,
        add_bos=bool(model_cfg.text_add_bos),
        local_files_only=False,
    )
    if tokenizer.vocab_size != model_cfg.text_vocab_size:
        raise ValueError(
            f"text_vocab_size mismatch: checkpoint={model_cfg.text_vocab_size} "
            f"tokenizer({model_cfg.text_tokenizer_repo})={tokenizer.vocab_size}"
        )

    codec = DACVAECodec.load(
        repo_id=codec_repo,
        device="cpu",
        dtype=torch.float32,
        deterministic_encode=True,
        deterministic_decode=True,
        enable_watermark=False,
    )
    if codec.latent_dim != model_cfg.latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: checkpoint={model_cfg.latent_dim} codec={codec.latent_dim}"
        )

    normalized_text = normalize_text(text).strip()
    if normalized_text == "":
        raise ValueError("text became empty after normalization.")

    text_ids, text_mask = tokenizer.batch_encode(
        [normalized_text],
        max_length=default_text_max_len(train_cfg),
    )
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    ref_latent = codec.encode_file(ref_wav)
    ref_latent_patched = patchify_latent(ref_latent, model_cfg.latent_patch_size)
    if ref_latent_patched.shape[1] <= 0:
        raise ValueError("Reference latent length became zero after patchify.")
    ref_latent_patched = ref_latent_patched.to(device=device, dtype=torch.float32)
    ref_mask = torch.ones(
        (1, ref_latent_patched.shape[1]),
        dtype=torch.bool,
        device=device,
    )

    target_samples = int(float(seconds) * float(codec.sample_rate))
    latent_steps = math.ceil(target_samples / int(codec.model.hop_length))
    seconds_derived_sequence_length = math.ceil(latent_steps / int(model_cfg.latent_patch_size))
    if sequence_length_override is None:
        sequence_length = int(seconds_derived_sequence_length)
        sequence_length_source = "seconds"
    else:
        sequence_length = int(sequence_length_override)
        sequence_length_source = "override"
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")

    with torch.inference_mode():
        (
            text_state,
            text_mask_encoded,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        ) = model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent_patched,
            ref_mask=ref_mask,
        )
    if caption_state is not None or caption_mask is not None:
        raise NotImplementedError(
            "Caption-conditioned checkpoints are not supported by this benchmark."
        )
    if speaker_state is None or speaker_mask is None:
        raise RuntimeError("Speaker-conditioned DEFAULT_CHECKPOINT did not produce speaker state.")

    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    x_t = torch.randn(
        (1, sequence_length, model_cfg.patched_latent_dim),
        dtype=torch.float32,
        generator=rng,
    ).to(device)
    t = torch.tensor([0.999], dtype=torch.float32, device=device)
    latent_mask = torch.ones((1, sequence_length), dtype=torch.bool, device=device)

    metadata = {
        "sequence_length": int(sequence_length),
        "sequence_length_source": sequence_length_source,
        "seconds_derived_sequence_length": int(seconds_derived_sequence_length),
        "seconds_derived_latent_steps": int(latent_steps),
        "sample_rate": int(codec.sample_rate),
        "hop_length": int(codec.model.hop_length),
        "text_len": int(text_state.shape[1]),
        "text_valid_tokens": int(text_mask_encoded.sum().item()),
        "ref_len": int(ref_latent_patched.shape[1]),
        "speaker_context_len": int(speaker_state.shape[1]),
        "normalized_text": normalized_text,
    }
    return (
        x_t,
        t,
        text_state,
        text_mask_encoded,
        speaker_state,
        speaker_mask,
        latent_mask,
    ), metadata


def cpu_coreml_inputs(
    inputs: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], dict[str, np.ndarray]]:
    x_t, t, text_state, text_mask, speaker_state, speaker_mask, latent_mask = inputs
    cpu_inputs = (
        x_t.detach().cpu(),
        t.detach().cpu(),
        text_state.detach().cpu(),
        text_mask.detach().cpu().to(dtype=torch.float32),
        speaker_state.detach().cpu(),
        speaker_mask.detach().cpu().to(dtype=torch.float32),
        latent_mask.detach().cpu().to(dtype=torch.float32),
    )
    predict_inputs = {
        "x_t": cpu_inputs[0].numpy(),
        "t": cpu_inputs[1].numpy(),
        "text_state": cpu_inputs[2].numpy(),
        "text_mask_f": cpu_inputs[3].numpy(),
        "speaker_state": cpu_inputs[4].numpy(),
        "speaker_mask_f": cpu_inputs[5].numpy(),
        "latent_mask_f": cpu_inputs[6].numpy(),
    }
    return cpu_inputs, predict_inputs


def convert_wrapper(
    ct: Any,
    wrapper: RealCoreMLDenoiserStep,
    cpu_inputs: tuple[torch.Tensor, ...],
    *,
    artifact_dir: Path,
    compute_precision: Any | None,
    verbose: bool,
) -> tuple[Any, float]:
    input_types = [
        ct.TensorType(name="x_t", shape=cpu_inputs[0].shape, dtype=np.float32),
        ct.TensorType(name="t", shape=cpu_inputs[1].shape, dtype=np.float32),
        ct.TensorType(name="text_state", shape=cpu_inputs[2].shape, dtype=np.float32),
        ct.TensorType(name="text_mask_f", shape=cpu_inputs[3].shape, dtype=np.float32),
        ct.TensorType(name="speaker_state", shape=cpu_inputs[4].shape, dtype=np.float32),
        ct.TensorType(name="speaker_mask_f", shape=cpu_inputs[5].shape, dtype=np.float32),
        ct.TensorType(name="latent_mask_f", shape=cpu_inputs[6].shape, dtype=np.float32),
    ]
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper.eval(), cpu_inputs, check_trace=False)

    compute_unit = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    if compute_unit is None:
        raise RuntimeError("coremltools does not expose ComputeUnit.CPU_AND_NE.")

    start = time.perf_counter()
    package_path = artifact_dir / "real_step.mlpackage"
    print(f"[artifacts] mlpackage: {package_path}", flush=True)
    convert_kwargs = {
        "inputs": input_types,
        "convert_to": "mlprogram",
        "minimum_deployment_target": ct.target.macOS13,
        "compute_units": compute_unit,
        "package_dir": str(package_path),
    }
    if compute_precision is not None:
        convert_kwargs["compute_precision"] = compute_precision
    try:
        mlmodel = quiet_convert(
            ct,
            traced,
            **convert_kwargs,
            verbose=verbose,
        )
    except Exception as exc:
        shapes = {
            "x_t": tuple(cpu_inputs[0].shape),
            "text_state": tuple(cpu_inputs[2].shape),
            "speaker_state": tuple(cpu_inputs[4].shape),
        }
        raise RuntimeError(
            "Core ML conversion failed for real denoiser-step shapes "
            f"{shapes}. This is not a toy-shape fallback; retry with "
            "--sequence-length 64 only as an explicit debug override. "
            f"Original error: {type(exc).__name__}: {first_line(exc)}"
        ) from exc
    return mlmodel, time.perf_counter() - start


@contextlib.contextmanager
def temporary_artifact_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="coreml-real-step-benchmark-") as tmpdir:
        yield Path(tmpdir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one real Irodori-TTS denoiser step on PyTorch MPS and "
            "Core ML CPU_AND_NE. Run with uv --with coremltools; generated artifacts "
            "stay in a temporary directory."
        )
    )
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT, help="Text to tokenize.")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--compute-precision",
        choices=("default", "float16", "float32"),
        default="float16",
        help=(
            "Core ML conversion compute precision. The default is float16 for intended "
            "ANE benchmarking; use 'default' to omit compute_precision."
        ),
    )
    parser.add_argument(
        "--no-require-ne-placement",
        action="store_true",
        help="Do not fail status when ios16.linear/matmul/softmax lack NE-preferred placement.",
    )
    parser.add_argument(
        "--max-rel-diff",
        type=float,
        default=0.05,
        help="Maximum allowed relative diff before the benchmark status is FAIL.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument(
        "--ref-wav",
        type=Path,
        default=DEFAULT_REF_WAV,
        help=(
            f"Reference wav path. Defaults to local {DEFAULT_REF_WAV}; clean checkouts "
            "may need --ref-wav pointing at an existing reference wav."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-convert", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seconds <= 0:
        raise ValueError(f"--seconds must be > 0, got {args.seconds}")
    if args.warmup < 0:
        raise ValueError(f"--warmup must be >= 0, got {args.warmup}")
    if args.iterations <= 0:
        raise ValueError(f"--iterations must be > 0, got {args.iterations}")
    if args.max_rel_diff < 0:
        raise ValueError(f"--max-rel-diff must be >= 0, got {args.max_rel_diff}")
    if args.sequence_length is not None and args.sequence_length <= 0:
        raise ValueError(f"--sequence-length must be > 0, got {args.sequence_length}")

    ct = import_coremltools()
    compute_precision, compute_precision_name = resolve_compute_precision(
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
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"[mode] {MODE_NO_CONTEXT_KV_CACHE}", flush=True)
    print(f"[convert] compute_precision={compute_precision_name}", flush=True)
    print(f"[coremltools] {ct.__version__}", flush=True)
    print(f"[torch] {torch.__version__}", flush=True)

    print("[load] actual checkpoint/model weights on MPS", flush=True)
    model, model_cfg, train_cfg = load_actual_model(checkpoint_path, mps_device)

    print(
        "[prepare] tokenizer, codec-derived length, rem.wav reference, encoded conditions",
        flush=True,
    )
    inputs_mps, metadata = prepare_real_inputs(
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

    print("[benchmark] PyTorch MPS original forward_with_encoded_conditions", flush=True)
    pytorch_output, pytorch_avg_ms = benchmark_torch_step(
        model,
        inputs_mps,
        warmup=int(args.warmup),
        iterations=int(args.iterations),
        device=mps_device,
    )
    cpu_inputs, predict_inputs = cpu_coreml_inputs(inputs_mps)

    print("[convert] moving denoiser weights to CPU for TorchScript/Core ML conversion", flush=True)
    model = model.to(device="cpu")
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    wrapper = RealCoreMLDenoiserStep(model, sequence_length=int(metadata["sequence_length"])).eval()
    with temporary_artifact_dir() as artifact_dir:
        print(f"[artifacts] temporary directory: {artifact_dir}", flush=True)
        mlmodel, convert_seconds = convert_wrapper(
            ct,
            wrapper,
            cpu_inputs,
            artifact_dir=artifact_dir,
            compute_precision=compute_precision,
            verbose=bool(args.verbose_convert),
        )
        print(
            f"[convert] Core ML mlprogram CPU_AND_NE conversion: {convert_seconds:.3f} s",
            flush=True,
        )

        compute_unit = ct.ComputeUnit.CPU_AND_NE
        ne_preferred_counts = compute_plan_counts(ct, mlmodel, compute_unit)
        print(
            "[compute_plan] " + json.dumps(ne_preferred_counts, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

        print("[benchmark] Core ML CPU_AND_NE predict", flush=True)
        coreml_output, coreml_avg_ms = benchmark_coreml_predict(
            mlmodel,
            predict_inputs,
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )

    diff = coreml_output.astype(np.float32) - pytorch_output.astype(np.float32)
    max_abs_diff = float(np.max(np.abs(diff)))
    denom = max(float(np.max(np.abs(pytorch_output.astype(np.float32)))), 1e-8)
    rel_diff = float(max_abs_diff / denom)
    speedup = float(pytorch_avg_ms / coreml_avg_ms) if coreml_avg_ms > 0 else None
    require_ne_placement = not bool(args.no_require_ne_placement)
    status, status_reasons = evaluate_benchmark_status(
        ne_preferred_counts=ne_preferred_counts,
        rel_diff=rel_diff,
        max_rel_diff=float(args.max_rel_diff),
        require_ne_placement=require_ne_placement,
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
        "status": status,
        "status_reasons": status_reasons,
        "sequence_length": int(metadata["sequence_length"]),
        "text_len": int(metadata["text_len"]),
        "ref_len": int(metadata["ref_len"]),
        "compute_precision": compute_precision_name,
        "convert_seconds": float(convert_seconds),
        "pytorch_mps_avg_ms": float(pytorch_avg_ms),
        "coreml_cpu_and_ne_avg_ms": float(coreml_avg_ms),
        "speedup_coreml_vs_mps": speedup,
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel_diff,
        "ne_preferred_counts": ne_preferred_counts,
        "mode": MODE_NO_CONTEXT_KV_CACHE,
        "seconds_derived_sequence_length": int(metadata["seconds_derived_sequence_length"]),
        "sequence_length_source": str(metadata["sequence_length_source"]),
        "text_valid_tokens": int(metadata["text_valid_tokens"]),
        "speaker_context_len": int(metadata["speaker_context_len"]),
        "require_ne_placement": require_ne_placement,
        "max_rel_diff": float(args.max_rel_diff),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
