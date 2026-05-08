from __future__ import annotations

import gc
import json
import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import ModelConfig
from .coreml_cache import (
    ALLOWED_CONDITION_BRANCH_LAYOUTS,
    BRANCH_LAYOUT_ALTERNATING_SPEAKER2,
    BRANCH_LAYOUT_ALTERNATING_TEXT2,
    BRANCH_LAYOUT_COND1,
    BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    BRANCH_LAYOUT_JOINT2,
    STATE_LAYOUT_PER_LAYER,
    ConditionCacheHandle,
    CoreMLConditionBucket,
    expected_per_layer_state_names,
)
from .coreml_stateful import (
    CoreMLStatefulDenoiserBackend,
    CoreMLStatefulUnavailableError,
    pack_context_kv_state,
)
from .lora import checkpoint_state_uses_lora
from .model import TextToLatentRFDiT
from .rf import _make_rng, sample_euler_rf_cfg, scale_speaker_kv_cache, temporal_score_rescale
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None or not hasattr(backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def resolve_runtime_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
        return resolved
    if resolved.type == "mps":
        if resolved.index is not None:
            raise ValueError("MPS device index is not supported. Use 'mps'.")
        if not _is_mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    raise ValueError(f"Unsupported inference device={resolved!s}. Expected one of: cpu, cuda, mps.")


def list_available_runtime_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if _is_mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def default_runtime_device() -> str:
    return list_available_runtime_devices()[0]


def list_available_runtime_precisions(device: str | torch.device) -> list[str]:
    resolved = resolve_runtime_device(device)
    if resolved.type == "cuda":
        return ["fp32", "bf16"]
    return ["fp32"]


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def _sync_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        key = (device.type, device.index)
        if key in seen:
            continue
        _sync_device(device)
        seen.add(key)


def _measure_start(device: torch.device, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter()


def _measure_end(device: torch.device, t0: float, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter() - t0


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


def find_flattening_point(
    latent: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    """
    Echo-style heuristic: find first index where a trailing window becomes near-flat and near-zero.

    Args:
      latent: (T, D) latent sequence.
    Returns:
      Flattening index in [0, T].
    """
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape (T, D), got {tuple(latent.shape)}")
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps

    pad = torch.zeros(
        (window_size, latent.shape[1]),
        device=latent.device,
        dtype=latent.dtype,
    )
    padded = torch.cat([latent, pad], dim=0)
    for i in range(padded.shape[0] - window_size):
        window = padded[i : i + window_size]
        window_std = window.std(unbiased=False)
        window_mean = window.mean()
        if window_std < std_threshold and torch.abs(window_mean - target_value) < mean_threshold:
            return int(i)
    return total_steps


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    enable_watermark: bool = False
    compile_model: bool = False
    compile_dynamic: bool = False


@dataclass
class SamplingRequest:
    text: str
    caption: str | None = None
    ref_wav: str | None = None
    ref_latent: str | None = None
    no_ref: bool = False
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    max_text_len: int | None = None
    max_caption_len: int | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool = True
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    seed: int | None = None
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int
    messages: list[str]


def _default_coreml_stateful_backend_factory(
    runtime: InferenceRuntime,
    *,
    condition_cache: ConditionCacheHandle,
    branch_layout: str,
) -> CoreMLStatefulDenoiserBackend:
    return CoreMLStatefulDenoiserBackend(
        runtime.model,
        sequence_length=int(condition_cache.sequence_length),
        c_ctx_bucket=int(condition_cache.c_ctx_bucket),
        branch_layout=branch_layout,
        state_layout=condition_cache.state_layout,
    )


COREML_STATEFUL_BACKEND_FACTORY = _default_coreml_stateful_backend_factory


def _maybe_compile_inference_model(
    model: TextToLatentRFDiT,
    *,
    enabled: bool,
    dynamic: bool,
) -> TextToLatentRFDiT:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
    compile_kwargs = {"dynamic": bool(dynamic)}
    model.encode_conditions = torch.compile(model.encode_conditions, **compile_kwargs)
    model.build_context_kv_cache = torch.compile(model.build_context_kv_cache, **compile_kwargs)
    model.forward_with_encoded_conditions = torch.compile(
        model.forward_with_encoded_conditions,
        **compile_kwargs,
    )
    return model


def resolve_runtime_dtype(*, precision: str, device: torch.device) -> torch.dtype:
    mode = str(precision).strip().lower()
    if mode == "fp32":
        return torch.float32
    if mode == "bf16":
        if device.type != "cuda":
            raise ValueError("precision='bf16' currently requires CUDA device.")
        return torch.bfloat16
    raise ValueError(f"Unsupported precision={precision!r}. Expected one of: fp32, bf16.")


def resolve_cfg_scales(
    *,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale: float | None,
    use_caption_condition: bool = True,
    use_speaker_condition: bool = True,
) -> tuple[float, float, float, list[str]]:
    """Normalize/validate CFG scales for guidance mode."""
    messages: list[str] = []
    text_val = float(cfg_scale_text)
    caption_val = float(cfg_scale_caption)
    speaker_val = float(cfg_scale_speaker)

    if cfg_scale is not None:
        text_val = float(cfg_scale)
        caption_val = float(cfg_scale)
        speaker_val = float(cfg_scale)
    if not use_speaker_condition:
        if speaker_val > 0.0:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring cfg_scale_speaker."
            )
        speaker_val = 0.0

    mode = str(cfg_guidance_mode).strip().lower()
    enabled_vals = [value for value in (text_val, speaker_val) if value > 0.0]
    if use_caption_condition and caption_val > 0.0:
        enabled_vals.append(caption_val)
    if mode == "joint" and enabled_vals and (max(enabled_vals) - min(enabled_vals) > 1e-6):
        raise ValueError(
            "cfg_guidance_mode='joint' requires equal enabled cfg_scale_text/cfg_scale_caption/cfg_scale_speaker, "
            "or set cfg_scale."
        )

    return text_val, caption_val, speaker_val, messages


def _load_torch_checkpoint_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


_CONFIG_META_KEY = "config_json"
_INFERENCE_CONFIG_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}


def _load_checkpoint_from_pt(path: Path) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    ckpt = _load_torch_checkpoint_payload(path)
    model_state = ckpt.get("model")
    model_cfg = ckpt.get("model_config")
    train_cfg = ckpt.get("train_config")

    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model weights dictionary: {path}")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Checkpoint missing model_config dictionary: {path}")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")

    if checkpoint_state_uses_lora(model_state):
        raise ValueError(
            f"LoRA checkpoints must be loaded from adapter directories or merged safetensors: {path}"
        )
    return model_state, model_cfg, _extract_inference_train_config(train_cfg)


def _parse_json_mapping(
    raw: str | None,
    *,
    field: str,
    path: Path,
    required: bool = False,
) -> dict | None:
    if raw is None:
        if required:
            raise ValueError(f"Missing required metadata field '{field}' in checkpoint: {path}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata for checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object: {path}")
    return payload


def _extract_inference_train_config(raw: dict | None) -> dict | None:
    if raw is None:
        return None

    inference_cfg: dict[str, int] = {}
    for key in _INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, int):
            raise ValueError(f"Inference config key '{key}' must be int, got {type(value)!r}.")
        inference_cfg[key] = int(value)

    return inference_cfg or None


def _split_flat_checkpoint_config(path: Path, flat_config: dict) -> tuple[dict, dict | None]:
    model_cfg: dict[str, object] = {}
    inference_cfg: dict[str, int] = {}
    for key, value in flat_config.items():
        if key in _INFERENCE_CONFIG_KEYS:
            if not isinstance(value, int):
                raise ValueError(
                    f"Inference config key '{key}' must be int in checkpoint metadata: {path}"
                )
            inference_cfg[key] = int(value)
            continue
        model_cfg[key] = value
    return model_cfg, (inference_cfg or None)


def _load_checkpoint_from_safetensors(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    model_state = load_safetensors_file(str(path), device="cpu")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Safetensors checkpoint has no model weights: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    flat_config = _parse_json_mapping(
        metadata.get(_CONFIG_META_KEY),
        field=_CONFIG_META_KEY,
        path=path,
        required=True,
    )
    model_cfg, inference_cfg = _split_flat_checkpoint_config(path=path, flat_config=flat_config)
    return model_state, model_cfg, inference_cfg


def _load_checkpoint_for_inference(path: Path) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    if path.suffix.lower() == ".safetensors":
        return _load_checkpoint_from_safetensors(path)
    return _load_checkpoint_from_pt(path)


class InferenceRuntime:
    def __init__(
        self,
        *,
        key: RuntimeKey,
        model_cfg: ModelConfig,
        train_cfg: dict | None,
        model: TextToLatentRFDiT,
        tokenizer: PretrainedTextTokenizer,
        caption_tokenizer: PretrainedTextTokenizer | None,
        codec: DACVAECodec,
        default_text_max_len: int,
        default_caption_max_len: int,
    ) -> None:
        self.key = key
        self.model_device = resolve_runtime_device(key.model_device)
        self.codec_device = resolve_runtime_device(key.codec_device)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = model
        self.tokenizer = tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.codec = codec
        self.default_text_max_len = default_text_max_len
        self.default_caption_max_len = default_caption_max_len
        self._infer_lock = threading.Lock()
        self._coreml_stateful_backends: dict[tuple[int, int, str, str], object] = {}
        self._coreml_stateful_backends_lock = threading.Lock()

    @classmethod
    def from_key(cls, key: RuntimeKey) -> InferenceRuntime:
        model_device = resolve_runtime_device(key.model_device)
        codec_device = resolve_runtime_device(key.codec_device)
        model_dtype = resolve_runtime_dtype(
            precision=key.model_precision,
            device=model_device,
        )
        codec_dtype = resolve_runtime_dtype(
            precision=key.codec_precision,
            device=codec_device,
        )

        model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(key.checkpoint)
        )
        model_cfg = ModelConfig(**model_cfg_dict)

        model = TextToLatentRFDiT(model_cfg).to(model_device)
        model.load_state_dict(model_state)
        model = model.to(dtype=model_dtype)
        model.eval()
        model = _maybe_compile_inference_model(
            model,
            enabled=bool(key.compile_model),
            dynamic=bool(key.compile_dynamic),
        )

        tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.text_tokenizer_repo,
            add_bos=bool(model_cfg.text_add_bos),
            local_files_only=False,
        )
        if tokenizer.vocab_size != model_cfg.text_vocab_size:
            raise ValueError(
                f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
                f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
            )
        caption_tokenizer = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                repo_id=model_cfg.caption_tokenizer_repo_resolved,
                add_bos=model_cfg.caption_add_bos_resolved,
                local_files_only=False,
            )
            if caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved:
                raise ValueError(
                    f"caption_vocab_size mismatch: checkpoint caption_vocab_size={model_cfg.caption_vocab_size_resolved} but tokenizer ({model_cfg.caption_tokenizer_repo_resolved}) "
                    f"vocab_size={caption_tokenizer.vocab_size}."
                )

        default_text_max_len = 256
        default_caption_max_len = default_text_max_len
        if isinstance(train_cfg, dict):
            ckpt_text_max_len = train_cfg.get("max_text_len")
            if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
                default_text_max_len = int(ckpt_text_max_len)
            ckpt_caption_max_len = train_cfg.get("max_caption_len")
            if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
                default_caption_max_len = int(ckpt_caption_max_len)
            else:
                default_caption_max_len = default_text_max_len

        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device=str(codec_device),
            dtype=codec_dtype,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
            enable_watermark=bool(key.enable_watermark),
        )
        if model_cfg.latent_dim != codec.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} but codec latent_dim={codec.latent_dim}. "
                "Use a compatible codec/checkpoint pair."
            )

        return cls(
            key=key,
            model_cfg=model_cfg,
            train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            default_text_max_len=default_text_max_len,
            default_caption_max_len=default_caption_max_len,
        )

    def _load_reference_latent(
        self,
        *,
        req: SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        runtime_dtype = next(self.model.parameters()).dtype
        if not self.model_cfg.use_speaker_condition:
            if req.ref_wav is not None or req.ref_latent is not None:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring reference input."
                )
            return None, None
        if req.no_ref:
            ref_len = max(1, int(self.model_cfg.speaker_patch_size))
            ref_latent_patched = torch.zeros(
                (
                    batch_size,
                    ref_len,
                    self.model_cfg.latent_dim * self.model_cfg.latent_patch_size,
                ),
                device=self.model_device,
                dtype=runtime_dtype,
            )
            ref_mask = torch.zeros(
                (batch_size, ref_len), dtype=torch.bool, device=self.model_device
            )
            return ref_latent_patched, ref_mask

        if req.ref_wav is None and req.ref_latent is None:
            raise ValueError("Specify either ref_wav/ref_latent, or set no_ref=True.")

        max_ref_latent_steps = None
        if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
            max_ref_latent_steps = max(
                1,
                math.ceil(
                    float(req.max_ref_seconds)
                    * float(self.codec.sample_rate)
                    / float(int(self.codec.model.hop_length))
                ),
            )

        if req.ref_latent is not None:
            latent_raw = torch.load(req.ref_latent, map_location="cpu", weights_only=True)
            ref_latent = _coerce_latent_shape(
                latent_raw, latent_dim=self.model_cfg.latent_dim
            ).unsqueeze(0)
            ref_latent = ref_latent.to(dtype=runtime_dtype)
        else:
            wav, sr = _load_audio(req.ref_wav)
            if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
                max_ref_samples = max(1, int(float(req.max_ref_seconds) * float(sr)))
                if wav.shape[1] > max_ref_samples:
                    messages.append(
                        f"warning: reference audio exceeds max_ref_seconds ({req.max_ref_seconds}s). "
                        f"Trimming from {float(wav.shape[1]) / float(sr):.2f}s to {float(max_ref_samples) / float(sr):.2f}s."
                    )
                    wav = wav[:, :max_ref_samples]
            if req.ref_normalize_db is not None:
                messages.append(
                    f"info: reference loudness normalize enabled (target_db={float(req.ref_normalize_db):.2f}, includes peak safety scaling)."
                )
            elif req.ref_ensure_max:
                messages.append("info: reference peak safety scaling enabled (ensure_max=True).")
            ref_latent = self.codec.encode_waveform(
                wav.unsqueeze(0),
                sample_rate=int(sr),
                normalize_db=req.ref_normalize_db,
                ensure_max=bool(req.ref_ensure_max),
            ).cpu()

        if max_ref_latent_steps is not None and ref_latent.shape[1] > max_ref_latent_steps:
            messages.append(
                f"warning: reference latent steps ({ref_latent.shape[1]}) exceed max_ref_seconds bound ({max_ref_latent_steps} steps). "
                "Trimming reference latent."
            )
            ref_latent = ref_latent[:, :max_ref_latent_steps]

        ref_latent_patched = patchify_latent(ref_latent, self.model_cfg.latent_patch_size).to(
            self.model_device
        )
        if ref_latent_patched.shape[1] == 0:
            raise ValueError(
                "Reference latent length became zero after patchify. Use longer reference audio."
            )
        if batch_size > 1:
            ref_latent_patched = ref_latent_patched.repeat(batch_size, 1, 1)
        ref_mask = torch.ones(
            (batch_size, ref_latent_patched.shape[1]), dtype=torch.bool, device=self.model_device
        )
        return ref_latent_patched, ref_mask

    def _coreml_stateful_dict_lock(self) -> threading.Lock:
        lock = getattr(self, "_coreml_stateful_backends_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._coreml_stateful_backends_lock = lock
        return lock

    def _get_coreml_stateful_backend(
        self,
        *,
        condition_cache: ConditionCacheHandle,
        branch_layout: str,
    ) -> object:
        key = (
            int(condition_cache.sequence_length),
            int(condition_cache.c_ctx_bucket),
            str(condition_cache.state_layout),
            str(branch_layout),
        )
        with self._coreml_stateful_dict_lock():
            backend = self._coreml_stateful_backends.get(key)
            if backend is None:
                backend = COREML_STATEFUL_BACKEND_FACTORY(
                    self,
                    condition_cache=condition_cache,
                    branch_layout=branch_layout,
                )
                self._coreml_stateful_backends[key] = backend
        return backend

    def precompile_coreml_stateful_buckets(
        self,
        buckets: list[CoreMLConditionBucket] | tuple[CoreMLConditionBucket, ...],
    ) -> None:
        """Precompile cond1 CoreML backends for each supplied bucket."""
        from datetime import datetime, timezone

        for bucket in buckets:
            synthetic_handle = ConditionCacheHandle(
                id="precompile",
                reference_cache_id="precompile",
                model_fingerprint="precompile",
                tokenizer_fingerprint="precompile",
                condition_fingerprint="precompile",
                bucket_id=bucket.bucket_id(BRANCH_LAYOUT_COND1),
                state_layout=STATE_LAYOUT_PER_LAYER,
                sequence_length=int(bucket.sequence_length),
                text_len=int(bucket.text_len),
                speaker_context_len=int(bucket.speaker_context_len_bucket),
                speaker_context_len_bucket=int(bucket.speaker_context_len_bucket),
                c_ctx_bucket=int(bucket.c_ctx_bucket),
                branch_layouts=(BRANCH_LAYOUT_COND1,),
                mlstate_keys=expected_per_layer_state_names(),
                created_at=datetime.now(timezone.utc),
                expires_at=None,
                memory_bytes=0,
                metadata={},
            )
            backend = self._get_coreml_stateful_backend(
                condition_cache=synthetic_handle,
                branch_layout=BRANCH_LAYOUT_COND1,
            )
            precompile = getattr(backend, "precompile", None)
            if callable(precompile):
                precompile()

    def coreml_stateful_metrics_snapshot(self) -> list[dict[str, object]]:
        with self._coreml_stateful_dict_lock():
            backends = list(self._coreml_stateful_backends.values())
        snapshots: list[dict[str, object]] = []
        for backend in backends:
            snapshot_fn = getattr(backend, "metrics_snapshot", None)
            if callable(snapshot_fn):
                snapshots.append(snapshot_fn())
        return snapshots

    def synthesize_with_condition_cache(
        self,
        req: SamplingRequest,
        *,
        condition_cache: ConditionCacheHandle,
        log_fn: Callable[[str], None] | None = None,
    ) -> SamplingResult:
        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        self._validate_coreml_stateful_surface(req, condition_cache=condition_cache)
        messages: list[str] = []
        _log(
            (
                "[runtime] start synthesize_with_condition_cache "
                "backend=coreml-stateful model_device={} model_precision={} "
                "codec_device={} codec_precision={} seconds={} steps={} seed={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                req.seconds,
                req.num_steps,
                "random" if req.seed is None else int(req.seed),
            )
        )

        if req.seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {req.seconds}")
        num_candidates = int(req.num_candidates)
        if num_candidates != 1:
            raise CoreMLStatefulUnavailableError(
                f"CoreML stateful fast path supports num_candidates=1, got {num_candidates}"
            )
        decode_mode = str(req.decode_mode).strip().lower()
        if decode_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"Unsupported decode_mode={req.decode_mode!r}. Expected one of: sequential, batch."
            )

        raw_text = str(req.text)
        normalized_text = normalize_text(raw_text).strip()
        if normalized_text == "":
            raise ValueError("text became empty after normalization.")

        text_max_len = int(condition_cache.text_len)
        if req.max_text_len is not None and int(req.max_text_len) != text_max_len:
            raise CoreMLStatefulUnavailableError(
                "CoreML condition cache text_len does not match requested max_text_len"
            )
        caption_max_len = (
            self.default_caption_max_len
            if req.max_caption_len is None
            else int(req.max_caption_len)
        )
        if self.model_cfg.use_caption_condition and caption_max_len <= 0:
            raise ValueError(f"max_caption_len must be > 0, got {caption_max_len}")
        has_caption_text = bool(
            self.model_cfg.use_caption_condition
            and req.caption is not None
            and str(req.caption).strip() != ""
        )

        truncation_factor = None if req.truncation_factor is None else float(req.truncation_factor)
        rescale_k = None if req.rescale_k is None else float(req.rescale_k)
        rescale_sigma = None if req.rescale_sigma is None else float(req.rescale_sigma)
        if truncation_factor is not None and truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and rescale_k <= 0:
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and rescale_sigma <= 0:
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

        speaker_kv_scale = None if req.speaker_kv_scale is None else float(req.speaker_kv_scale)
        speaker_kv_min_t: float | None = None
        speaker_kv_max_layers = (
            None if req.speaker_kv_max_layers is None else int(req.speaker_kv_max_layers)
        )
        if speaker_kv_scale is not None:
            if speaker_kv_scale <= 0:
                raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
            speaker_kv_min_t = 0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)
            if not (0.0 <= speaker_kv_min_t <= 1.0):
                raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
            if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                raise ValueError(
                    f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                )

        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise CoreMLStatefulUnavailableError(
                f"CoreML stateful fast path does not support cfg_guidance_mode={cfg_mode!r}"
            )
        cfg_min_t = float(req.cfg_min_t)
        cfg_max_t = float(req.cfg_max_t)
        if not (0.0 <= cfg_min_t <= cfg_max_t <= 1.0):
            raise ValueError("cfg_min_t/cfg_max_t must satisfy 0.0 <= min <= max <= 1.0")

        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_scale=req.cfg_scale,
            use_caption_condition=has_caption_text,
            use_speaker_condition=self.model_cfg.use_speaker_condition,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)
        if has_caption_text and cfg_scale_caption > 0.0:
            raise CoreMLStatefulUnavailableError(
                "caption CFG is not supported by the CoreML stateful fast path"
            )

        branch_layouts = tuple(condition_cache.branch_layouts)
        active_cfg_requested = bool(cfg_scale_text > 0.0 or cfg_scale_speaker > 0.0)
        enabled_alt_names: list[str] = []
        if cfg_mode == "alternating":
            if cfg_scale_text > 0.0:
                enabled_alt_names.append("text")
            if cfg_scale_speaker > 0.0:
                enabled_alt_names.append("speaker")
        if active_cfg_requested:
            if cfg_mode == "independent" and BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3 not in (
                branch_layouts
            ):
                raise CoreMLStatefulUnavailableError(
                    "active independent CFG requires independent_text_speaker3 CoreML state"
                )
            if cfg_mode == "joint" and BRANCH_LAYOUT_JOINT2 not in branch_layouts:
                raise CoreMLStatefulUnavailableError(
                    "active joint CFG requires joint2 CoreML state"
                )
            if cfg_mode == "alternating":
                if (
                    "text" in enabled_alt_names
                    and BRANCH_LAYOUT_ALTERNATING_TEXT2 not in branch_layouts
                ):
                    raise CoreMLStatefulUnavailableError(
                        "alternating text CFG requires alternating_text2 CoreML state"
                    )
                if (
                    "speaker" in enabled_alt_names
                    and BRANCH_LAYOUT_ALTERNATING_SPEAKER2 not in branch_layouts
                ):
                    raise CoreMLStatefulUnavailableError(
                        "alternating speaker CFG requires alternating_speaker2 CoreML state"
                    )

        stage_timings: list[tuple[str, float]] = []
        if req.seed is None:
            used_seed = int(secrets.randbits(63))
            msg = f"info: seed not specified; using random seed {used_seed}."
            messages.append(msg)
            _log(msg)
        else:
            used_seed = int(req.seed)
            _log(f"[runtime] using seed: {used_seed}")
        post_load_t0 = _measure_start(self.model_device, self.codec_device)

        with self._infer_lock, torch.inference_mode():
            t0 = _measure_start(self.model_device)
            text_ids, text_mask = self.tokenizer.batch_encode(
                [normalized_text],
                max_length=text_max_len,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("tokenize_text", stage_sec))
            _log(f"[runtime] tokenize_text: {stage_sec * 1000.0:.1f} ms")
            text_ids = text_ids.to(self.model_device)
            text_mask = text_mask.to(self.model_device)

            target_samples = int(float(req.seconds) * self.codec.sample_rate)
            latent_steps = math.ceil(target_samples / int(self.codec.model.hop_length))
            patched_steps = math.ceil(latent_steps / self.model_cfg.latent_patch_size)
            bucket_steps = int(condition_cache.sequence_length)
            if patched_steps > bucket_steps:
                raise CoreMLStatefulUnavailableError(
                    "requested patched_steps "
                    f"{patched_steps} exceeds CoreML condition cache bucket {bucket_steps}"
                )

            t0 = _measure_start(self.model_device, self.codec_device)
            msg_count_before_ref = len(messages)
            ref_latent, ref_mask = self._load_reference_latent(
                req=req,
                batch_size=1,
                messages=messages,
            )
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("prepare_reference", stage_sec))
            for msg in messages[msg_count_before_ref:]:
                _log(msg)
            _log(f"[runtime] prepare_reference: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z_patched = self._sample_coreml_stateful_rf_cfg(
                condition_cache=condition_cache,
                text_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=bucket_steps,
                actual_sequence_length=patched_steps,
                num_steps=int(req.num_steps),
                cfg_guidance_mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_min_t=cfg_min_t,
                cfg_max_t=cfg_max_t,
                seed=used_seed,
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_min_t=speaker_kv_min_t,
                speaker_kv_max_layers=speaker_kv_max_layers,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("sample_rf_coreml_stateful", stage_sec))
            _log(f"[runtime] sample_rf_coreml_stateful: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z = unpatchify_latent(
                z_patched,
                patch_size=self.model_cfg.latent_patch_size,
                latent_dim=self.model_cfg.latent_dim,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("unpatchify_latent", stage_sec))
            _log(f"[runtime] unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
            z = z[:, :latent_steps]

            t0 = _measure_start(self.model_device, self.codec_device)
            trimmed_audios: list[torch.Tensor] = []
            if decode_mode == "batch":
                audio_batch = self.codec.decode_latent(z).cpu()
                for i in range(num_candidates):
                    audio_i = audio_batch[i]
                    max_samples = self._trimmed_sample_count(req, z[i], target_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            else:
                for i in range(num_candidates):
                    audio_i = self.codec.decode_latent(z[i : i + 1]).cpu()[0]
                    max_samples = self._trimmed_sample_count(req, z[i], target_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("decode_latent", stage_sec))
            _log(f"[runtime] decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

            total_to_decode = _measure_end(self.model_device, post_load_t0, self.codec_device)
            _log(f"[runtime] total_to_decode: {total_to_decode:.3f} s")

        _log("[runtime] done synthesize_with_condition_cache")
        return SamplingResult(
            audio=trimmed_audios[0],
            audios=trimmed_audios,
            sample_rate=int(self.codec.sample_rate),
            stage_timings=stage_timings,
            total_to_decode=total_to_decode,
            used_seed=used_seed,
            messages=messages,
        )

    def _validate_coreml_stateful_surface(
        self,
        req: SamplingRequest,
        *,
        condition_cache: ConditionCacheHandle,
    ) -> None:
        if not isinstance(condition_cache, ConditionCacheHandle):
            raise CoreMLStatefulUnavailableError("condition_cache must be a ConditionCacheHandle")
        if int(req.num_candidates) != 1:
            raise CoreMLStatefulUnavailableError(
                f"CoreML stateful fast path supports num_candidates=1, got {req.num_candidates}"
            )
        if condition_cache.state_layout != STATE_LAYOUT_PER_LAYER:
            raise CoreMLStatefulUnavailableError(
                f"unsupported CoreML state layout: {condition_cache.state_layout}"
            )
        branch_layouts = tuple(condition_cache.branch_layouts)
        if branch_layouts not in ALLOWED_CONDITION_BRANCH_LAYOUTS:
            raise CoreMLStatefulUnavailableError(
                f"unsupported CoreML branch layouts: {branch_layouts}"
            )
        if self.model_cfg.use_caption_condition:
            raise CoreMLStatefulUnavailableError(
                "caption-conditioned checkpoints are not supported by the CoreML stateful fast path"
            )
        if not self.model_cfg.use_speaker_condition:
            raise CoreMLStatefulUnavailableError(
                "speaker-conditioned checkpoints are required by the CoreML stateful fast path"
            )
        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise CoreMLStatefulUnavailableError(
                f"cfg_guidance_mode={cfg_mode!r} is not supported by the CoreML stateful fast path"
            )

    def _sample_coreml_stateful_rf_cfg(
        self,
        *,
        condition_cache: ConditionCacheHandle,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        sequence_length: int,
        actual_sequence_length: int,
        num_steps: int,
        cfg_guidance_mode: str = "independent",
        cfg_scale_text: float = 0.0,
        cfg_scale_speaker: float = 0.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        seed: int = 0,
        truncation_factor: float | None = None,
        rescale_k: float | None = None,
        rescale_sigma: float | None = None,
        speaker_kv_scale: float | None = None,
        speaker_kv_min_t: float | None = None,
        speaker_kv_max_layers: int | None = None,
    ) -> torch.Tensor:
        device = self.model_device
        dtype = next(self.model.parameters()).dtype
        latent_dim = int(self.model_cfg.patched_latent_dim)
        rng, rng_device = _make_rng(seed=seed, device=device)
        x_t = torch.randn(
            (1, int(sequence_length), latent_dim),
            device=rng_device,
            dtype=dtype,
            generator=rng,
        )
        if rng_device != device:
            x_t = x_t.to(device=device)
        if truncation_factor is not None:
            x_t = x_t * float(truncation_factor)

        latent_mask = torch.zeros((1, int(sequence_length)), dtype=torch.bool, device=device)
        latent_mask[:, : int(actual_sequence_length)] = True
        t_schedule = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device) * 0.999

        (
            text_state_cond,
            text_mask_cond,
            speaker_state_cond,
            speaker_mask_cond,
            caption_state_cond,
            _caption_mask_cond,
        ) = self.model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=None,
            caption_mask=None,
        )
        if caption_state_cond is not None:
            raise CoreMLStatefulUnavailableError(
                "caption state is not supported by the CoreML stateful fast path"
            )
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise CoreMLStatefulUnavailableError(
                "speaker state is required by the CoreML stateful fast path"
            )

        speaker_context_bucket = int(condition_cache.speaker_context_len_bucket)
        backend_cond = self._get_coreml_stateful_backend(
            condition_cache=condition_cache,
            branch_layout=BRANCH_LAYOUT_COND1,
        )

        def _pack_and_prepare(
            *,
            text_state: torch.Tensor,
            text_mask_val: torch.Tensor,
            speaker_state: torch.Tensor,
            speaker_mask_val: torch.Tensor,
            apply_speaker_kv_scale: bool,
        ):
            context_kv = self.model.build_context_kv_cache(
                text_state=text_state,
                speaker_state=speaker_state,
                caption_state=None,
            )
            if apply_speaker_kv_scale and speaker_kv_scale is not None:
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv,
                    scale=float(speaker_kv_scale),
                    max_layers=speaker_kv_max_layers,
                )
            payload = pack_context_kv_state(
                context_kv,
                text_mask=text_mask_val,
                speaker_mask=speaker_mask_val,
                speaker_context_bucket=speaker_context_bucket,
                branch_layout=BRANCH_LAYOUT_COND1,
            )
            return backend_cond.prepare_state(payload)

        text_state_uncond = torch.zeros_like(text_state_cond)
        text_mask_uncond = torch.zeros_like(text_mask_cond)
        speaker_state_uncond = torch.zeros_like(speaker_state_cond)
        speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)

        cfg_mode = str(cfg_guidance_mode).strip().lower()
        enabled_cfg_names: list[str] = []
        cfg_scales: dict[str, float] = {}
        if cfg_scale_text > 0.0:
            enabled_cfg_names.append("text")
            cfg_scales["text"] = float(cfg_scale_text)
        if cfg_scale_speaker > 0.0:
            enabled_cfg_names.append("speaker")
            cfg_scales["speaker"] = float(cfg_scale_speaker)
        active_cfg_possible = bool(enabled_cfg_names)

        # Always prepare cond state(s). If speaker_kv_scale set, prepare both normal and scaled.
        state_cond_normal = _pack_and_prepare(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            apply_speaker_kv_scale=False,
        )
        state_cond_scaled = None
        if speaker_kv_scale is not None:
            state_cond_scaled = _pack_and_prepare(
                text_state=text_state_cond,
                text_mask_val=text_mask_cond,
                speaker_state=speaker_state_cond,
                speaker_mask_val=speaker_mask_cond,
                apply_speaker_kv_scale=True,
            )

        # Mode-specific uncond states.
        state_independent_text_uncond_normal = None
        state_independent_text_uncond_scaled = None
        state_independent_speaker_uncond = None
        state_joint_uncond = None
        state_alternating: dict[str, dict[str, object | None]] = {}

        if active_cfg_possible:
            if cfg_mode == "independent":
                # text-uncond branch keeps speaker_cond, so it benefits from speaker_kv_scale.
                if "text" in enabled_cfg_names:
                    state_independent_text_uncond_normal = _pack_and_prepare(
                        text_state=text_state_uncond,
                        text_mask_val=text_mask_uncond,
                        speaker_state=speaker_state_cond,
                        speaker_mask_val=speaker_mask_cond,
                        apply_speaker_kv_scale=False,
                    )
                    if speaker_kv_scale is not None:
                        state_independent_text_uncond_scaled = _pack_and_prepare(
                            text_state=text_state_uncond,
                            text_mask_val=text_mask_uncond,
                            speaker_state=speaker_state_cond,
                            speaker_mask_val=speaker_mask_cond,
                            apply_speaker_kv_scale=True,
                        )
                # speaker-uncond branch zeroes speaker, so scaling has no effect.
                if "speaker" in enabled_cfg_names:
                    state_independent_speaker_uncond = _pack_and_prepare(
                        text_state=text_state_cond,
                        text_mask_val=text_mask_cond,
                        speaker_state=speaker_state_uncond,
                        speaker_mask_val=speaker_mask_uncond,
                        apply_speaker_kv_scale=False,
                    )
            elif cfg_mode == "joint":
                # Joint requires equal scales. Validate by reusing resolve_cfg_scales semantics.
                if len(enabled_cfg_names) > 1:
                    joint_scales = [cfg_scales[name] for name in enabled_cfg_names]
                    if max(joint_scales) - min(joint_scales) > 1e-6:
                        raise ValueError(
                            "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                            "set matching cfg_scale_text/cfg_scale_speaker or use cfg_scale.",
                        )
                # joint_uncond zeroes both text and speaker -> no speaker, scaling no-op.
                state_joint_uncond = _pack_and_prepare(
                    text_state=text_state_uncond,
                    text_mask_val=text_mask_uncond,
                    speaker_state=speaker_state_uncond,
                    speaker_mask_val=speaker_mask_uncond,
                    apply_speaker_kv_scale=False,
                )
            elif cfg_mode == "alternating":
                if "text" in enabled_cfg_names:
                    # text alt-uncond keeps speaker_cond.
                    text_alt_normal = _pack_and_prepare(
                        text_state=text_state_uncond,
                        text_mask_val=text_mask_uncond,
                        speaker_state=speaker_state_cond,
                        speaker_mask_val=speaker_mask_cond,
                        apply_speaker_kv_scale=False,
                    )
                    text_alt_scaled = None
                    if speaker_kv_scale is not None:
                        text_alt_scaled = _pack_and_prepare(
                            text_state=text_state_uncond,
                            text_mask_val=text_mask_uncond,
                            speaker_state=speaker_state_cond,
                            speaker_mask_val=speaker_mask_cond,
                            apply_speaker_kv_scale=True,
                        )
                    state_alternating["text"] = {
                        "normal": text_alt_normal,
                        "scaled": text_alt_scaled,
                    }
                if "speaker" in enabled_cfg_names:
                    speaker_alt_normal = _pack_and_prepare(
                        text_state=text_state_cond,
                        text_mask_val=text_mask_cond,
                        speaker_state=speaker_state_uncond,
                        speaker_mask_val=speaker_mask_uncond,
                        apply_speaker_kv_scale=False,
                    )
                    state_alternating["speaker"] = {
                        "normal": speaker_alt_normal,
                        "scaled": None,
                    }
            else:
                raise CoreMLStatefulUnavailableError(
                    f"CoreML stateful fast path does not support cfg_guidance_mode={cfg_mode!r}"
                )

        for i in range(int(num_steps)):
            t = t_schedule[i]
            t_next = t_schedule[i + 1]
            tt = torch.full((1,), t, device=device, dtype=dtype)
            use_cfg = active_cfg_possible and (
                float(cfg_min_t) <= float(t.item()) <= float(cfg_max_t)
            )
            speaker_kv_active_now = (
                speaker_kv_scale is not None
                and speaker_kv_min_t is not None
                and float(t.item()) >= float(speaker_kv_min_t)
            )
            state_cond_for_step = (
                state_cond_scaled
                if speaker_kv_active_now and state_cond_scaled is not None
                else state_cond_normal
            )
            cond = backend_cond.predict_step(
                state_cond_for_step,
                x_t=x_t,
                t=tt,
                latent_mask=latent_mask,
            )
            if use_cfg:
                if cfg_mode == "independent":
                    v = cond
                    if "text" in enabled_cfg_names:
                        text_state_for_step = (
                            state_independent_text_uncond_scaled
                            if (
                                speaker_kv_active_now
                                and state_independent_text_uncond_scaled is not None
                            )
                            else state_independent_text_uncond_normal
                        )
                        if text_state_for_step is None:
                            raise CoreMLStatefulUnavailableError(
                                "independent text CFG requires text uncond state",
                            )
                        text_uncond = backend_cond.predict_step(
                            text_state_for_step,
                            x_t=x_t,
                            t=tt,
                            latent_mask=latent_mask,
                        )
                        v = v + cfg_scales["text"] * (cond - text_uncond)
                    if "speaker" in enabled_cfg_names:
                        if state_independent_speaker_uncond is None:
                            raise CoreMLStatefulUnavailableError(
                                "independent speaker CFG requires speaker uncond state",
                            )
                        speaker_uncond = backend_cond.predict_step(
                            state_independent_speaker_uncond,
                            x_t=x_t,
                            t=tt,
                            latent_mask=latent_mask,
                        )
                        v = v + cfg_scales["speaker"] * (cond - speaker_uncond)
                elif cfg_mode == "joint":
                    if state_joint_uncond is None:
                        raise CoreMLStatefulUnavailableError(
                            "joint CFG requires joint uncond state",
                        )
                    joint_uncond = backend_cond.predict_step(
                        state_joint_uncond,
                        x_t=x_t,
                        t=tt,
                        latent_mask=latent_mask,
                    )
                    joint_scale = cfg_scales[enabled_cfg_names[0]]
                    v = cond + joint_scale * (cond - joint_uncond)
                elif cfg_mode == "alternating":
                    alt_name = enabled_cfg_names[i % len(enabled_cfg_names)]
                    alt_states = state_alternating.get(alt_name)
                    if alt_states is None:
                        raise CoreMLStatefulUnavailableError(
                            f"alternating {alt_name} CFG requires alt uncond state",
                        )
                    alt_state_for_step = (
                        alt_states["scaled"]
                        if speaker_kv_active_now and alt_states.get("scaled") is not None
                        else alt_states["normal"]
                    )
                    alt_uncond = backend_cond.predict_step(
                        alt_state_for_step,
                        x_t=x_t,
                        t=tt,
                        latent_mask=latent_mask,
                    )
                    v = cond + cfg_scales[alt_name] * (cond - alt_uncond)
                else:
                    raise CoreMLStatefulUnavailableError(
                        f"CoreML stateful fast path does not support cfg_guidance_mode={cfg_mode!r}"
                    )
            else:
                v = cond

            if rescale_k is not None and rescale_sigma is not None:
                v = temporal_score_rescale(
                    v_pred=v,
                    x_t=x_t,
                    t=t,
                    rescale_k=float(rescale_k),
                    rescale_sigma=float(rescale_sigma),
                )
            x_t = x_t + v * (t_next - t)

        return x_t[:, : int(actual_sequence_length)]

    def _trimmed_sample_count(
        self,
        req: SamplingRequest,
        z: torch.Tensor,
        target_samples: int,
    ) -> int:
        max_samples = target_samples
        if bool(req.trim_tail):
            flattening_point = find_flattening_point(
                z,
                window_size=max(1, int(req.tail_window_size)),
                std_threshold=float(req.tail_std_threshold),
                mean_threshold=float(req.tail_mean_threshold),
            )
            flattening_samples = int(flattening_point * int(self.codec.model.hop_length))
            if flattening_samples > 0:
                max_samples = min(max_samples, flattening_samples)
        return max_samples

    def synthesize(
        self,
        req: SamplingRequest,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> SamplingResult:
        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        messages: list[str] = []
        _log(
            (
                "[runtime] start synthesize "
                "model_device={} model_precision={} codec_device={} codec_precision={} "
                "watermark={} mode={} seconds={} steps={} seed={} candidates={} decode_mode={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                self.codec.enable_watermark,
                req.cfg_guidance_mode,
                req.seconds,
                req.num_steps,
                "random" if req.seed is None else int(req.seed),
                req.num_candidates,
                req.decode_mode,
            )
        )

        if req.seconds <= 0:
            raise ValueError(f"seconds must be > 0, got {req.seconds}")
        num_candidates = int(req.num_candidates)
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {num_candidates}")
        decode_mode = str(req.decode_mode).strip().lower()
        if decode_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"Unsupported decode_mode={req.decode_mode!r}. Expected one of: sequential, batch."
            )

        raw_text = str(req.text)
        normalized_text = normalize_text(raw_text).strip()
        if normalized_text == "":
            raise ValueError("text became empty after normalization.")

        text_max_len = (
            self.default_text_max_len if req.max_text_len is None else int(req.max_text_len)
        )
        if text_max_len <= 0:
            raise ValueError(f"max_text_len must be > 0, got {text_max_len}")
        caption_max_len = (
            self.default_caption_max_len
            if req.max_caption_len is None
            else int(req.max_caption_len)
        )
        if self.model_cfg.use_caption_condition and caption_max_len <= 0:
            raise ValueError(f"max_caption_len must be > 0, got {caption_max_len}")
        has_caption_text = bool(
            self.model_cfg.use_caption_condition
            and req.caption is not None
            and str(req.caption).strip() != ""
        )

        truncation_factor = None if req.truncation_factor is None else float(req.truncation_factor)
        rescale_k = None if req.rescale_k is None else float(req.rescale_k)
        rescale_sigma = None if req.rescale_sigma is None else float(req.rescale_sigma)
        if truncation_factor is not None and truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and rescale_k <= 0:
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and rescale_sigma <= 0:
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

        speaker_kv_scale = None if req.speaker_kv_scale is None else float(req.speaker_kv_scale)
        speaker_kv_min_t = None
        speaker_kv_max_layers = (
            None if req.speaker_kv_max_layers is None else int(req.speaker_kv_max_layers)
        )
        if speaker_kv_scale is not None:
            if not self.model_cfg.use_speaker_condition:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring speaker_kv_scale."
                )
                speaker_kv_scale = None
            else:
                if speaker_kv_scale <= 0:
                    raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
                speaker_kv_min_t = (
                    0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)
                )
                if not (0.0 <= speaker_kv_min_t <= 1.0):
                    raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
                if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                    raise ValueError(
                        f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                    )

        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise ValueError(
                f"Unsupported cfg_guidance_mode={req.cfg_guidance_mode!r}. "
                "Expected one of: independent, joint, alternating."
            )

        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_scale=req.cfg_scale,
            use_caption_condition=has_caption_text,
            use_speaker_condition=self.model_cfg.use_speaker_condition,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)

        stage_timings: list[tuple[str, float]] = []
        if req.seed is None:
            used_seed = int(secrets.randbits(63))
            msg = f"info: seed not specified; using random seed {used_seed}."
            messages.append(msg)
            _log(msg)
        else:
            used_seed = int(req.seed)
            _log(f"[runtime] using seed: {used_seed}")
        post_load_t0 = _measure_start(self.model_device, self.codec_device)

        with self._infer_lock, torch.inference_mode():
            t0 = _measure_start(self.model_device)
            text_ids, text_mask = self.tokenizer.batch_encode(
                [normalized_text] * num_candidates,
                max_length=text_max_len,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("tokenize_text", stage_sec))
            _log(f"[runtime] tokenize_text: {stage_sec * 1000.0:.1f} ms")
            text_ids = text_ids.to(self.model_device)
            text_mask = text_mask.to(self.model_device)
            caption_ids = None
            caption_mask = None
            if self.model_cfg.use_caption_condition:
                if self.caption_tokenizer is None:
                    raise RuntimeError(
                        "Caption conditioning is enabled but caption tokenizer is not loaded."
                    )
                caption_text = "" if req.caption is None else str(req.caption).strip()
                caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                    [caption_text] * num_candidates,
                    max_length=caption_max_len,
                )
                if caption_text == "":
                    caption_mask.zero_()
                caption_ids = caption_ids.to(self.model_device)
                caption_mask = caption_mask.to(self.model_device)

            target_samples = int(float(req.seconds) * self.codec.sample_rate)
            latent_steps = math.ceil(target_samples / int(self.codec.model.hop_length))
            patched_steps = math.ceil(latent_steps / self.model_cfg.latent_patch_size)

            if isinstance(self.train_cfg, dict):
                fixed_steps = self.train_cfg.get("fixed_target_latent_steps")
                if isinstance(fixed_steps, int) and fixed_steps > 0 and latent_steps > fixed_steps:
                    msg = (
                        f"warning: requested latent length ({latent_steps}) exceeds fixed_target_latent_steps ({fixed_steps}) "
                        "used in training. Long-tail stability may degrade."
                    )
                    messages.append(msg)
                    _log(msg)

            t0 = _measure_start(self.model_device, self.codec_device)
            msg_count_before_ref = len(messages)
            ref_latent, ref_mask = self._load_reference_latent(
                req=req,
                batch_size=num_candidates,
                messages=messages,
            )
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("prepare_reference", stage_sec))
            for msg in messages[msg_count_before_ref:]:
                _log(msg)
            _log(f"[runtime] prepare_reference: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z_patched = sample_euler_rf_cfg(
                model=self.model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=patched_steps,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                num_steps=int(req.num_steps),
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode=cfg_mode,
                cfg_min_t=float(req.cfg_min_t),
                cfg_max_t=float(req.cfg_max_t),
                seed=used_seed,
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                use_context_kv_cache=bool(req.context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_max_layers=speaker_kv_max_layers,
                speaker_kv_min_t=speaker_kv_min_t,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("sample_rf", stage_sec))
            _log(f"[runtime] sample_rf: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z = unpatchify_latent(
                z_patched,
                patch_size=self.model_cfg.latent_patch_size,
                latent_dim=self.model_cfg.latent_dim,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("unpatchify_latent", stage_sec))
            _log(f"[runtime] unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
            z = z[:, :latent_steps]

            t0 = _measure_start(self.model_device, self.codec_device)
            trimmed_audios: list[torch.Tensor] = []
            if decode_mode == "batch":
                audio_batch = self.codec.decode_latent(z).cpu()
                for i in range(num_candidates):
                    audio_i = audio_batch[i]
                    max_samples = target_samples
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            else:
                for i in range(num_candidates):
                    audio_i = self.codec.decode_latent(z[i : i + 1]).cpu()[0]
                    max_samples = target_samples
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("decode_latent", stage_sec))
            _log(f"[runtime] decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

            total_to_decode = _measure_end(self.model_device, post_load_t0, self.codec_device)
            _log(f"[runtime] total_to_decode: {total_to_decode:.3f} s")

        _log("[runtime] done synthesize")
        return SamplingResult(
            audio=trimmed_audios[0],
            audios=trimmed_audios,
            sample_rate=int(self.codec.sample_rate),
            stage_timings=stage_timings,
            total_to_decode=total_to_decode,
            used_seed=used_seed,
            messages=messages,
        )

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        del self.codec
        gc.collect()
        for device in (self.model_device, self.codec_device):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                mps = getattr(torch, "mps", None)
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()


_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE_KEY: RuntimeKey | None = None
_RUNTIME_CACHE_VALUE: InferenceRuntime | None = None


def get_cached_runtime(key: RuntimeKey) -> tuple[InferenceRuntime, bool]:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        if _RUNTIME_CACHE_VALUE is not None and _RUNTIME_CACHE_KEY == key:
            return _RUNTIME_CACHE_VALUE, False

        old_runtime = _RUNTIME_CACHE_VALUE
        runtime = InferenceRuntime.from_key(key)
        _RUNTIME_CACHE_KEY = key
        _RUNTIME_CACHE_VALUE = runtime

    if old_runtime is not None:
        old_runtime.unload()

    return runtime, True


def clear_cached_runtime() -> None:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE_VALUE
        _RUNTIME_CACHE_KEY = None
        _RUNTIME_CACHE_VALUE = None

    if runtime is not None:
        runtime.unload()


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, sr


def save_wav(path: str | Path, audio: torch.Tensor, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torchaudio.save(str(out_path), audio, sample_rate)
    except RuntimeError:
        import soundfile as sf

        sf.write(str(out_path), audio.squeeze(0).numpy(), sample_rate)
    return out_path
