#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download

from irodori_tts.coreml_cache import (
    BRANCH_LAYOUT_ALTERNATING_SPEAKER2,
    BRANCH_LAYOUT_ALTERNATING_TEXT2,
    BRANCH_LAYOUT_COND1,
    BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    BRANCH_LAYOUT_JOINT2,
    CACHE_MODE_AUTO,
    CACHE_MODE_OFF,
    CACHE_MODE_PREPARE,
    CACHE_MODE_REFRESH,
    CACHE_MODE_REQUIRE,
    CacheConflictError,
    CacheCreateResult,
    CacheExpiredError,
    CacheNotFoundError,
    CacheValidationError,
    ConditionCacheHandle,
    ConditionCacheRequest,
    CoreMLConditionBucket,
    InMemoryCoreMLCacheManager,
    ReferenceCacheRequest,
    cache_create_status_code,
    cache_exception_to_http_error,
    condition_cache_response,
    normalize_speech_cache_mode,
    reference_cache_response,
)
from irodori_tts.coreml_stateful import CoreMLStatefulUnavailableError
from irodori_tts.inference_runtime import (
    ResidentReferenceTensors,
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
)
from irodori_tts.text_normalization import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_API_MODEL_ID = "irodori-tts-500m-v2"
MODEL_CREATED = 1700000000
SUPPORTED_RESPONSE_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
SECONDS_ROUND_INCREMENT = 0.5
MAX_SAFE_SEGMENT_SECONDS = 70.0
CACHE_MODE_CREATE_OR_REUSE = "create_or_reuse"
CACHE_MANAGER_REFRESH_MODE = "refresh"
CACHE_CREATE_MODES = {CACHE_MODE_CREATE_OR_REUSE, CACHE_MANAGER_REFRESH_MODE}
CACHE_EXCEPTIONS = (
    CacheConflictError,
    CacheExpiredError,
    CacheNotFoundError,
    CacheValidationError,
)
CONDITION_CFG_SCALE_DEFAULTS = {
    "scale_text": 3.0,
    "scale_speaker": 5.0,
    "scale_caption": 0.0,
}
CONDITION_CFG_WINDOW_DEFAULTS = {
    "min_t": 0.5,
    "max_t": 1.0,
}
AUTO_BUCKET_PRESETS: tuple[tuple[int, int], ...] = (
    (256, 32),
    (512, 64),
    (1024, 128),
    (1536, 192),
    (2048, 256),
)
AUTO_SEQUENCE_LENGTH_MAX = AUTO_BUCKET_PRESETS[-1][0]
AUTO_TEXT_LEN_MAX = AUTO_BUCKET_PRESETS[-1][1]
AUTO_SPEAKER_CONTEXT_LEN_BUCKET = 160
_MISSING = object()


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    checkpoint: str
    reference_wav: Path
    api_model_id: str
    model_device: str
    codec_device: str
    model_precision: str
    codec_precision: str
    codec_repo: str
    default_num_steps: int
    max_num_steps: int
    seconds: float | None
    min_seconds: float
    max_seconds: float
    chars_per_second: float
    seconds_padding: float
    max_ref_seconds: float | None
    preload: bool
    log_timings: bool
    cache_max_memory_bytes: int | None = None
    warmup_buckets: tuple[CoreMLConditionBucket, ...] = ()
    auto_prepare_default_cache: bool = True
    strict_coreml: bool = False
    default_reference_cache_prepare: bool = True
    default_condition_cache_prepare_text: str | None = None
    condition_cache_default_ttl_seconds: float | None = 86400.0
    max_resident_speaker_kv_buckets: int = 3
    max_resident_condition_cache_entries: int = 0
    enable_resident_reference_cache: bool = True
    enable_resident_speaker_kv: bool = True
    enable_condition_packed_kv_cache: bool = False


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    seconds: float


@dataclass(frozen=True)
class SpeechSegmentPlan:
    segments: tuple[SpeechSegment, ...]
    total_seconds: float
    seconds_mode: str


@dataclass(frozen=True)
class AutoSpeechPlanningContext:
    segment_text: str
    normalized_text: str
    seconds: float
    sample_rate: int
    hop_length: int
    latent_patch_size: int
    tokenizer_fingerprint: str
    token_len: int
    token_ids_hash: str
    patched_steps: int


@dataclass(frozen=True)
class AutoBucketResolution:
    bucket: CoreMLConditionBucket | None
    reason: str | None
    attempted: str | None
    planning: AutoSpeechPlanningContext | None = None


class RuntimeState:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._runtime_key: RuntimeKey | None = None
        self._runtime: Any | None = None

    def runtime_key(self) -> RuntimeKey:
        with self._lock:
            if self._runtime_key is None:
                self._runtime_key = RuntimeKey(
                    checkpoint=_resolve_checkpoint_path(self.settings.checkpoint),
                    model_device=self.settings.model_device,
                    codec_repo=self.settings.codec_repo,
                    model_precision=self.settings.model_precision,
                    codec_device=self.settings.codec_device,
                    codec_precision=self.settings.codec_precision,
                    codec_deterministic_encode=True,
                    codec_deterministic_decode=True,
                    enable_watermark=False,
                    compile_model=False,
                    compile_dynamic=False,
                )
            return self._runtime_key

    def get_runtime(self):
        runtime, _ = get_cached_runtime(self.runtime_key())
        with self._lock:
            newly_cached = self._runtime is not runtime
            self._runtime = runtime
        if newly_cached:
            self._apply_runtime_settings(runtime)
        return runtime

    def _apply_runtime_settings(self, runtime: Any) -> None:
        configure = getattr(runtime, "configure_resident_speaker_kv", None)
        if callable(configure):
            configure(
                enabled=self.settings.enable_resident_speaker_kv,
                max_entries=self.settings.max_resident_speaker_kv_buckets,
            )

    @property
    def runtime_if_loaded(self) -> Any | None:
        with self._lock:
            return self._runtime

    @property
    def runtime_loaded(self) -> bool:
        with self._lock:
            return self._runtime is not None


def _prefer_mps_device() -> str:
    devices = list_available_runtime_devices()
    if "mps" in devices:
        return "mps"
    return default_runtime_device()


def _project_relative_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _validate_reference_wav(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Configured reference_wav is not a file: {path}")


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint must be non-empty.")

    candidate = Path(checkpoint).expanduser()
    local_candidates = [candidate]
    if not candidate.is_absolute():
        local_candidates.append(PROJECT_ROOT / candidate)
    for local_path in local_candidates:
        if local_path.is_file():
            return str(local_path)

    suffix = candidate.suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return str(hf_hub_download(repo_id=checkpoint, filename="model.safetensors"))


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


def _cache_error_response(
    exc: Exception,
    *,
    cache_id: str | None = None,
    cache_mode: str | None = None,
) -> JSONResponse:
    error = cache_exception_to_http_error(
        exc,
        cache_id=cache_id,
        cache_mode=cache_mode,
    )
    return JSONResponse(status_code=error.status_code, content=error.payload)


def _strict_auto_unavailable_reason(resolution: _SpeechCacheResolution) -> str:
    parts: list[str] = []
    if resolution.fallback_reason:
        parts.append(f"reason={resolution.fallback_reason}")
    if resolution.auto_status:
        parts.append(f"auto_status={resolution.auto_status}")
    if resolution.bucket is not None:
        parts.append(f"bucket={_bucket_header_value(resolution.bucket)}")
    detail = ", ".join(parts) if parts else "no condition cache available"
    return f"strict_coreml: AUTO request without usable condition cache ({detail})"


def _coreml_backend_error_response(
    exc: CoreMLStatefulUnavailableError,
    *,
    cache_id: str | None = None,
    cache_mode: str | None = None,
) -> JSONResponse:
    error: dict[str, object] = {
        "type": "coreml_backend_unavailable",
        "message": str(exc),
    }
    if cache_id is not None:
        error["cache_id"] = cache_id
    if cache_mode is not None:
        error["cache_mode"] = cache_mode
    return JSONResponse(status_code=503, content={"error": error})


def _cache_require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CacheValidationError("request body must be a JSON object")
    return payload


def _cache_create_mode(payload: dict[str, Any]) -> str:
    value = payload.get("cache_mode", CACHE_MODE_CREATE_OR_REUSE)
    if not isinstance(value, str):
        raise CacheValidationError("cache_mode must be a string")
    cache_mode = value.strip().lower()
    if cache_mode not in CACHE_CREATE_MODES:
        raise CacheValidationError("cache_mode must be create_or_reuse or refresh")
    return cache_mode


CONDITION_CFG_MODES = {"cond", "none", "independent", "joint", "alternating"}


def _cache_optional_text(
    payload: dict[str, Any],
    field: str,
    default: str | None = None,
    *,
    allow_none: bool = False,
) -> str | None:
    value = payload.get(field, _MISSING)
    if value is _MISSING:
        return default
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CacheValidationError(f"{field} must be a non-empty string")
    return value


def _cache_optional_positive_int(
    payload: dict[str, Any],
    field: str,
    default: int | None = None,
) -> int | None:
    value = payload.get(field, _MISSING)
    if value is _MISSING:
        return default
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheValidationError(f"{field} must be a positive int")
    return value


def _cache_optional_non_negative_int(
    payload: dict[str, Any],
    field: str,
    default: int,
) -> int:
    value = payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheValidationError(f"{field} must be a non-negative int")
    return value


def _cache_optional_bool(payload: dict[str, Any], field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise CacheValidationError(f"{field} must be a bool")
    return value


def _cache_optional_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("metadata")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CacheValidationError("metadata must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CacheValidationError("metadata keys must be strings")
    return value


def _cache_ttl_seconds(payload: dict[str, Any]) -> float | None:
    value = payload.get("ttl_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CacheValidationError("ttl_seconds must be positive when supplied")
    if value <= 0:
        raise CacheValidationError("ttl_seconds must be positive when supplied")
    return float(value)


def _cache_required_normalized_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise CacheValidationError(f"{field} must be a string")
    text = value.strip()
    if text == "":
        raise CacheValidationError(f"{field} must be non-empty")
    return text


def _cache_optional_normalized_text(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CacheValidationError(f"{field} must be a string")
    text = value.strip()
    return text or None


def _cache_resolve_condition_caption(payload: dict[str, Any]) -> str | None:
    instructions = _cache_optional_normalized_text(payload, "instructions")
    instruction = _cache_optional_normalized_text(payload, "instruction")
    caption = _cache_optional_normalized_text(payload, "caption")

    if instructions is not None and instruction is not None and instructions != instruction:
        raise CacheValidationError(
            "instructions and instruction must match when both are provided",
        )
    resolved_instruction = instructions if instructions is not None else instruction

    if caption is not None and resolved_instruction is not None and caption != resolved_instruction:
        raise CacheValidationError("caption must match instructions when both are provided")

    return resolved_instruction if resolved_instruction is not None else caption


def _cache_optional_seconds(payload: dict[str, Any]) -> float | None:
    value = payload.get("seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CacheValidationError("seconds must be a number")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise CacheValidationError("seconds must be finite")
    if seconds <= 0:
        raise CacheValidationError("seconds must be positive")
    return seconds


def _cache_delete_cascade(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise CacheValidationError("cascade must be one of: true, false, 1, 0, yes, no")


def _validate_reference_cache_source(payload: dict[str, Any]) -> None:
    source = payload.get("source")
    if source is None:
        return
    if not isinstance(source, dict):
        raise CacheValidationError("source must be an object")
    source_type = source.get("type", "server_default")
    if source_type != "server_default":
        raise CacheValidationError("source.type must be server_default for metadata-only caches")


def _reference_cache_request(
    payload: dict[str, Any],
    settings: ServerSettings,
) -> ReferenceCacheRequest:
    _validate_reference_cache_source(payload)
    return ReferenceCacheRequest(
        model_fingerprint=str(
            _cache_optional_text(
                payload,
                "model_fingerprint",
                f"model:{settings.checkpoint}",
            ),
        ),
        codec_fingerprint=str(
            _cache_optional_text(
                payload,
                "codec_fingerprint",
                f"codec:{settings.codec_repo}",
            ),
        ),
        reference_fingerprint=str(
            _cache_optional_text(
                payload,
                "reference_fingerprint",
                f"server_default:{settings.reference_wav}",
            ),
        ),
        speaker_context_len=int(
            _cache_optional_positive_int(payload, "speaker_context_len", 1),
        ),
        memory_bytes=_cache_optional_non_negative_int(payload, "memory_bytes", 0),
        ttl_seconds=_cache_ttl_seconds(payload),
        metadata=_cache_optional_metadata(payload),
        model=_cache_optional_text(payload, "model", settings.api_model_id, allow_none=True),
        ref_len=_cache_optional_positive_int(payload, "ref_len"),
        speaker_dim=_cache_optional_positive_int(payload, "speaker_dim"),
        memory_bytes_estimated=_cache_optional_bool(
            payload,
            "memory_bytes_estimated",
            True,
        ),
    )


def _condition_cache_bucket(payload: dict[str, Any]) -> CoreMLConditionBucket:
    value = payload.get("bucket")
    if value is None:
        bucket_payload: dict[str, Any] = {}
    elif isinstance(value, dict):
        bucket_payload = value
    else:
        raise CacheValidationError("bucket must be an object")

    speaker_context_len_field = (
        "speaker_context_len_bucket"
        if "speaker_context_len_bucket" in bucket_payload
        else "speaker_context_len"
    )
    try:
        return CoreMLConditionBucket(
            sequence_length=int(
                _cache_optional_positive_int(bucket_payload, "sequence_length", 100),
            ),
            text_len=int(_cache_optional_positive_int(bucket_payload, "text_len", 256)),
            speaker_context_len_bucket=int(
                _cache_optional_positive_int(
                    bucket_payload,
                    speaker_context_len_field,
                    160,
                ),
            ),
        )
    except ValueError as exc:
        raise CacheValidationError(str(exc)) from exc


def _condition_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("cfg", {})
    if value is None:
        cfg_payload: dict[str, Any] = {}
    elif isinstance(value, dict):
        cfg_payload = value
    else:
        raise CacheValidationError("cfg must be an object")

    mode_value = cfg_payload.get("mode", "cond")
    if mode_value is None:
        mode = "cond"
    elif isinstance(mode_value, str):
        mode = mode_value.strip().lower()
    else:
        raise CacheValidationError("cfg.mode must be a string")
    if mode not in CONDITION_CFG_MODES:
        raise CacheValidationError(
            "cfg.mode must be one of: " + ", ".join(sorted(CONDITION_CFG_MODES)),
        )

    canonical_cfg: dict[str, Any] = {"mode": mode}
    for field, default in CONDITION_CFG_SCALE_DEFAULTS.items():
        canonical_cfg[field] = _condition_cfg_float(
            cfg_payload,
            field,
            default,
            min_value=0.0,
        )
    for field, default in CONDITION_CFG_WINDOW_DEFAULTS.items():
        canonical_cfg[field] = _condition_cfg_float(
            cfg_payload,
            field,
            default,
            min_value=0.0,
            max_value=1.0,
        )
    if canonical_cfg["min_t"] > canonical_cfg["max_t"]:
        raise CacheValidationError("cfg.min_t must be <= cfg.max_t")

    if float(canonical_cfg["scale_caption"]) > 0.0:
        raise CacheValidationError(
            "caption CFG is not supported by the CoreML stateful fast path; set cfg.scale_caption=0",
        )

    if mode == "joint":
        enabled_joint_scales = [
            float(canonical_cfg[field])
            for field in ("scale_text", "scale_speaker")
            if float(canonical_cfg[field]) > 0.0
        ]
        if len(enabled_joint_scales) > 1 and (
            max(enabled_joint_scales) - min(enabled_joint_scales) > 1e-6
        ):
            raise CacheValidationError(
                "cfg.mode='joint' requires equal enabled cfg.scale_text/cfg.scale_speaker",
            )

    speaker_kv_scale_raw = cfg_payload.get("speaker_kv_scale")
    if speaker_kv_scale_raw is None:
        canonical_cfg["speaker_kv_scale"] = None
        canonical_cfg["speaker_kv_min_t"] = None
        canonical_cfg["speaker_kv_max_layers"] = None
    else:
        if isinstance(speaker_kv_scale_raw, bool) or not isinstance(
            speaker_kv_scale_raw,
            int | float,
        ):
            raise CacheValidationError("cfg.speaker_kv_scale must be a number")
        scale = float(speaker_kv_scale_raw)
        if not math.isfinite(scale) or scale <= 0.0:
            raise CacheValidationError("cfg.speaker_kv_scale must be > 0")
        canonical_cfg["speaker_kv_scale"] = scale

        min_t_raw = cfg_payload.get("speaker_kv_min_t", 0.9)
        if min_t_raw is None:
            min_t = 0.9
        elif isinstance(min_t_raw, bool) or not isinstance(min_t_raw, int | float):
            raise CacheValidationError("cfg.speaker_kv_min_t must be a number")
        else:
            min_t = float(min_t_raw)
        if not math.isfinite(min_t) or not (0.0 <= min_t <= 1.0):
            raise CacheValidationError("cfg.speaker_kv_min_t must be in [0, 1]")
        canonical_cfg["speaker_kv_min_t"] = min_t

        max_layers_raw = cfg_payload.get("speaker_kv_max_layers")
        if max_layers_raw is None:
            canonical_cfg["speaker_kv_max_layers"] = None
        elif isinstance(max_layers_raw, bool) or not isinstance(max_layers_raw, int):
            raise CacheValidationError("cfg.speaker_kv_max_layers must be a non-negative int")
        elif int(max_layers_raw) < 0:
            raise CacheValidationError("cfg.speaker_kv_max_layers must be a non-negative int")
        else:
            canonical_cfg["speaker_kv_max_layers"] = int(max_layers_raw)
    return canonical_cfg


def _condition_cfg_float(
    cfg_payload: dict[str, Any],
    field: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = cfg_payload.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CacheValidationError(f"cfg.{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CacheValidationError(f"cfg.{field} must be finite")
    if min_value is not None and parsed < min_value:
        raise CacheValidationError(f"cfg.{field} must be >= {min_value}")
    if max_value is not None and parsed > max_value:
        raise CacheValidationError(f"cfg.{field} must be <= {max_value}")
    return parsed


def _condition_branch_layouts(
    payload: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[str, ...] | list[str]:
    canonical = _canonical_condition_branch_layouts(cfg)
    value = payload.get("branch_layouts", _MISSING)
    if value is not _MISSING:
        if not isinstance(value, list | tuple):
            raise CacheValidationError("branch_layouts must be a list")
        explicit = tuple(value)
        if explicit != canonical:
            raise CacheValidationError("branch_layouts conflict with cfg")
        return explicit

    return canonical


def _canonical_condition_branch_layouts(cfg: dict[str, Any]) -> tuple[str, ...]:
    mode = cfg["mode"]
    if mode in {"cond", "none"}:
        return (BRANCH_LAYOUT_COND1,)
    has_text = float(cfg["scale_text"]) > 0.0
    has_speaker = float(cfg["scale_speaker"]) > 0.0
    has_caption = float(cfg["scale_caption"]) > 0.0
    if not (has_text or has_speaker or has_caption):
        return (BRANCH_LAYOUT_COND1,)
    if mode == "independent":
        return (BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3)
    if mode == "joint":
        return (BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_JOINT2)
    if mode == "alternating":
        layouts: list[str] = [BRANCH_LAYOUT_COND1]
        if has_text:
            layouts.append(BRANCH_LAYOUT_ALTERNATING_TEXT2)
        if has_speaker:
            layouts.append(BRANCH_LAYOUT_ALTERNATING_SPEAKER2)
        return tuple(layouts)
    raise CacheValidationError(
        "cfg.mode must be one of: " + ", ".join(sorted(CONDITION_CFG_MODES)),
    )


def _condition_fingerprint(
    payload: dict[str, Any],
    bucket: CoreMLConditionBucket,
    cfg: dict[str, Any],
    input_text: str,
    caption: str | None,
    seconds: float | None,
) -> str:
    supplied = _cache_optional_text(
        payload,
        "condition_fingerprint",
        None,
        allow_none=True,
    )

    fingerprint_payload = {
        "input": input_text,
        "caption": caption,
        "seconds": seconds,
        "bucket": {
            "sequence_length": bucket.sequence_length,
            "text_len": bucket.text_len,
            "speaker_context_len": bucket.speaker_context_len_bucket,
        },
        "cfg": cfg,
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical = f"condition:{hashlib.sha256(encoded).hexdigest()}"
    if supplied is not None and supplied != canonical:
        raise CacheValidationError("condition_fingerprint does not match canonical request")
    return canonical


def _internal_auto_condition_fingerprint(
    *,
    normalized_text: str,
    token_ids_hash: str,
    tokenizer_fingerprint: str,
    model_fingerprint: str,
    reference_fingerprint: str,
    caption: str | None,
    bucket: CoreMLConditionBucket,
    cfg: dict[str, Any],
    branch_layouts: tuple[str, ...] | list[str],
    speaker_context_len: int,
    state_copies: int,
) -> str:
    fingerprint_payload = {
        "kind": "condition_auto_v2",
        "normalized_text": normalized_text,
        "token_ids_hash": token_ids_hash,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_fingerprint": model_fingerprint,
        "reference_fingerprint": reference_fingerprint,
        "caption": caption,
        "bucket": {
            "sequence_length": int(bucket.sequence_length),
            "text_len": int(bucket.text_len),
            "speaker_context_len": int(bucket.speaker_context_len_bucket),
        },
        "speaker_context_len": int(speaker_context_len),
        "branch_layouts": list(branch_layouts),
        "state_copies": int(state_copies),
        "cfg": cfg,
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"condition:auto-v2:{hashlib.sha256(encoded).hexdigest()}"


def _condition_cache_request(
    payload: dict[str, Any],
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
) -> ConditionCacheRequest:
    reference_cache_id = _cache_optional_text(payload, "reference_cache_id")
    if reference_cache_id is None:
        raise CacheValidationError("reference_cache_id must be a non-empty string")

    input_text = _cache_required_normalized_text(payload, "input")
    caption = _cache_resolve_condition_caption(payload)
    seconds = _cache_optional_seconds(payload)
    bucket = _condition_cache_bucket(payload)
    cfg = _condition_cfg(payload)
    speaker_context_len = _cache_optional_positive_int(payload, "speaker_context_len")
    if speaker_context_len is None:
        speaker_context_len = cache_manager.get_reference_cache(
            reference_cache_id
        ).speaker_context_len

    return ConditionCacheRequest(
        reference_cache_id=reference_cache_id,
        model_fingerprint=str(
            _cache_optional_text(
                payload,
                "model_fingerprint",
                f"model:{settings.checkpoint}",
            ),
        ),
        tokenizer_fingerprint=str(
            _cache_optional_text(
                payload,
                "tokenizer_fingerprint",
                "tokenizer:default",
            ),
        ),
        condition_fingerprint=_condition_fingerprint(
            payload,
            bucket,
            cfg,
            input_text,
            caption,
            seconds,
        ),
        bucket=bucket,
        speaker_context_len=int(speaker_context_len),
        branch_layouts=_condition_branch_layouts(payload, cfg),
        ttl_seconds=_cache_ttl_seconds(payload),
        metadata=_cache_optional_metadata(payload),
        state_copies=_state_copies_for_cfg(cfg),
    )


def _speech_irodori_extension(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("irodori", _MISSING)
    if value is _MISSING:
        return None
    if not isinstance(value, dict):
        raise CacheValidationError("irodori must be an object")
    return value


def _speech_cache_mode(irodori: dict[str, Any] | None) -> str:
    if irodori is None:
        return CACHE_MODE_AUTO
    return normalize_speech_cache_mode(irodori.get("cache_mode"), default=CACHE_MODE_AUTO)


def _speech_cache_id(irodori: dict[str, Any] | None) -> str | None:
    if irodori is None:
        return None
    value = irodori.get("cache_id", _MISSING)
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CacheValidationError("irodori.cache_id must be a non-empty string")
    return value.strip()


def _bucket_header_value(bucket: CoreMLConditionBucket) -> str:
    return (
        f"S{int(bucket.sequence_length)}_T{int(bucket.text_len)}_"
        f"R{int(bucket.speaker_context_len_bucket)}"
    )


def _auto_bucket_attempted(
    sequence_length: int | None,
    text_len: int | None,
) -> str:
    s_value = str(sequence_length) if sequence_length is not None else f">{AUTO_SEQUENCE_LENGTH_MAX}"
    t_value = str(text_len) if text_len is not None else f">{AUTO_TEXT_LEN_MAX}"
    return f"S{s_value}_T{t_value}_R{AUTO_SPEAKER_CONTEXT_LEN_BUCKET}"


def _select_auto_preset(
    patched_steps: int,
    token_len: int,
) -> tuple[int, int] | None:
    for sequence_length, text_len in AUTO_BUCKET_PRESETS:
        if int(patched_steps) <= int(sequence_length) and int(token_len) <= int(text_len):
            return int(sequence_length), int(text_len)
    return None


def _auto_select_bucket(planning: AutoSpeechPlanningContext) -> AutoBucketResolution:
    preset = _select_auto_preset(
        int(planning.patched_steps),
        int(planning.token_len),
    )
    if preset is None:
        oversize_s = int(planning.patched_steps) > AUTO_SEQUENCE_LENGTH_MAX
        oversize_t = int(planning.token_len) > AUTO_TEXT_LEN_MAX
        sequence_length: int | None = None if oversize_s else AUTO_SEQUENCE_LENGTH_MAX
        text_len: int | None = None if oversize_t else AUTO_TEXT_LEN_MAX
        reason = "oversize_s" if oversize_s else "oversize_t"
        return AutoBucketResolution(
            bucket=None,
            reason=reason,
            attempted=_auto_bucket_attempted(sequence_length, text_len),
            planning=planning,
        )
    sequence_length, text_len = preset
    return AutoBucketResolution(
        bucket=CoreMLConditionBucket(
            sequence_length=sequence_length,
            text_len=text_len,
            speaker_context_len_bucket=AUTO_SPEAKER_CONTEXT_LEN_BUCKET,
        ),
        reason=None,
        attempted=None,
        planning=planning,
    )


def _runtime_tokenizer_fingerprint(runtime: Any) -> str:
    explicit = getattr(runtime, "tokenizer_fingerprint", None)
    if callable(explicit):
        return str(explicit())
    if explicit is not None:
        return str(explicit)

    model_cfg = getattr(runtime, "model_cfg", None)
    tokenizer = getattr(runtime, "tokenizer", None)
    payload = {
        "repo": getattr(model_cfg, "text_tokenizer_repo", "unknown"),
        "add_bos": getattr(model_cfg, "text_add_bos", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"tokenizer:{hashlib.sha256(encoded).hexdigest()}"


def _build_auto_speech_planning_context(
    segment: SpeechSegment,
    runtime: Any,
) -> AutoSpeechPlanningContext:
    normalized_text = normalize_text(segment.text).strip()
    if normalized_text == "":
        raise ValueError("text became empty after normalization.")

    token_len, token_ids_hash = runtime.tokenize_for_bucket(normalized_text)
    return AutoSpeechPlanningContext(
        segment_text=segment.text,
        normalized_text=normalized_text,
        seconds=float(segment.seconds),
        sample_rate=int(runtime.codec.sample_rate),
        hop_length=int(runtime.codec.model.hop_length),
        latent_patch_size=int(runtime.model_cfg.latent_patch_size),
        tokenizer_fingerprint=_runtime_tokenizer_fingerprint(runtime),
        token_len=int(token_len),
        token_ids_hash=str(token_ids_hash),
        patched_steps=int(runtime.estimate_patched_steps(float(segment.seconds))),
    )


def _resolve_auto_bucket_resolution(
    irodori: dict[str, Any] | None,
    segment: SpeechSegment,
    runtime: Any,
) -> AutoBucketResolution:
    if irodori is not None and "bucket" in irodori:
        return AutoBucketResolution(
            bucket=_condition_cache_bucket(irodori),
            reason=None,
            attempted=None,
        )
    return _auto_select_bucket(_build_auto_speech_planning_context(segment, runtime))


def _resolve_auto_bucket_resolutions(
    irodori: dict[str, Any] | None,
    segments: tuple[SpeechSegment, ...],
    runtime: Any,
) -> list[AutoBucketResolution]:
    return [_resolve_auto_bucket_resolution(irodori, segment, runtime) for segment in segments]


def _speech_reference_cache_request(settings: ServerSettings) -> ReferenceCacheRequest:
    return _reference_cache_request(
        {"source": {"type": "server_default"}},
        settings,
    )


def _server_reference_fingerprint(settings: ServerSettings) -> str:
    digest = hashlib.sha256()
    with settings.reference_wav.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    max_ref_seconds = (
        "none" if settings.max_ref_seconds is None else f"{float(settings.max_ref_seconds):.6g}"
    )
    return (
        f"server_default:{settings.reference_wav}:"
        f"sha256:{digest.hexdigest()}:max_ref_seconds:{max_ref_seconds}"
    )


def _resident_reference_prepare_key(settings: ServerSettings) -> str:
    encoded = _server_reference_fingerprint(settings).encode("utf-8")
    return f"resident_default_{hashlib.sha256(encoded).hexdigest()[:32]}"


def _resident_reference_cache_request(
    settings: ServerSettings,
    resident: ResidentReferenceTensors,
) -> ReferenceCacheRequest:
    return ReferenceCacheRequest(
        model_fingerprint=f"model:{settings.checkpoint}",
        codec_fingerprint=f"codec:{settings.codec_repo}",
        reference_fingerprint=resident.reference_fingerprint
        or _server_reference_fingerprint(settings),
        speaker_context_len=int(resident.speaker_context_len),
        memory_bytes=int(resident.memory_bytes),
        ttl_seconds=None,
        metadata={
            "source": {"type": "server_default", "path": str(settings.reference_wav)},
            "resident": True,
            "device": resident.device,
            "dtype": resident.dtype,
        },
        model=settings.api_model_id,
        ref_len=int(resident.ref_len),
        speaker_dim=int(resident.speaker_dim),
        memory_bytes_estimated=False,
        resident_layers=("ref_latent", "ref_mask", "speaker_state", "speaker_mask"),
    )


@dataclass(frozen=True)
class _ResidentReferencePlan:
    provisional_id: str
    resident: ResidentReferenceTensors
    request: ReferenceCacheRequest


def _prepare_runtime_default_reference_plan(
    runtime: Any,
    settings: ServerSettings,
    *,
    force_refresh: bool = False,
) -> _ResidentReferencePlan:
    prepare = getattr(runtime, "prepare_default_reference_tensors", None)
    if not callable(prepare):
        raise RuntimeError("runtime does not expose prepare_default_reference_tensors")

    reference_fingerprint = _server_reference_fingerprint(settings)
    provisional_id = _resident_reference_prepare_key(settings)
    resident = prepare(
        settings.reference_wav,
        settings.max_ref_seconds,
        reference_cache_id=provisional_id,
        reference_fingerprint=reference_fingerprint,
        force_refresh=force_refresh,
    )
    request = _resident_reference_cache_request(settings, resident)
    return _ResidentReferencePlan(
        provisional_id=provisional_id,
        resident=resident,
        request=request,
    )


def _delete_runtime_resident_reference(runtime: Any, reference_cache_id: str) -> None:
    delete_resident = getattr(runtime, "delete_resident_reference_tensors", None)
    if callable(delete_resident):
        delete_resident(reference_cache_id)


def _materialize_runtime_default_reference_resident(
    runtime: Any,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
    plan: _ResidentReferencePlan,
    cache_mode: str = CACHE_MODE_CREATE_OR_REUSE,
    force_refresh: bool = False,
) -> CacheCreateResult:
    final_id = cache_manager.reference_cache_id_for_request(plan.request)
    final_resident_prepared = False
    try:
        prepare = getattr(runtime, "prepare_default_reference_tensors", None)
        if not callable(prepare):
            raise RuntimeError("runtime does not expose prepare_default_reference_tensors")
        final_resident = prepare(
            settings.reference_wav,
            settings.max_ref_seconds,
            reference_cache_id=final_id,
            reference_fingerprint=plan.resident.reference_fingerprint
            or _server_reference_fingerprint(settings),
            force_refresh=force_refresh,
        )
        final_resident_prepared = True
        final_request = _resident_reference_cache_request(settings, final_resident)
        final_request_id = cache_manager.reference_cache_id_for_request(final_request)
        if final_request_id != final_id:
            _delete_runtime_resident_reference(runtime, final_id)
            final_resident_prepared = False
            raise CacheConflictError(
                "resident reference metadata changed during materialization",
            )
        commit_mode = cache_mode
        if cache_mode == CACHE_MODE_CREATE_OR_REUSE:
            try:
                existing_handle = cache_manager.peek_reference_cache(final_id)
            except (CacheExpiredError, CacheNotFoundError):
                pass
            else:
                if not _is_resident_reference_handle(existing_handle):
                    commit_mode = CACHE_MANAGER_REFRESH_MODE
        return cache_manager.prepare_reference_cache(final_request, cache_mode=commit_mode)
    except Exception:
        if final_resident_prepared:
            _delete_runtime_resident_reference(runtime, final_id)
        raise
    finally:
        if plan.provisional_id != final_id:
            _delete_runtime_resident_reference(runtime, plan.provisional_id)


def _prepare_runtime_default_reference_resident(
    runtime: Any,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
    cache_mode: str = CACHE_MODE_CREATE_OR_REUSE,
) -> CacheCreateResult:
    force_refresh = cache_mode == CACHE_MANAGER_REFRESH_MODE
    plan = _prepare_runtime_default_reference_plan(
        runtime,
        settings,
        force_refresh=force_refresh,
    )
    return _materialize_runtime_default_reference_resident(
        runtime,
        settings,
        cache_manager,
        plan,
        cache_mode=cache_mode,
        force_refresh=force_refresh,
    )


def _raise_speech_reference_conflict(
    field_name: str,
    actual: object,
    expected: object,
) -> None:
    if actual != expected:
        raise CacheConflictError(f"{field_name} conflicts with server reference cache")


def _validate_speech_reference_cache(
    reference_handle: Any,
    expected_request: ReferenceCacheRequest,
    *,
    validate_speaker_context_len: bool = True,
) -> None:
    _raise_speech_reference_conflict(
        "model_fingerprint",
        reference_handle.model_fingerprint,
        expected_request.model_fingerprint,
    )
    _raise_speech_reference_conflict(
        "codec_fingerprint",
        reference_handle.codec_fingerprint,
        expected_request.codec_fingerprint,
    )
    _raise_speech_reference_conflict(
        "reference_fingerprint",
        reference_handle.reference_fingerprint,
        expected_request.reference_fingerprint,
    )
    if validate_speaker_context_len:
        _raise_speech_reference_conflict(
            "speaker_context_len",
            reference_handle.speaker_context_len,
            expected_request.speaker_context_len,
        )


def _is_resident_reference_handle(reference_handle: Any) -> bool:
    metadata = getattr(reference_handle, "metadata", {}) or {}
    return not bool(getattr(reference_handle, "memory_bytes_estimated", True)) and bool(
        metadata.get("resident")
    )


def _expected_speech_reference_cache_request_for_handle(
    reference_handle: Any,
    settings: ServerSettings,
) -> ReferenceCacheRequest:
    if not _is_resident_reference_handle(reference_handle):
        return _speech_reference_cache_request(settings)
    return ReferenceCacheRequest(
        model_fingerprint=f"model:{settings.checkpoint}",
        codec_fingerprint=f"codec:{settings.codec_repo}",
        reference_fingerprint=_server_reference_fingerprint(settings),
        speaker_context_len=int(reference_handle.speaker_context_len),
    )


def _speech_condition_cache_request(
    irodori: dict[str, Any],
    settings: ServerSettings,
    reference_cache_id: str,
    speaker_context_len: int,
    segment: SpeechSegment,
    caption: str | None,
) -> ConditionCacheRequest:
    condition_payload = dict(irodori)
    if "cfg" not in condition_payload:
        condition_payload["cfg"] = {"mode": "independent"}

    bucket = _condition_cache_bucket(condition_payload)
    cfg = _condition_cfg(condition_payload)
    return ConditionCacheRequest(
        reference_cache_id=reference_cache_id,
        model_fingerprint=str(
            _cache_optional_text(
                condition_payload,
                "model_fingerprint",
                f"model:{settings.checkpoint}",
            ),
        ),
        tokenizer_fingerprint=str(
            _cache_optional_text(
                condition_payload,
                "tokenizer_fingerprint",
                "tokenizer:default",
            ),
        ),
        condition_fingerprint=_condition_fingerprint(
            condition_payload,
            bucket,
            cfg,
            segment.text,
            caption,
            float(segment.seconds),
        ),
        bucket=bucket,
        speaker_context_len=int(speaker_context_len),
        branch_layouts=_condition_branch_layouts(condition_payload, cfg),
        state_copies=_state_copies_for_cfg(cfg),
    )


def _speech_fast_path_cfg(irodori: dict[str, Any] | None) -> dict[str, Any]:
    condition_payload = dict(irodori or {})
    if "cfg" not in condition_payload:
        condition_payload["cfg"] = {"mode": "independent"}
    cfg = _condition_cfg(condition_payload)
    mode = cfg["mode"]
    guidance_mode = "independent" if mode in {"cond", "none"} else mode
    scale_text = 0.0 if mode in {"cond", "none"} else float(cfg["scale_text"])
    scale_speaker = 0.0 if mode in {"cond", "none"} else float(cfg["scale_speaker"])
    scale_caption = 0.0 if mode in {"cond", "none"} else float(cfg["scale_caption"])
    return {
        "guidance_mode": guidance_mode,
        "scale_text": scale_text,
        "scale_caption": scale_caption,
        "scale_speaker": scale_speaker,
        "min_t": float(cfg["min_t"]),
        "max_t": float(cfg["max_t"]),
        "speaker_kv_scale": cfg.get("speaker_kv_scale"),
        "speaker_kv_min_t": cfg.get("speaker_kv_min_t"),
        "speaker_kv_max_layers": cfg.get("speaker_kv_max_layers"),
    }


def _state_copies_for_cfg(cfg: dict[str, Any]) -> int:
    return 2 if cfg.get("speaker_kv_scale") is not None else 1


@dataclass(frozen=True)
class _SpeechCacheResolution:
    condition_handle: ConditionCacheHandle | None
    reference_cache_id: str | None
    reference_created: bool
    condition_created: bool
    bucket: CoreMLConditionBucket | None = None
    auto_status: str | None = None
    fallback_reason: str | None = None


def _condition_bucket_from_handle(handle: ConditionCacheHandle) -> CoreMLConditionBucket:
    return CoreMLConditionBucket(
        sequence_length=int(handle.sequence_length),
        text_len=int(handle.text_len),
        speaker_context_len_bucket=int(handle.speaker_context_len_bucket),
    )


def _resolve_speech_cache(
    irodori: dict[str, Any] | None,
    cache_mode: str,
    cache_id: str | None,
    segment_plan: SpeechSegmentPlan,
    caption: str | None,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
    *,
    runtime: Any | None = None,
    resident_reference_plan: _ResidentReferencePlan | None = None,
) -> list[_SpeechCacheResolution]:
    if cache_mode == CACHE_MODE_OFF:
        return [
            _SpeechCacheResolution(None, None, False, False, auto_status="off")
            for _segment in segment_plan.segments
        ]

    if cache_mode in {CACHE_MODE_PREPARE, CACHE_MODE_REFRESH}:
        if len(segment_plan.segments) != 1:
            raise CacheConflictError(
                "cache_mode=prepare/refresh does not support multi-segment speech",
            )
        return [
            _prepare_or_refresh_speech_cache(
                irodori=irodori,
                cache_mode=cache_mode,
                cache_id=cache_id,
                segment=segment_plan.segments[0],
                caption=caption,
                settings=settings,
                cache_manager=cache_manager,
                runtime=runtime,
                resident_reference_plan=resident_reference_plan,
            )
        ]

    if cache_mode == CACHE_MODE_REQUIRE and cache_id is None:
        raise CacheValidationError("irodori.cache_id is required when cache_mode=require")
    if cache_id is None:
        return [
            _SpeechCacheResolution(
                None,
                None,
                False,
                False,
                auto_status="miss-fallback",
            )
            for _segment in segment_plan.segments
        ]
    if len(segment_plan.segments) != 1:
        raise CacheConflictError("single cache_id cannot be used with multi-segment speech")

    try:
        condition_handle = cache_manager.peek_condition_cache(cache_id)
    except (CacheExpiredError, CacheNotFoundError):
        if cache_mode == CACHE_MODE_AUTO:
            return [
                _SpeechCacheResolution(
                    None,
                    None,
                    False,
                    False,
                    auto_status="miss-fallback",
                )
            ]
        raise

    try:
        reference_handle = cache_manager.peek_reference_cache(
            condition_handle.reference_cache_id,
        )
        expected_reference_request = _expected_speech_reference_cache_request_for_handle(
            reference_handle,
            settings,
        )
        _validate_speech_reference_cache(
            reference_handle,
            expected_reference_request,
        )
        expected_request = _speech_condition_cache_request(
            irodori or {},
            settings,
            reference_handle.id,
            reference_handle.speaker_context_len,
            segment_plan.segments[0],
            caption,
        )
        validated = cache_manager.require_condition_cache(cache_id, expected_request)
    except (CacheExpiredError, CacheNotFoundError):
        if cache_mode == CACHE_MODE_AUTO:
            return [
                _SpeechCacheResolution(
                    None,
                    None,
                    False,
                    False,
                    auto_status="miss-fallback",
                )
            ]
        raise
    return [
        _SpeechCacheResolution(
            condition_handle=validated,
            reference_cache_id=validated.reference_cache_id,
            reference_created=False,
            condition_created=False,
            bucket=_condition_bucket_from_handle(validated),
            auto_status="hit",
        )
    ]


def _prepare_or_refresh_speech_cache(
    *,
    irodori: dict[str, Any] | None,
    cache_mode: str,
    cache_id: str | None,
    segment: SpeechSegment,
    caption: str | None,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
    runtime: Any | None = None,
    resident_reference_plan: _ResidentReferencePlan | None = None,
) -> _SpeechCacheResolution:
    reference_request = (
        resident_reference_plan.request
        if resident_reference_plan is not None
        else _speech_reference_cache_request(settings)
    )
    expected_reference_cache_id = cache_manager.reference_cache_id_for_request(reference_request)
    condition_request = _speech_condition_cache_request(
        irodori or {},
        settings,
        expected_reference_cache_id,
        reference_request.speaker_context_len,
        segment,
        caption,
    )
    expected_condition_cache_id = cache_manager.condition_cache_id_for_request(
        condition_request,
    )
    if cache_id is not None and cache_id != expected_condition_cache_id:
        if runtime is not None and resident_reference_plan is not None:
            _delete_runtime_resident_reference(runtime, resident_reference_plan.provisional_id)
        raise CacheConflictError(
            "cache_id does not match prepared condition cache for this request",
        )

    reference_cache_mode = (
        CACHE_MANAGER_REFRESH_MODE
        if cache_mode == CACHE_MODE_REFRESH
        else CACHE_MODE_CREATE_OR_REUSE
    )
    force_resident_refresh = reference_cache_mode == CACHE_MANAGER_REFRESH_MODE
    if resident_reference_plan is not None:
        if runtime is None:
            raise RuntimeError("runtime is required to materialize resident reference metadata")
        reference_result = _materialize_runtime_default_reference_resident(
            runtime,
            settings,
            cache_manager,
            resident_reference_plan,
            cache_mode=reference_cache_mode,
            force_refresh=force_resident_refresh,
        )
    else:
        reference_result = cache_manager.prepare_reference_cache(
            reference_request,
            cache_mode=reference_cache_mode,
        )
    try:
        condition_result = cache_manager.prepare_condition_cache(
            condition_request,
            cache_mode=reference_cache_mode,
        )
    except CACHE_EXCEPTIONS:
        if not reference_result.reused:
            cache_manager.delete_reference_cache(reference_result.handle.id, cascade=True)
        raise
    return _SpeechCacheResolution(
        condition_handle=condition_result.handle,
        reference_cache_id=reference_result.handle.id,
        reference_created=not reference_result.reused,
        condition_created=not condition_result.reused,
        bucket=_condition_bucket_from_handle(condition_result.handle),
        auto_status="reused" if condition_result.reused else "prepared",
    )


def _auto_planning_for_segment(
    bucket_resolution: AutoBucketResolution,
    segment: SpeechSegment,
    runtime: Any,
) -> AutoSpeechPlanningContext:
    if bucket_resolution.planning is not None:
        return bucket_resolution.planning
    return _build_auto_speech_planning_context(segment, runtime)


def _auto_internal_condition_cache_request(
    *,
    irodori: dict[str, Any] | None,
    settings: ServerSettings,
    runtime: Any,
    reference_handle: Any,
    bucket: CoreMLConditionBucket,
    planning: AutoSpeechPlanningContext,
    caption: str | None,
) -> ConditionCacheRequest:
    condition_payload: dict[str, Any] = dict(irodori or {})
    if "cfg" not in condition_payload:
        condition_payload["cfg"] = {"mode": "independent"}
    cfg = _condition_cfg(condition_payload)
    branch_layouts = _canonical_condition_branch_layouts(cfg)
    state_copies = _state_copies_for_cfg(cfg)
    tokenizer_fingerprint = _runtime_tokenizer_fingerprint(runtime)
    model_fingerprint = f"model:{settings.checkpoint}"
    reference_fingerprint = _server_reference_fingerprint(settings)
    speaker_context_len = int(reference_handle.speaker_context_len)

    condition_fingerprint = _internal_auto_condition_fingerprint(
        normalized_text=planning.normalized_text,
        token_ids_hash=planning.token_ids_hash,
        tokenizer_fingerprint=tokenizer_fingerprint,
        model_fingerprint=model_fingerprint,
        reference_fingerprint=reference_fingerprint,
        caption=caption,
        bucket=bucket,
        cfg=cfg,
        branch_layouts=branch_layouts,
        speaker_context_len=speaker_context_len,
        state_copies=state_copies,
    )
    return ConditionCacheRequest(
        reference_cache_id=reference_handle.id,
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        condition_fingerprint=condition_fingerprint,
        bucket=bucket,
        speaker_context_len=speaker_context_len,
        branch_layouts=branch_layouts,
        state_copies=state_copies,
        ttl_seconds=settings.condition_cache_default_ttl_seconds,
    )


def _auto_prepare_speech_caches(
    *,
    irodori: dict[str, Any] | None,
    runtime: Any,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
    segments: tuple[SpeechSegment, ...],
    caption: str | None,
    bucket_resolutions: list[AutoBucketResolution],
) -> list[_SpeechCacheResolution]:
    if not settings.auto_prepare_default_cache:
        return [
            _SpeechCacheResolution(
                condition_handle=None,
                reference_cache_id=None,
                reference_created=False,
                condition_created=False,
                bucket=bucket_resolution.bucket,
                auto_status="miss-fallback",
                fallback_reason=(
                    bucket_resolution.reason
                    if bucket_resolution.bucket is None
                    else "auto-prepare-disabled"
                ),
            )
            for bucket_resolution in bucket_resolutions
        ]
    if not settings.enable_resident_reference_cache:
        return [
            _SpeechCacheResolution(
                condition_handle=None,
                reference_cache_id=None,
                reference_created=False,
                condition_created=False,
                bucket=bucket_resolution.bucket,
                auto_status="miss-fallback",
                fallback_reason=(
                    bucket_resolution.reason
                    if bucket_resolution.bucket is None
                    else "resident_reference_disabled"
                ),
            )
            for bucket_resolution in bucket_resolutions
        ]

    has_workable_bucket = any(
        bucket_resolution.bucket is not None for bucket_resolution in bucket_resolutions
    )
    reference_handle: Any | None = None
    reference_failure_reason: str | None = None
    if has_workable_bucket:
        try:
            resident_plan = _prepare_runtime_default_reference_plan(runtime, settings)
            reference_result = _materialize_runtime_default_reference_resident(
                runtime,
                settings,
                cache_manager,
                resident_plan,
                cache_mode=CACHE_MODE_CREATE_OR_REUSE,
            )
            reference_handle = reference_result.handle
        except CACHE_EXCEPTIONS:
            if settings.strict_coreml:
                raise
            reference_failure_reason = "reference-prepare-failed"
        except CoreMLStatefulUnavailableError:
            if settings.strict_coreml:
                raise
            reference_failure_reason = "coreml-stateful-unavailable"
        except RuntimeError:
            if settings.strict_coreml:
                raise
            reference_failure_reason = "reference-prepare-failed"

    resolutions: list[_SpeechCacheResolution] = []
    for segment, bucket_resolution in zip(segments, bucket_resolutions, strict=True):
        if bucket_resolution.bucket is None:
            resolutions.append(
                _SpeechCacheResolution(
                    condition_handle=None,
                    reference_cache_id=None,
                    reference_created=False,
                    condition_created=False,
                    bucket=None,
                    auto_status="miss-fallback",
                    fallback_reason=bucket_resolution.reason,
                ),
            )
            continue
        if reference_handle is None:
            resolutions.append(
                _SpeechCacheResolution(
                    condition_handle=None,
                    reference_cache_id=None,
                    reference_created=False,
                    condition_created=False,
                    bucket=bucket_resolution.bucket,
                    auto_status="miss-fallback",
                    fallback_reason=reference_failure_reason
                    or "reference-prepare-failed",
                ),
            )
            continue
        try:
            planning = _auto_planning_for_segment(bucket_resolution, segment, runtime)
            condition_request = _auto_internal_condition_cache_request(
                irodori=irodori,
                settings=settings,
                runtime=runtime,
                reference_handle=reference_handle,
                bucket=bucket_resolution.bucket,
                planning=planning,
                caption=caption,
            )
            condition_result = cache_manager.prepare_condition_cache(
                condition_request,
                cache_mode=CACHE_MODE_CREATE_OR_REUSE,
            )
        except CACHE_EXCEPTIONS:
            if settings.strict_coreml:
                raise
            resolutions.append(
                _SpeechCacheResolution(
                    condition_handle=None,
                    reference_cache_id=None,
                    reference_created=False,
                    condition_created=False,
                    bucket=bucket_resolution.bucket,
                    auto_status="miss-fallback",
                    fallback_reason="condition-prepare-failed",
                ),
            )
            continue
        except CoreMLStatefulUnavailableError:
            if settings.strict_coreml:
                raise
            resolutions.append(
                _SpeechCacheResolution(
                    condition_handle=None,
                    reference_cache_id=None,
                    reference_created=False,
                    condition_created=False,
                    bucket=bucket_resolution.bucket,
                    auto_status="miss-fallback",
                    fallback_reason="coreml-stateful-unavailable",
                ),
            )
            continue
        resolutions.append(
            _SpeechCacheResolution(
                condition_handle=condition_result.handle,
                reference_cache_id=reference_handle.id,
                reference_created=False,
                condition_created=not condition_result.reused,
                bucket=bucket_resolution.bucket,
                auto_status="reused" if condition_result.reused else "prepared",
            ),
        )
    return resolutions


def _aggregate_backend_header(segment_backends: list[str]) -> str:
    if not segment_backends:
        return "pytorch"
    unique_backends = set(segment_backends)
    if unique_backends == {"coreml-stateful"}:
        return "coreml-stateful"
    if unique_backends == {"pytorch"}:
        return "pytorch"
    return "mixed"


def _summarize_cache_id_header(cache_ids: list[str]) -> str | None:
    if not cache_ids:
        return None
    if len(cache_ids) >= 6:
        return f"{cache_ids[0]},+{len(cache_ids) - 1} more"
    return ",".join(cache_ids)


def _summarize_header_values(values: list[str | None]) -> str | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(set(present)) == 1:
        return present[0]
    return "mixed"


def _bucket_headers_for_resolutions(resolutions: list[_SpeechCacheResolution]) -> list[str]:
    return [
        _bucket_header_value(resolution.bucket)
        for resolution in resolutions
        if resolution.bucket is not None
    ]


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a string.")
    text = value.strip()
    if text == "":
        raise HTTPException(status_code=400, detail=f"'{field}' must be non-empty.")
    return text


def _optional_text(payload: dict[str, Any], field: str, default: str | None = None) -> str | None:
    value = payload.get(field, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a string.")
    return value


def _resolve_caption(payload: dict[str, Any]) -> str | None:
    """Resolve the style/control caption for SamplingRequest.

    Accepts the OpenAI-compatible ``instructions`` field, the singular alias
    ``instruction``, and the original Irodori ``caption`` extension. Whitespace-only
    values are treated as unspecified. When multiple non-empty values are provided
    they must be equal; otherwise HTTP 400 is raised.
    """

    def _normalized(field: str) -> str | None:
        raw = _optional_text(payload, field, None)
        if raw is None:
            return None
        stripped = raw.strip()
        return stripped or None

    instructions = _normalized("instructions")
    instruction = _normalized("instruction")
    caption = _normalized("caption")

    if instructions is not None and instruction is not None and instructions != instruction:
        raise HTTPException(
            status_code=400,
            detail="'instructions' and 'instruction' must match when both are provided.",
        )
    resolved_instruction = instructions if instructions is not None else instruction

    if caption is not None and resolved_instruction is not None and caption != resolved_instruction:
        raise HTTPException(
            status_code=400,
            detail="'caption' must match 'instructions' when both are provided.",
        )

    return resolved_instruction if resolved_instruction is not None else caption


def _optional_int(
    payload: dict[str, Any],
    field: str,
    default: int | None = None,
    *,
    min_value: int | None = None,
) -> int | None:
    value = payload.get(field, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"'{field}' must be an integer.")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"'{field}' must be an integer.") from exc
    if min_value is not None and out < min_value:
        raise HTTPException(
            status_code=400,
            detail=f"'{field}' must be >= {min_value}.",
        )
    return out


def _optional_float(
    payload: dict[str, Any],
    field: str,
    default: float | None = None,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    value = payload.get(field, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a number.")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"'{field}' must be a number.") from exc
    if not math.isfinite(out):
        raise HTTPException(status_code=400, detail=f"'{field}' must be finite.")
    if min_value is not None and out < min_value:
        raise HTTPException(status_code=400, detail=f"'{field}' must be >= {min_value}.")
    if max_value is not None and out > max_value:
        raise HTTPException(status_code=400, detail=f"'{field}' must be <= {max_value}.")
    return out


def _normalize_response_format(raw_format: str | None) -> tuple[str, str]:
    requested = (raw_format or "mp3").strip().lower()
    if requested not in SUPPORTED_RESPONSE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format={requested!r}. Supported: {supported}.",
        )
    if requested in {"aac", "opus"}:
        return requested, "mp3"
    return requested, requested


def _format_seconds_header(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _count_non_whitespace_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _estimate_generation_seconds_uncapped(text: str, settings: ServerSettings) -> float:
    char_count = _count_non_whitespace_chars(text)
    raw_seconds = (float(char_count) / float(settings.chars_per_second)) + float(
        settings.seconds_padding
    )
    return math.ceil(raw_seconds / SECONDS_ROUND_INCREMENT) * SECONDS_ROUND_INCREMENT


def _effective_segment_max_seconds(settings: ServerSettings) -> float:
    return min(float(settings.max_seconds), MAX_SAFE_SEGMENT_SECONDS)


def _estimate_generation_seconds(text: str, settings: ServerSettings) -> float:
    rounded = _estimate_generation_seconds_uncapped(text, settings)
    return min(max(rounded, float(settings.min_seconds)), _effective_segment_max_seconds(settings))


SEGMENT_MAX_NON_WHITESPACE_CHARS = 256
PRIMARY_BOUNDARY_CHARS: frozenset[str] = frozenset({"\n", "\r", "。", "！", "？", "!", "?"})
SECONDARY_BOUNDARY_CHARS: frozenset[str] = frozenset(
    {"、", "，", ",", "；", ";", "：", ":"},
)


def _speech_chunk_char_budget(settings: ServerSettings) -> int:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    usable_seconds = max(0.0, segment_max_seconds - float(settings.seconds_padding))
    budget = int(math.floor(usable_seconds * float(settings.chars_per_second)))
    return max(20, min(SEGMENT_MAX_NON_WHITESPACE_CHARS, budget))


def _split_chunk_at_priority_boundary(text: str) -> tuple[str, str] | None:
    n = len(text)
    if n <= 1:
        return None
    midpoint = n // 2

    def _scan(predicate) -> int | None:
        best_index: int | None = None
        best_distance: int | None = None
        for index, char in enumerate(text):
            if index >= n - 1:
                break
            if not predicate(char):
                continue
            distance = abs(index - midpoint)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    candidate_predicates = (
        lambda ch: ch in PRIMARY_BOUNDARY_CHARS,
        lambda ch: ch in SECONDARY_BOUNDARY_CHARS,
        lambda ch: ch.isspace() and ch not in PRIMARY_BOUNDARY_CHARS,
    )
    for predicate in candidate_predicates:
        index = _scan(predicate)
        if index is None:
            continue
        left, right = text[: index + 1], text[index + 1 :]
        if left.strip() and right.strip():
            return left, right
    return None


def _split_text_recursive_by_char_budget(text: str, char_budget: int) -> list[str]:
    if text == "":
        return []
    if _count_non_whitespace_chars(text) <= char_budget:
        stripped = text.strip()
        return [stripped] if stripped else []

    split = _split_chunk_at_priority_boundary(text)
    if split is None:
        n = len(text)
        if n <= 1:
            stripped = text.strip()
            return [stripped] if stripped else []
        mid = n // 2
        left_text, right_text = text[:mid], text[mid:]
    else:
        left_text, right_text = split

    out: list[str] = []
    out.extend(_split_text_recursive_by_char_budget(left_text, char_budget))
    out.extend(_split_text_recursive_by_char_budget(right_text, char_budget))
    return out


def _split_text_for_auto_chunks(text: str, settings: ServerSettings) -> list[str]:
    char_budget = _speech_chunk_char_budget(settings)
    return _split_text_recursive_by_char_budget(text, char_budget)


def _build_speech_segment_plan(
    payload: dict[str, Any],
    text: str,
    settings: ServerSettings,
) -> SpeechSegmentPlan:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    request_seconds = _optional_float(
        payload,
        "seconds",
        None,
        min_value=0.1,
        max_value=segment_max_seconds,
    )
    if request_seconds is not None:
        seconds = float(request_seconds)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "request")

    if settings.seconds is not None:
        seconds = min(float(settings.seconds), segment_max_seconds)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "fixed")

    char_budget = _speech_chunk_char_budget(settings)
    uncapped_seconds = _estimate_generation_seconds_uncapped(text, settings)
    if (
        _count_non_whitespace_chars(text) <= char_budget
        and uncapped_seconds <= segment_max_seconds
    ):
        seconds = _estimate_generation_seconds(text, settings)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "auto")

    chunks = _split_text_for_auto_chunks(text, settings)
    segments = tuple(
        SpeechSegment(text=chunk, seconds=_estimate_generation_seconds(chunk, settings))
        for chunk in chunks
    )
    if not segments:
        seconds = _estimate_generation_seconds(text, settings)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "auto")

    seconds_mode = "auto-chunked" if len(segments) > 1 else "auto"
    total_seconds = sum(segment.seconds for segment in segments)
    return SpeechSegmentPlan(segments, total_seconds, seconds_mode)


def _segment_fits_runtime_limits(
    segment: SpeechSegment,
    runtime: Any,
) -> tuple[bool, str | None]:
    normalized = normalize_text(segment.text).strip()
    if normalized == "":
        return True, None
    token_len, _ = runtime.tokenize_for_bucket(normalized)
    if int(token_len) > AUTO_TEXT_LEN_MAX:
        return False, "oversize_t"
    if int(runtime.estimate_patched_steps(float(segment.seconds))) > AUTO_SEQUENCE_LENGTH_MAX:
        return False, "oversize_s"
    return True, None


def _runtime_split_segment_recursive(
    segment: SpeechSegment,
    runtime: Any,
    settings: ServerSettings,
) -> list[SpeechSegment]:
    if segment.text.strip() == "":
        return []
    fits, _ = _segment_fits_runtime_limits(segment, runtime)
    if fits:
        return [segment]

    parent_normalized = normalize_text(segment.text).strip()
    parent_token_len: int | None = None
    if parent_normalized:
        parent_tokens, _ = runtime.tokenize_for_bucket(parent_normalized)
        parent_token_len = int(parent_tokens)

    split = _split_chunk_at_priority_boundary(segment.text)
    if split is None:
        n = len(segment.text)
        if n <= 1:
            return [segment]
        mid = n // 2
        left_text, right_text = segment.text[:mid], segment.text[mid:]
    else:
        left_text, right_text = split
    if not left_text.strip() or not right_text.strip():
        return [segment]

    if parent_token_len is not None:
        left_normalized = normalize_text(left_text).strip()
        if left_normalized:
            left_tokens, _ = runtime.tokenize_for_bucket(left_normalized)
            if int(left_tokens) >= parent_token_len:
                return [segment]

    left_segment = SpeechSegment(
        text=left_text.strip(),
        seconds=_estimate_generation_seconds(left_text, settings),
    )
    right_segment = SpeechSegment(
        text=right_text.strip(),
        seconds=_estimate_generation_seconds(right_text, settings),
    )
    refined: list[SpeechSegment] = []
    refined.extend(_runtime_split_segment_recursive(left_segment, runtime, settings))
    refined.extend(_runtime_split_segment_recursive(right_segment, runtime, settings))
    return refined


def _lifespan_warmup_default_condition_cache(
    *,
    runtime: Any,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
) -> None:
    text = settings.default_condition_cache_prepare_text
    if not text:
        return
    plan = _build_speech_segment_plan({}, text, settings)
    if plan.seconds_mode.startswith("auto"):
        plan = _runtime_refine_segment_plan_for_auto(plan, runtime, settings)
    bucket_resolutions = _resolve_auto_bucket_resolutions(None, plan.segments, runtime)
    cache_resolutions = _auto_prepare_speech_caches(
        irodori=None,
        runtime=runtime,
        settings=settings,
        cache_manager=cache_manager,
        segments=plan.segments,
        caption=None,
        bucket_resolutions=bucket_resolutions,
    )

    prepared = sum(1 for r in cache_resolutions if r.auto_status == "prepared")
    reused = sum(1 for r in cache_resolutions if r.auto_status == "reused")
    miss = sum(1 for r in cache_resolutions if r.condition_handle is None)
    bucket_ids = [
        _bucket_header_value(r.bucket) if r.bucket is not None else "none"
        for r in cache_resolutions
    ]
    print(
        "[lifespan] default condition cache warmup: "
        f"segments={len(cache_resolutions)} prepared={prepared} "
        f"reused={reused} miss={miss} buckets={bucket_ids}",
    )
    if miss and settings.strict_coreml:
        raise CoreMLStatefulUnavailableError(
            f"default condition cache warmup failed for {miss}/{len(cache_resolutions)} "
            f"segments (buckets={bucket_ids})",
        )


def _runtime_refine_segment_plan_for_auto(
    plan: SpeechSegmentPlan,
    runtime: Any,
    settings: ServerSettings,
) -> SpeechSegmentPlan:
    refined: list[SpeechSegment] = []
    for segment in plan.segments:
        refined.extend(_runtime_split_segment_recursive(segment, runtime, settings))
    if not refined:
        return plan
    if len(refined) == len(plan.segments) and all(
        new.text == old.text and new.seconds == old.seconds
        for new, old in zip(refined, plan.segments, strict=True)
    ):
        return plan
    seconds_mode = "auto-chunked" if len(refined) > 1 else plan.seconds_mode
    total_seconds = sum(segment.seconds for segment in refined)
    return SpeechSegmentPlan(tuple(refined), total_seconds, seconds_mode)


def _resolve_generation_seconds(
    payload: dict[str, Any],
    text: str,
    settings: ServerSettings,
) -> tuple[float, str]:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    request_seconds = _optional_float(
        payload,
        "seconds",
        None,
        min_value=0.1,
        max_value=segment_max_seconds,
    )
    if request_seconds is not None:
        return float(request_seconds), "request"
    if settings.seconds is not None:
        return min(float(settings.seconds), segment_max_seconds), "fixed"
    return _estimate_generation_seconds(text, settings), "auto"


def _soundfile_format(format_name: str) -> tuple[str, str | None]:
    if format_name == "wav":
        return "WAV", "PCM_16"
    if format_name == "mp3":
        return "MP3", "MPEG_LAYER_III"
    if format_name == "flac":
        return "FLAC", "PCM_16"
    raise ValueError(f"soundfile format not available for {format_name!r}.")


def _content_type(format_name: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }[format_name]


def _audio_array(audio):
    tensor = audio.detach().to("cpu").float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected audio tensor with shape (channels, samples), got {tuple(tensor.shape)}"
        )
    return tensor.clamp(-1.0, 1.0).transpose(0, 1).contiguous().numpy()


def _serialize_audio(audio, sample_rate: int, format_name: str) -> bytes:
    array = _audio_array(audio)
    if format_name == "pcm":
        return (array * 32767.0).astype("<i2", copy=False).tobytes()

    import soundfile as sf

    sf_format, subtype = _soundfile_format(format_name)
    if sf_format not in sf.available_formats():
        raise HTTPException(
            status_code=400,
            detail=f"response_format={format_name!r} is not supported by this soundfile build.",
        )
    if subtype is not None and subtype not in sf.available_subtypes(sf_format):
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format={format_name!r} subtype {subtype!r} is not supported "
                "by this soundfile build."
            ),
        )

    buffer = BytesIO()
    sf.write(buffer, array, int(sample_rate), format=sf_format, subtype=subtype)
    return buffer.getvalue()


def _normalize_audio_segment(audio):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(
            f"Expected audio tensor with shape (channels, samples), got {tuple(audio.shape)}"
        )
    return audio


def _concatenate_audio_segments(audio_segments: list[Any]):
    if not audio_segments:
        raise ValueError("No audio segments were generated.")
    if len(audio_segments) == 1:
        return audio_segments[0]

    channel_count = int(audio_segments[0].shape[0])
    total_samples = sum(int(audio.shape[1]) for audio in audio_segments)
    combined = audio_segments[0].new_empty((channel_count, total_samples))
    offset = 0
    for audio in audio_segments:
        samples = int(audio.shape[1])
        combined[:, offset : offset + samples] = audio
        offset += samples
    return combined


def _health_payload(settings: ServerSettings, state: RuntimeState) -> dict[str, Any]:
    return {
        "status": "ok",
        "object": "health",
        "model": settings.api_model_id,
        "reference_wav": str(settings.reference_wav.relative_to(PROJECT_ROOT))
        if settings.reference_wav.is_relative_to(PROJECT_ROOT)
        else str(settings.reference_wav),
        "reference_wav_exists": settings.reference_wav.is_file(),
        "model_device": settings.model_device,
        "codec_device": settings.codec_device,
        "runtime_loaded": state.runtime_loaded,
    }


def _warmup_resident_speaker_kv(
    runtime: Any,
    reference_cache_id: str,
    buckets: list[CoreMLConditionBucket],
) -> int:
    warmup = getattr(runtime, "warmup_resident_speaker_kv", None)
    get_resident = getattr(runtime, "get_resident_reference_tensors", None)
    if not callable(warmup) or not callable(get_resident):
        return 0
    resident = get_resident(reference_cache_id)
    if resident is None:
        return 0
    warmed = 0
    for bucket in buckets:
        try:
            warmup(
                resident_reference=resident,
                sequence_length_bucket=int(bucket.sequence_length),
                speaker_context_len_bucket=int(bucket.speaker_context_len_bucket),
            )
            warmed += 1
        except Exception as exc:
            print(
                f"[lifespan] resident speaker KV warmup failed for "
                f"S={bucket.sequence_length},R={bucket.speaker_context_len_bucket}: {exc}"
            )
    return warmed


def create_app(settings: ServerSettings) -> FastAPI:
    state = RuntimeState(settings)

    def prune_runtime_resident_reference(cache_id: str) -> None:
        runtime = state.runtime_if_loaded
        if runtime is not None:
            _delete_runtime_resident_reference(runtime, cache_id)

    cache_manager = InMemoryCoreMLCacheManager(
        max_memory_bytes=settings.cache_max_memory_bytes,
        on_reference_removed=prune_runtime_resident_reference,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _validate_reference_wav(settings.reference_wav)
        if settings.preload or settings.warmup_buckets:
            runtime = await asyncio.to_thread(state.get_runtime)
            if settings.warmup_buckets:
                precompile = getattr(runtime, "precompile_coreml_stateful_buckets", None)
                if callable(precompile):
                    try:
                        await asyncio.to_thread(
                            precompile,
                            list(settings.warmup_buckets),
                        )
                    except CoreMLStatefulUnavailableError as exc:
                        if settings.strict_coreml:
                            raise
                        print(
                            "[lifespan] CoreML stateful warmup unavailable; "
                            f"continuing without precompiled buckets: {exc}",
                        )
            reference_result: CacheCreateResult | None = None
            if (
                settings.default_reference_cache_prepare
                and settings.enable_resident_reference_cache
            ):
                reference_result = await asyncio.to_thread(
                    _prepare_runtime_default_reference_resident,
                    runtime,
                    settings,
                    cache_manager,
                )
                print(
                    "[lifespan] default reference resident prepared: "
                    f"{reference_result.handle.id}",
                )
            if (
                settings.enable_resident_speaker_kv
                and settings.warmup_buckets
                and reference_result is not None
            ):
                try:
                    warmed = await asyncio.to_thread(
                        _warmup_resident_speaker_kv,
                        runtime,
                        reference_result.handle.id,
                        list(settings.warmup_buckets),
                    )
                    print(
                        f"[lifespan] resident speaker KV warmup buckets prepared: {warmed}",
                    )
                except Exception as exc:
                    print(f"[lifespan] resident speaker KV warmup skipped: {exc}")
            if settings.default_condition_cache_prepare_text:
                try:
                    await asyncio.to_thread(
                        _lifespan_warmup_default_condition_cache,
                        runtime=runtime,
                        settings=settings,
                        cache_manager=cache_manager,
                    )
                except CoreMLStatefulUnavailableError:
                    if settings.strict_coreml:
                        raise
                    print(
                        "[lifespan] default condition cache warmup unavailable; "
                        "continuing without prepared caches",
                    )
                except Exception as exc:
                    if settings.strict_coreml:
                        raise
                    print(f"[lifespan] default condition cache warmup skipped: {exc}")
        try:
            yield
        finally:
            clear_cached_runtime()

    app = FastAPI(title="Irodori-TTS OpenAI-compatible API", version="0.1.0", lifespan=lifespan)
    app.state.coreml_cache_manager = cache_manager

    @app.api_route("/v1/health", methods=["GET", "POST"])
    async def health() -> dict[str, Any]:
        return _health_payload(settings, state)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.api_model_id,
                    "object": "model",
                    "created": MODEL_CREATED,
                    "owned_by": "irodori-tts",
                }
            ],
        }

    @app.post("/v1/tts/reference-caches")
    async def create_reference_cache(payload: Any = Body(None)) -> JSONResponse:
        try:
            data = _cache_require_object(payload)
            cache_mode = _cache_create_mode(data)
            result = cache_manager.prepare_reference_cache(
                _reference_cache_request(data, settings),
                cache_mode=cache_mode,
            )
            return JSONResponse(
                status_code=cache_create_status_code(result),
                content=reference_cache_response(result),
            )
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc)

    @app.get("/v1/tts/reference-caches/{cache_id}")
    async def get_reference_cache(cache_id: str) -> JSONResponse:
        try:
            handle = cache_manager.get_reference_cache(cache_id)
            return JSONResponse(content=reference_cache_response(handle))
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc)

    @app.delete("/v1/tts/reference-caches/{cache_id}")
    async def delete_reference_cache(cache_id: str, request: Request) -> Response:
        try:
            cascade = _cache_delete_cascade(request.query_params.get("cascade"))
            if not cache_manager.delete_reference_cache(cache_id, cascade=cascade):
                raise CacheNotFoundError(f"reference cache not found: {cache_id}")
            return Response(status_code=204)
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc)

    @app.post("/v1/tts/condition-caches")
    async def create_condition_cache(payload: Any = Body(None)) -> JSONResponse:
        try:
            data = _cache_require_object(payload)
            cache_mode = _cache_create_mode(data)
            result = cache_manager.prepare_condition_cache(
                _condition_cache_request(data, settings, cache_manager),
                cache_mode=cache_mode,
            )
            return JSONResponse(
                status_code=cache_create_status_code(result),
                content=condition_cache_response(result),
            )
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc)

    @app.get("/v1/tts/condition-caches/{cache_id}")
    async def get_condition_cache(cache_id: str) -> JSONResponse:
        try:
            handle = cache_manager.get_condition_cache(cache_id)
            return JSONResponse(content=condition_cache_response(handle))
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc)

    @app.delete("/v1/tts/condition-caches/{cache_id}")
    async def delete_condition_cache(cache_id: str) -> Response:
        try:
            if not cache_manager.delete_condition_cache(cache_id):
                raise CacheNotFoundError(f"condition cache not found: {cache_id}")
            return Response(status_code=204)
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc)

    @app.get("/v1/tts/cache-metrics")
    async def cache_metrics() -> JSONResponse:
        snapshot = cache_manager.metrics_snapshot()
        runtime_metrics: list[dict[str, Any]] = []
        resident_reference_metrics: dict[str, Any] = {}
        resident_speaker_kv_metrics: dict[str, Any] = {}
        runtime = state.runtime_if_loaded
        if runtime is not None:
            metrics_fn = getattr(runtime, "coreml_stateful_metrics_snapshot", None)
            if callable(metrics_fn):
                runtime_metrics = list(metrics_fn())
            resident_metrics_fn = getattr(runtime, "resident_reference_metrics_snapshot", None)
            if callable(resident_metrics_fn):
                resident_reference_metrics = dict(resident_metrics_fn())
            speaker_kv_metrics_fn = getattr(
                runtime, "resident_speaker_kv_metrics_snapshot", None
            )
            if callable(speaker_kv_metrics_fn):
                resident_speaker_kv_metrics = dict(speaker_kv_metrics_fn())
        return JSONResponse(
            content={
                "cache_manager": snapshot,
                "coreml_stateful_backends": runtime_metrics,
                "resident_references": resident_reference_metrics,
                "resident_speaker_kv": resident_speaker_kv_metrics,
            },
        )

    @app.post("/v1/audio/speech")
    async def audio_speech(payload: Any = Body(...)) -> Response:
        data = _require_object(payload)
        irodori: dict[str, Any] | None = None
        cache_mode: str | None = None
        cache_id: str | None = None
        try:
            irodori = _speech_irodori_extension(data)
            cache_mode = _speech_cache_mode(irodori)
            if cache_mode != CACHE_MODE_OFF:
                cache_id = _speech_cache_id(irodori)
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc, cache_id=cache_id, cache_mode=cache_mode)

        text = _required_text(data, "input")
        # Compatibility-only fields such as model, voice, and reference_audio are
        # deliberately ignored. Runtime selection and speaker reference are server-owned.
        _optional_float(data, "speed", 1.0, min_value=0.25, max_value=4.0)
        requested_format, output_format = _normalize_response_format(
            _optional_text(data, "response_format", "mp3")
        )
        num_steps = _optional_int(
            data,
            "num_steps",
            settings.default_num_steps,
            min_value=1,
        )
        if int(num_steps) > settings.max_num_steps:
            raise HTTPException(
                status_code=400,
                detail=f"'num_steps' must be <= {settings.max_num_steps}.",
            )
        seed = _optional_int(data, "seed", None)
        caption = _resolve_caption(data)
        segment_plan = _build_speech_segment_plan(data, text, settings)
        cache_resolutions = [
            _SpeechCacheResolution(None, None, False, False) for _segment in segment_plan.segments
        ]
        auto_bucket_resolutions: list[AutoBucketResolution] = []
        runtime: Any | None = None
        resident_reference_plan: _ResidentReferencePlan | None = None
        try:
            if cache_mode in {CACHE_MODE_PREPARE, CACHE_MODE_REFRESH}:
                if len(segment_plan.segments) != 1:
                    raise CacheConflictError(
                        "cache_mode=prepare/refresh does not support multi-segment speech",
                    )
                _speech_condition_cache_request(
                    irodori or {},
                    settings,
                    "ref_prevalidate",
                    1,
                    segment_plan.segments[0],
                    caption,
                )
                if not settings.reference_wav.is_file():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Configured reference_wav not found: {settings.reference_wav}",
                    )
                runtime = await asyncio.to_thread(state.get_runtime)
                resident_reference_plan = await asyncio.to_thread(
                    _prepare_runtime_default_reference_plan,
                    runtime,
                    settings,
                    force_refresh=cache_mode == CACHE_MODE_REFRESH,
                )
            cache_resolutions = _resolve_speech_cache(
                irodori,
                str(cache_mode),
                cache_id,
                segment_plan,
                caption,
                settings,
                cache_manager,
                runtime=runtime,
                resident_reference_plan=resident_reference_plan,
            )
            if len(cache_resolutions) != len(segment_plan.segments):
                raise CacheConflictError("speech cache resolution count did not match segments")
        except CACHE_EXCEPTIONS as exc:
            if runtime is not None and resident_reference_plan is not None:
                _delete_runtime_resident_reference(runtime, resident_reference_plan.provisional_id)
            return _cache_error_response(exc, cache_id=cache_id, cache_mode=cache_mode)
        except CoreMLStatefulUnavailableError as exc:
            if runtime is not None and resident_reference_plan is not None:
                _delete_runtime_resident_reference(runtime, resident_reference_plan.provisional_id)
            return _coreml_backend_error_response(
                exc,
                cache_id=cache_id,
                cache_mode=cache_mode,
            )
        except ValueError as exc:
            if runtime is not None and resident_reference_plan is not None:
                _delete_runtime_resident_reference(runtime, resident_reference_plan.provisional_id)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if runtime is not None and resident_reference_plan is not None:
                _delete_runtime_resident_reference(runtime, resident_reference_plan.provisional_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not settings.reference_wav.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Configured reference_wav not found: {settings.reference_wav}",
            )

        try:
            if runtime is None:
                runtime = await asyncio.to_thread(state.get_runtime)
            if cache_mode == CACHE_MODE_AUTO and cache_id is None:
                if segment_plan.seconds_mode.startswith("auto"):
                    refined_plan = await asyncio.to_thread(
                        _runtime_refine_segment_plan_for_auto,
                        segment_plan,
                        runtime,
                        settings,
                    )
                    if refined_plan is not segment_plan:
                        segment_plan = refined_plan
                        cache_resolutions = [
                            _SpeechCacheResolution(None, None, False, False)
                            for _segment in segment_plan.segments
                        ]
                auto_bucket_resolutions = await asyncio.to_thread(
                    _resolve_auto_bucket_resolutions,
                    irodori,
                    segment_plan.segments,
                    runtime,
                )
                cache_resolutions = await asyncio.to_thread(
                    _auto_prepare_speech_caches,
                    irodori=irodori,
                    runtime=runtime,
                    settings=settings,
                    cache_manager=cache_manager,
                    segments=segment_plan.segments,
                    caption=caption,
                    bucket_resolutions=auto_bucket_resolutions,
                )
            audio_segments: list[Any] = []
            sample_rate: int | None = None
            channel_count: int | None = None
            segment_backends: list[str] = []
            used_condition_cache_ids: list[str] = []
            used_reference_cache_ids: list[str] = []
            fast_path_cfg = (
                _speech_fast_path_cfg(irodori)
                if any(resolution.condition_handle is not None for resolution in cache_resolutions)
                else None
            )
            for segment_index, segment in enumerate(segment_plan.segments):
                cache_resolution = cache_resolutions[segment_index]
                condition_cache_handle = cache_resolution.condition_handle
                segment_seed = None if seed is None else int(seed) + segment_index
                sampling_request = SamplingRequest(
                    text=segment.text,
                    caption=caption,
                    ref_wav=str(settings.reference_wav),
                    ref_latent=None,
                    no_ref=False,
                    num_steps=int(num_steps),
                    seconds=float(segment.seconds),
                    max_ref_seconds=settings.max_ref_seconds,
                    seed=segment_seed,
                )
                if condition_cache_handle is not None and fast_path_cfg is not None:
                    sampling_request.cfg_guidance_mode = str(fast_path_cfg["guidance_mode"])
                    sampling_request.cfg_scale_text = float(fast_path_cfg["scale_text"])
                    sampling_request.cfg_scale_caption = float(fast_path_cfg["scale_caption"])
                    sampling_request.cfg_scale_speaker = float(fast_path_cfg["scale_speaker"])
                    sampling_request.cfg_min_t = float(fast_path_cfg["min_t"])
                    sampling_request.cfg_max_t = float(fast_path_cfg["max_t"])
                    speaker_kv_scale = fast_path_cfg.get("speaker_kv_scale")
                    sampling_request.speaker_kv_scale = (
                        None if speaker_kv_scale is None else float(speaker_kv_scale)
                    )
                    speaker_kv_min_t = fast_path_cfg.get("speaker_kv_min_t")
                    sampling_request.speaker_kv_min_t = (
                        None if speaker_kv_min_t is None else float(speaker_kv_min_t)
                    )
                    speaker_kv_max_layers = fast_path_cfg.get("speaker_kv_max_layers")
                    sampling_request.speaker_kv_max_layers = (
                        None if speaker_kv_max_layers is None else int(speaker_kv_max_layers)
                    )

                if condition_cache_handle is None:
                    if cache_mode == CACHE_MODE_AUTO and settings.strict_coreml:
                        raise CoreMLStatefulUnavailableError(
                            _strict_auto_unavailable_reason(cache_resolution),
                        )
                    result = await asyncio.to_thread(
                        runtime.synthesize,
                        sampling_request,
                        log_fn=print if settings.log_timings else None,
                    )
                    segment_backends.append("pytorch")
                else:
                    try:
                        synthesize_with_condition_cache = getattr(
                            runtime,
                            "synthesize_with_condition_cache",
                            None,
                        )
                        if synthesize_with_condition_cache is None:
                            raise CoreMLStatefulUnavailableError(
                                "runtime does not expose synthesize_with_condition_cache"
                            )
                        resident_reference_tensors = None
                        if cache_resolution.reference_cache_id is not None:
                            get_resident_reference = getattr(
                                runtime,
                                "get_resident_reference_tensors",
                                None,
                            )
                            if callable(get_resident_reference):
                                resident_reference_tensors = get_resident_reference(
                                    cache_resolution.reference_cache_id,
                                )
                        synthesize_kwargs: dict[str, Any] = {
                            "condition_cache": condition_cache_handle,
                            "log_fn": print if settings.log_timings else None,
                        }
                        if resident_reference_tensors is not None:
                            synthesize_kwargs["resident_reference_tensors"] = (
                                resident_reference_tensors
                            )
                        result = await asyncio.to_thread(
                            synthesize_with_condition_cache,
                            sampling_request,
                            **synthesize_kwargs,
                        )
                        segment_backends.append("coreml-stateful")
                        used_condition_cache_ids.append(condition_cache_handle.id)
                        if cache_resolution.reference_cache_id is not None:
                            used_reference_cache_ids.append(
                                cache_resolution.reference_cache_id,
                            )
                    except CoreMLStatefulUnavailableError:
                        if cache_mode != CACHE_MODE_AUTO or settings.strict_coreml:
                            raise
                        cache_resolutions[segment_index] = replace(
                            cache_resolution,
                            auto_status="miss-fallback",
                            fallback_reason="coreml-stateful-unavailable",
                        )
                        result = await asyncio.to_thread(
                            runtime.synthesize,
                            sampling_request,
                            log_fn=print if settings.log_timings else None,
                        )
                        segment_backends.append("pytorch")
                audio = _normalize_audio_segment(result.audio)
                result_sample_rate = int(result.sample_rate)
                result_channel_count = int(audio.shape[0])
                if sample_rate is None:
                    sample_rate = result_sample_rate
                    channel_count = result_channel_count
                elif result_sample_rate != sample_rate:
                    raise ValueError("Generated audio sample rates did not match across chunks.")
                elif result_channel_count != channel_count:
                    raise ValueError("Generated audio channel counts did not match across chunks.")
                audio_segments.append(audio)

            if sample_rate is None:
                raise ValueError("No audio segments were generated.")
            audio = _concatenate_audio_segments(audio_segments)
            audio_bytes = _serialize_audio(audio, sample_rate, output_format)
        except HTTPException:
            raise
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc, cache_id=cache_id, cache_mode=cache_mode)
        except CoreMLStatefulUnavailableError as exc:
            return _coreml_backend_error_response(
                exc,
                cache_id=cache_id,
                cache_mode=cache_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        denoiser_backend = _aggregate_backend_header(segment_backends)
        condition_cache_id_header = _summarize_cache_id_header(used_condition_cache_ids)
        reference_cache_id_header = _summarize_cache_id_header(
            list(dict.fromkeys(used_reference_cache_ids)),
        )
        headers = {
            "Content-Disposition": f'attachment; filename="speech.{output_format}"',
            "X-Irodori-Requested-Format": requested_format,
            "X-Irodori-Voice-Resolved": "server-default",
            "X-Irodori-Denoiser-Backend": denoiser_backend,
            "X-Irodori-Backend": denoiser_backend,
            "X-Irodori-Generation-Seconds": _format_seconds_header(segment_plan.total_seconds),
            "X-Irodori-Seconds-Mode": segment_plan.seconds_mode,
            "X-Irodori-Chunk-Count": str(len(segment_plan.segments)),
            "X-Irodori-Chunk-Seconds": ",".join(
                _format_seconds_header(segment.seconds) for segment in segment_plan.segments
            ),
            "X-Irodori-Num-Steps": str(int(num_steps)),
        }
        if len(segment_backends) > 1 or denoiser_backend == "mixed":
            headers["X-Irodori-Backend-Per-Segment"] = ",".join(segment_backends)
        if condition_cache_id_header is not None:
            headers["X-Irodori-Condition-Cache-Id"] = condition_cache_id_header
            headers["X-Irodori-Cache-Condition-Id"] = condition_cache_id_header
            headers["X-Irodori-Cache-Condition-Count"] = str(
                len(used_condition_cache_ids),
            )
        if reference_cache_id_header is not None and condition_cache_id_header is not None:
            headers["X-Irodori-Reference-Cache-Id"] = reference_cache_id_header
            headers["X-Irodori-Cache-Reference-Id"] = reference_cache_id_header
        if cache_mode == CACHE_MODE_AUTO:
            auto_status = _summarize_header_values(
                [resolution.auto_status for resolution in cache_resolutions],
            )
            if auto_status is not None:
                headers["X-Irodori-Cache-Auto"] = auto_status
        bucket_headers = _bucket_headers_for_resolutions(cache_resolutions)
        if bucket_headers:
            headers["X-Irodori-Bucket"] = ",".join(bucket_headers)
        fallback_reason = _summarize_header_values(
            [resolution.fallback_reason for resolution in cache_resolutions],
        )
        if fallback_reason is not None:
            headers["X-Irodori-Fallback-Reason"] = fallback_reason
        bucket_attempted_headers = [
            auto_bucket_resolution.attempted
            for auto_bucket_resolution in auto_bucket_resolutions
            if auto_bucket_resolution.attempted is not None
        ]
        if bucket_attempted_headers:
            headers["X-Irodori-Bucket-Attempted"] = ",".join(bucket_attempted_headers)
        return Response(
            content=audio_bytes,
            media_type=_content_type(output_format),
            headers=headers,
        )

    return app


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if value in {"0", "false", "no", "off", "n", "f"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    default_device = _prefer_mps_device()
    parser = argparse.ArgumentParser(description="OpenAI-compatible HTTP TTS API for Irodori-TTS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Local checkpoint file or Hugging Face repo id.",
    )
    parser.add_argument(
        "--reference-wav",
        default="rem.wav",
        help="Server-forced speaker reference wav. Client voice/reference fields are ignored.",
    )
    parser.add_argument("--api-model-id", default=DEFAULT_API_MODEL_ID)
    parser.add_argument("--model-device", default=default_device)
    parser.add_argument("--codec-device", default=default_device)
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--default-num-steps", type=int, default=40)
    parser.add_argument("--max-num-steps", type=int, default=80)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Fixed generation horizon for every request. If omitted, seconds are estimated from input length.",
    )
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--max-seconds", type=float, default=70.0)
    parser.add_argument(
        "--chars-per-second",
        type=float,
        default=5.5,
        help=(
            "Chars-per-second used to estimate AUTO speech duration and CoreML bucket "
            "selection. Calibrated for Japanese explanatory speech."
        ),
    )
    parser.add_argument("--seconds-padding", type=float, default=1.5)
    parser.add_argument(
        "--max-ref-seconds",
        type=float,
        default=30.0,
        help="Maximum server reference duration. Set <=0 to disable the cap.",
    )
    parser.add_argument("--preload", action="store_true", help="Load the runtime during startup.")
    parser.add_argument(
        "--log-timings",
        action="store_true",
        help="Print runtime timing logs. Request payloads and secrets are never printed.",
    )
    parser.add_argument(
        "--cache-max-memory-bytes",
        type=int,
        default=None,
        help="Optional in-memory CoreML cache LRU budget in bytes. Empty disables eviction.",
    )
    parser.add_argument(
        "--warmup-bucket",
        action="append",
        default=[],
        metavar="S=<seq>,T=<text_len>,R=<speaker_ctx_bucket>",
        help=(
            "Optional cond1 CoreML bucket to precompile at startup. Repeatable. "
            "Example: --warmup-bucket S=100,T=256,R=160"
        ),
    )
    parser.add_argument(
        "--auto-prepare-default-cache",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("IRODORI_AUTO_PREPARE_DEFAULT_CACHE", True),
        help=(
            "Auto-prepare condition cache for AUTO requests without cache_id. "
            "When false, AUTO requests fall back to PyTorch instead of preparing condition caches."
        ),
    )
    parser.add_argument(
        "--strict-coreml",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("IRODORI_STRICT_COREML", False),
        help=(
            "When true, AUTO requests return strict errors instead of silently falling back "
            "to PyTorch on CoreML/cache prep failures."
        ),
    )
    parser.add_argument(
        "--default-reference-cache-prepare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepare default reference resident tensors during lifespan startup.",
    )
    parser.add_argument(
        "--default-condition-cache-prepare-text",
        type=str,
        default=None,
        help="Optional warmup text to log a best-effort default condition cache prep on startup.",
    )
    parser.add_argument(
        "--condition-cache-default-ttl-seconds",
        type=float,
        default=86400.0,
        help=(
            "TTL (seconds) for internally-prepared AUTO condition caches. "
            "Use <=0 to disable TTL (None)."
        ),
    )
    parser.add_argument(
        "--max-resident-speaker-kv-buckets",
        type=int,
        default=_env_int("IRODORI_MAX_RESIDENT_SPEAKER_KV_BUCKETS", 3),
        help="Maximum resident speaker KV bucket entries kept in the runtime pool.",
    )
    parser.add_argument(
        "--max-resident-condition-cache-entries",
        type=int,
        default=_env_int("IRODORI_MAX_RESIDENT_CONDITION_CACHE_ENTRIES", 0),
        help="Reserved for layer-4 condition packed-KV residency. 0 disables.",
    )
    parser.add_argument(
        "--enable-resident-reference-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When false, AUTO requests bypass resident reference cache prep and fall back "
            "to PyTorch (no metadata-only CoreML path)."
        ),
    )
    parser.add_argument(
        "--enable-resident-speaker-kv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable runtime-resident speaker KV pool. When false, fresh build per request.",
    )
    parser.add_argument(
        "--enable-condition-packed-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserved for layer-4 packed-KV residency. Currently unimplemented.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def _parse_warmup_bucket_spec(spec: str) -> CoreMLConditionBucket:
    parts = [part.strip() for part in str(spec).split(",") if part.strip()]
    fields: dict[str, int] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"warmup bucket spec must be key=value, got {part!r}")
        key, value = part.split("=", 1)
        key = key.strip().upper()
        try:
            fields[key] = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"warmup bucket {key} must be int, got {value!r}") from exc
    if not {"S", "T", "R"}.issubset(fields):
        raise ValueError(f"warmup bucket spec must include S, T, R (got {sorted(fields.keys())})")
    return CoreMLConditionBucket(
        sequence_length=fields["S"],
        text_len=fields["T"],
        speaker_context_len_bucket=fields["R"],
    )


def build_settings_from_args(args: argparse.Namespace) -> ServerSettings:
    if int(args.default_num_steps) <= 0:
        raise ValueError("--default-num-steps must be > 0.")
    if int(args.max_num_steps) <= 0:
        raise ValueError("--max-num-steps must be > 0.")
    if int(args.default_num_steps) > int(args.max_num_steps):
        raise ValueError("--default-num-steps must be <= --max-num-steps.")
    if args.seconds is not None and float(args.seconds) <= 0:
        raise ValueError("--seconds must be > 0.")
    if float(args.min_seconds) <= 0:
        raise ValueError("--min-seconds must be > 0.")
    if float(args.max_seconds) < float(args.min_seconds):
        raise ValueError("--max-seconds must be >= --min-seconds.")
    if float(args.chars_per_second) <= 0:
        raise ValueError("--chars-per-second must be > 0.")
    if float(args.seconds_padding) < 0:
        raise ValueError("--seconds-padding must be >= 0.")
    max_ref_seconds = float(args.max_ref_seconds)
    cache_max_memory_bytes = (
        None if args.cache_max_memory_bytes is None else int(args.cache_max_memory_bytes)
    )
    if cache_max_memory_bytes is not None and cache_max_memory_bytes <= 0:
        raise ValueError("--cache-max-memory-bytes must be > 0 when supplied.")
    warmup_buckets = tuple(_parse_warmup_bucket_spec(spec) for spec in (args.warmup_bucket or []))

    ttl_raw = float(args.condition_cache_default_ttl_seconds)
    if math.isnan(ttl_raw) or math.isinf(ttl_raw):
        raise ValueError("--condition-cache-default-ttl-seconds must be finite.")
    condition_cache_default_ttl_seconds: float | None = None if ttl_raw <= 0 else ttl_raw

    max_resident_speaker_kv_buckets = int(args.max_resident_speaker_kv_buckets)
    if max_resident_speaker_kv_buckets < 0:
        raise ValueError("--max-resident-speaker-kv-buckets must be >= 0.")
    max_resident_condition_cache_entries = int(args.max_resident_condition_cache_entries)
    if max_resident_condition_cache_entries < 0:
        raise ValueError("--max-resident-condition-cache-entries must be >= 0.")

    prepare_text_raw = args.default_condition_cache_prepare_text
    prepare_text = None if prepare_text_raw is None else str(prepare_text_raw).strip() or None

    return ServerSettings(
        host=str(args.host),
        port=int(args.port),
        checkpoint=str(args.checkpoint),
        reference_wav=_project_relative_path(str(args.reference_wav)),
        api_model_id=str(args.api_model_id),
        model_device=str(args.model_device),
        codec_device=str(args.codec_device),
        model_precision=str(args.model_precision),
        codec_precision=str(args.codec_precision),
        codec_repo=str(args.codec_repo),
        default_num_steps=int(args.default_num_steps),
        max_num_steps=int(args.max_num_steps),
        seconds=None if args.seconds is None else float(args.seconds),
        min_seconds=float(args.min_seconds),
        max_seconds=float(args.max_seconds),
        chars_per_second=float(args.chars_per_second),
        seconds_padding=float(args.seconds_padding),
        max_ref_seconds=None if max_ref_seconds <= 0 else max_ref_seconds,
        preload=bool(args.preload),
        log_timings=bool(args.log_timings),
        cache_max_memory_bytes=cache_max_memory_bytes,
        warmup_buckets=warmup_buckets,
        auto_prepare_default_cache=bool(args.auto_prepare_default_cache),
        strict_coreml=bool(args.strict_coreml),
        default_reference_cache_prepare=bool(args.default_reference_cache_prepare),
        default_condition_cache_prepare_text=prepare_text,
        condition_cache_default_ttl_seconds=condition_cache_default_ttl_seconds,
        max_resident_speaker_kv_buckets=max_resident_speaker_kv_buckets,
        max_resident_condition_cache_entries=max_resident_condition_cache_entries,
        enable_resident_reference_cache=bool(args.enable_resident_reference_cache),
        enable_resident_speaker_kv=bool(args.enable_resident_speaker_kv),
        enable_condition_packed_kv_cache=bool(args.enable_condition_packed_kv_cache),
    )


def main() -> None:
    settings = build_settings_from_args(parse_args())
    _validate_reference_wav(settings.reference_wav)

    import uvicorn

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
