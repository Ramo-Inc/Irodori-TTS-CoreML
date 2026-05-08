#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_API_MODEL_ID = "irodori-tts-500m-v2"
MODEL_CREATED = 1700000000
SUPPORTED_RESPONSE_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
SECONDS_ROUND_INCREMENT = 0.5
MAX_SAFE_SEGMENT_SECONDS = 30.0
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


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    seconds: float


@dataclass(frozen=True)
class SpeechSegmentPlan:
    segments: tuple[SpeechSegment, ...]
    total_seconds: float
    seconds_mode: str


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
            self._runtime = runtime
        return runtime

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
        return CACHE_MODE_OFF
    return normalize_speech_cache_mode(irodori.get("cache_mode"))


def _speech_cache_id(irodori: dict[str, Any] | None) -> str | None:
    if irodori is None:
        return None
    value = irodori.get("cache_id", _MISSING)
    if value is _MISSING or value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CacheValidationError("irodori.cache_id must be a non-empty string")
    return value.strip()


def _speech_reference_cache_request(settings: ServerSettings) -> ReferenceCacheRequest:
    return _reference_cache_request(
        {"source": {"type": "server_default"}},
        settings,
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
    _raise_speech_reference_conflict(
        "speaker_context_len",
        reference_handle.speaker_context_len,
        expected_request.speaker_context_len,
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


def _resolve_speech_cache(
    irodori: dict[str, Any] | None,
    cache_mode: str,
    cache_id: str | None,
    segment_plan: SpeechSegmentPlan,
    caption: str | None,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
) -> _SpeechCacheResolution:
    if cache_mode == CACHE_MODE_OFF:
        return _SpeechCacheResolution(None, None, False, False)

    if cache_mode in {CACHE_MODE_PREPARE, CACHE_MODE_REFRESH}:
        if len(segment_plan.segments) != 1:
            raise CacheConflictError(
                "cache_mode=prepare/refresh does not support multi-segment speech",
            )
        return _prepare_or_refresh_speech_cache(
            irodori=irodori,
            cache_mode=cache_mode,
            cache_id=cache_id,
            segment=segment_plan.segments[0],
            caption=caption,
            settings=settings,
            cache_manager=cache_manager,
        )

    if cache_mode == CACHE_MODE_REQUIRE and cache_id is None:
        raise CacheValidationError("irodori.cache_id is required when cache_mode=require")
    if cache_id is None:
        return _SpeechCacheResolution(None, None, False, False)
    if len(segment_plan.segments) != 1:
        raise CacheConflictError("single cache_id cannot be used with multi-segment speech")

    try:
        condition_handle = cache_manager.peek_condition_cache(cache_id)
    except (CacheExpiredError, CacheNotFoundError):
        if cache_mode == CACHE_MODE_AUTO:
            return _SpeechCacheResolution(None, None, False, False)
        raise

    try:
        expected_reference_request = _speech_reference_cache_request(settings)
        expected_reference_cache_id = cache_manager.reference_cache_id_for_request(
            expected_reference_request,
        )
        reference_handle = cache_manager.peek_reference_cache(
            condition_handle.reference_cache_id,
        )
        _validate_speech_reference_cache(reference_handle, expected_reference_request)
        expected_request = _speech_condition_cache_request(
            irodori or {},
            settings,
            expected_reference_cache_id,
            expected_reference_request.speaker_context_len,
            segment_plan.segments[0],
            caption,
        )
        validated = cache_manager.require_condition_cache(cache_id, expected_request)
    except (CacheExpiredError, CacheNotFoundError):
        if cache_mode == CACHE_MODE_AUTO:
            return _SpeechCacheResolution(None, None, False, False)
        raise
    return _SpeechCacheResolution(
        condition_handle=validated,
        reference_cache_id=validated.reference_cache_id,
        reference_created=False,
        condition_created=False,
    )


def _prepare_or_refresh_speech_cache(
    *,
    irodori: dict[str, Any] | None,
    cache_mode: str,
    cache_id: str | None,
    segment: SpeechSegment,
    caption: str | None,
    settings: ServerSettings,
    cache_manager: InMemoryCoreMLCacheManager,
) -> _SpeechCacheResolution:
    reference_request = _speech_reference_cache_request(settings)
    expected_reference_cache_id = cache_manager.reference_cache_id_for_request(
        reference_request,
    )
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
        raise CacheConflictError(
            "cache_id does not match prepared condition cache for this request",
        )

    reference_cache_mode = (
        CACHE_MANAGER_REFRESH_MODE
        if cache_mode == CACHE_MODE_REFRESH
        else CACHE_MODE_CREATE_OR_REUSE
    )
    reference_result = cache_manager.prepare_reference_cache(
        reference_request,
        cache_mode=reference_cache_mode,
    )
    condition_result = cache_manager.prepare_condition_cache(
        condition_request,
        cache_mode=reference_cache_mode,
    )
    return _SpeechCacheResolution(
        condition_handle=condition_result.handle,
        reference_cache_id=reference_result.handle.id,
        reference_created=not reference_result.reused,
        condition_created=not condition_result.reused,
    )


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


def _speech_chunk_char_budget(settings: ServerSettings) -> int:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    usable_seconds = max(0.0, segment_max_seconds - float(settings.seconds_padding))
    budget = int(math.floor(usable_seconds * float(settings.chars_per_second) * 0.9))
    return max(20, budget)


def _split_at_boundaries(text: str, boundaries: set[str]) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in boundaries:
            part = "".join(buffer)
            if part.strip():
                parts.append(part)
            buffer = []
    part = "".join(buffer)
    if part.strip():
        parts.append(part)
    return parts


def _split_at_secondary_boundaries(text: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in {"、", "，", ",", "；", ";", "：", ":"} or char.isspace():
            part = "".join(buffer)
            if part.strip():
                parts.append(part)
            buffer = []
    part = "".join(buffer)
    if part.strip():
        parts.append(part)
    return parts


def _hard_split_by_char_budget(text: str, char_budget: int) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    char_count = 0
    for char in text:
        char_weight = 0 if char.isspace() else 1
        if buffer and char_count + char_weight > char_budget:
            part = "".join(buffer).strip()
            if part:
                parts.append(part)
            buffer = []
            char_count = 0
        buffer.append(char)
        char_count += char_weight
    part = "".join(buffer).strip()
    if part:
        parts.append(part)
    return parts


def _pack_text_chunks(parts: list[str], char_budget: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part.strip():
            continue
        candidate = f"{current}{part}" if current else part
        if current and _count_non_whitespace_chars(candidate) > char_budget:
            chunk = current.strip()
            if chunk:
                chunks.append(chunk)
            current = part
        else:
            current = candidate

    chunk = current.strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _split_text_for_auto_chunks(text: str, settings: ServerSettings) -> list[str]:
    char_budget = _speech_chunk_char_budget(settings)
    sentence_parts = _split_at_boundaries(text, {"。", "！", "？", "!", "?", "\n", "\r"})
    small_parts: list[str] = []

    for sentence in sentence_parts:
        if _count_non_whitespace_chars(sentence) <= char_budget:
            small_parts.append(sentence)
            continue

        for part in _split_at_secondary_boundaries(sentence):
            if _count_non_whitespace_chars(part) <= char_budget:
                small_parts.append(part)
            else:
                small_parts.extend(_hard_split_by_char_budget(part, char_budget))

    return _pack_text_chunks(small_parts, char_budget)


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

    uncapped_seconds = _estimate_generation_seconds_uncapped(text, settings)
    if uncapped_seconds <= segment_max_seconds:
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


def create_app(settings: ServerSettings) -> FastAPI:
    state = RuntimeState(settings)
    cache_manager = InMemoryCoreMLCacheManager(
        max_memory_bytes=settings.cache_max_memory_bytes,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _validate_reference_wav(settings.reference_wav)
        if settings.preload or settings.warmup_buckets:
            runtime = await asyncio.to_thread(state.get_runtime)
            if settings.warmup_buckets:
                precompile = getattr(runtime, "precompile_coreml_stateful_buckets", None)
                if callable(precompile):
                    await asyncio.to_thread(
                        precompile,
                        list(settings.warmup_buckets),
                    )
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
        runtime = state.runtime_if_loaded
        if runtime is not None:
            metrics_fn = getattr(runtime, "coreml_stateful_metrics_snapshot", None)
            if callable(metrics_fn):
                runtime_metrics = list(metrics_fn())
        return JSONResponse(
            content={
                "cache_manager": snapshot,
                "coreml_stateful_backends": runtime_metrics,
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
        condition_cache_handle: ConditionCacheHandle | None = None
        cache_resolution = _SpeechCacheResolution(None, None, False, False)
        try:
            cache_resolution = _resolve_speech_cache(
                irodori,
                str(cache_mode),
                cache_id,
                segment_plan,
                caption,
                settings,
                cache_manager,
            )
            condition_cache_handle = cache_resolution.condition_handle
        except CACHE_EXCEPTIONS as exc:
            return _cache_error_response(exc, cache_id=cache_id, cache_mode=cache_mode)

        if not settings.reference_wav.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Configured reference_wav not found: {settings.reference_wav}",
            )

        try:
            runtime = await asyncio.to_thread(state.get_runtime)
            audio_segments: list[Any] = []
            sample_rate: int | None = None
            channel_count: int | None = None
            denoiser_backend = (
                "coreml-stateful" if condition_cache_handle is not None else "pytorch"
            )
            condition_cache_id_header = (
                condition_cache_handle.id if condition_cache_handle is not None else None
            )
            fast_path_cfg = (
                _speech_fast_path_cfg(irodori) if condition_cache_handle is not None else None
            )
            for segment_index, segment in enumerate(segment_plan.segments):
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
                if fast_path_cfg is not None:
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
                    result = await asyncio.to_thread(
                        runtime.synthesize,
                        sampling_request,
                        log_fn=print if settings.log_timings else None,
                    )
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
                        result = await asyncio.to_thread(
                            synthesize_with_condition_cache,
                            sampling_request,
                            condition_cache=condition_cache_handle,
                            log_fn=print if settings.log_timings else None,
                        )
                    except CoreMLStatefulUnavailableError:
                        if cache_mode != CACHE_MODE_AUTO:
                            raise
                        denoiser_backend = "pytorch"
                        condition_cache_id_header = None
                        result = await asyncio.to_thread(
                            runtime.synthesize,
                            sampling_request,
                            log_fn=print if settings.log_timings else None,
                        )
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
        except CoreMLStatefulUnavailableError as exc:
            return _coreml_backend_error_response(
                exc,
                cache_id=cache_id,
                cache_mode=cache_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        headers = {
            "Content-Disposition": f'attachment; filename="speech.{output_format}"',
            "X-Irodori-Requested-Format": requested_format,
            "X-Irodori-Denoiser-Backend": denoiser_backend,
            "X-Irodori-Generation-Seconds": _format_seconds_header(segment_plan.total_seconds),
            "X-Irodori-Seconds-Mode": segment_plan.seconds_mode,
            "X-Irodori-Chunk-Count": str(len(segment_plan.segments)),
            "X-Irodori-Chunk-Seconds": ",".join(
                _format_seconds_header(segment.seconds) for segment in segment_plan.segments
            ),
            "X-Irodori-Num-Steps": str(int(num_steps)),
        }
        if condition_cache_id_header is not None:
            headers["X-Irodori-Condition-Cache-Id"] = condition_cache_id_header
        if (
            cache_resolution.reference_cache_id is not None
            and condition_cache_id_header is not None
            and cache_mode in {CACHE_MODE_PREPARE, CACHE_MODE_REFRESH}
        ):
            headers["X-Irodori-Reference-Cache-Id"] = cache_resolution.reference_cache_id
        return Response(
            content=audio_bytes,
            media_type=_content_type(output_format),
            headers=headers,
        )

    return app


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--chars-per-second", type=float, default=4.0)
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
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
    settings = ServerSettings(
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
    )
    _validate_reference_wav(settings.reference_wav)

    import uvicorn

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
