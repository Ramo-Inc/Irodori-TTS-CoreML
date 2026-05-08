from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

STATE_LAYOUT_PER_LAYER = "per_layer_text_speaker_context_v1"
BRANCH_LAYOUT_COND1 = "cond1"
BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3 = "independent_text_speaker3"
ALLOWED_CONDITION_BRANCH_LAYOUTS = (
    (BRANCH_LAYOUT_COND1,),
    (BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3),
)

CACHE_MODE_OFF = "off"
CACHE_MODE_AUTO = "auto"
CACHE_MODE_PREPARE = "prepare"
CACHE_MODE_REQUIRE = "require"
CACHE_MODE_REFRESH = "refresh"
ALLOWED_SPEECH_CACHE_MODES = frozenset(
    {
        CACHE_MODE_OFF,
        CACHE_MODE_AUTO,
        CACHE_MODE_PREPARE,
        CACHE_MODE_REQUIRE,
        CACHE_MODE_REFRESH,
    },
)

DEFAULT_NUM_LAYERS = 12
DEFAULT_NUM_HEADS = 20
DEFAULT_HEAD_DIM = 64
FP16_BYTES = 2


class CacheNotFoundError(Exception):
    """Raised when a requested in-memory cache handle does not exist."""


class CacheExpiredError(Exception):
    """Raised when a requested in-memory cache handle exists but is expired."""


class CacheConflictError(Exception):
    """Raised when an existing cache handle is incompatible with a request."""


class CacheValidationError(ValueError):
    """Raised when cache request metadata is malformed."""


@dataclass(frozen=True)
class CacheAPIError:
    status_code: int
    payload: dict[str, object]


def normalize_speech_cache_mode(
    value: str | None,
    default: str = CACHE_MODE_OFF,
) -> str:
    normalized_default = _normalize_speech_cache_mode_value(default, "default")
    if value is None:
        return normalized_default
    return _normalize_speech_cache_mode_value(value, "cache_mode")


def _normalize_speech_cache_mode_value(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise CacheValidationError(f"{name} must be a string")

    normalized = value.strip().lower()
    if normalized not in ALLOWED_SPEECH_CACHE_MODES:
        allowed = ", ".join(sorted(ALLOWED_SPEECH_CACHE_MODES))
        raise CacheValidationError(f"{name} must be one of: {allowed}")
    return normalized


def cache_exception_to_http_error(
    exc: Exception,
    *,
    cache_id: str | None = None,
    cache_mode: str | None = None,
) -> CacheAPIError:
    if isinstance(exc, CacheValidationError):
        status_code = 400
        error_type = "cache_validation_error"
    elif isinstance(exc, CacheNotFoundError):
        status_code = 404
        error_type = "cache_not_found"
    elif isinstance(exc, CacheConflictError):
        status_code = 409
        error_type = "cache_mismatch"
    elif isinstance(exc, CacheExpiredError):
        status_code = 410
        error_type = "cache_expired"
    else:
        status_code = 500
        error_type = "cache_internal_error"

    error: dict[str, object] = {
        "type": error_type,
        "message": str(exc),
    }
    if cache_id is not None:
        error["cache_id"] = cache_id
    if cache_mode is not None:
        error["cache_mode"] = cache_mode
    return CacheAPIError(status_code=status_code, payload={"error": error})


def cache_datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _normalise_datetime(value).isoformat().replace("+00:00", "Z")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")


def _validate_request_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheValidationError(f"{name} must be a positive int")


def _validate_request_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheValidationError(f"{name} must be a non-negative int")


def _validate_optional_request_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    _validate_request_positive_int(name, value)


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise CacheValidationError(f"{name} must be a bool")


def _validate_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CacheValidationError(f"{name} must be a non-empty string")


def _validate_ttl_seconds(ttl_seconds: float | None) -> None:
    if ttl_seconds is None:
        return
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int | float)
        or not math.isfinite(ttl_seconds)
        or ttl_seconds <= 0
    ):
        raise CacheValidationError("ttl_seconds must be positive when supplied")


def _validate_metadata(metadata: Mapping[str, object] | None) -> None:
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise CacheValidationError("metadata must be a mapping when supplied")
    for key in metadata:
        if not isinstance(key, str):
            raise CacheValidationError("metadata keys must be strings")


def _normalise_string_tuple(name: str, value: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise CacheValidationError(f"{name} must be a tuple or list")
    normalised = tuple(value)
    for item in normalised:
        _validate_non_empty_string(name, item)
    return normalised


def _copy_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    _validate_metadata(metadata)
    if metadata is None:
        return {}
    return dict(metadata)


@dataclass(frozen=True)
class CoreMLConditionBucket:
    sequence_length: int
    text_len: int
    speaker_context_len_bucket: int

    def __post_init__(self) -> None:
        _validate_positive_int("sequence_length", self.sequence_length)
        _validate_positive_int("text_len", self.text_len)
        _validate_positive_int(
            "speaker_context_len_bucket",
            self.speaker_context_len_bucket,
        )

    @property
    def c_ctx_bucket(self) -> int:
        return self.text_len + self.speaker_context_len_bucket

    def bucket_id(self, branch_layout: str) -> str:
        branch_count_for_layout(branch_layout)
        return (
            f"S{self.sequence_length}_T{self.text_len}_"
            f"R{self.speaker_context_len_bucket}_{branch_layout}"
        )


def expected_per_layer_state_names(num_layers: int = DEFAULT_NUM_LAYERS) -> tuple[str, ...]:
    _validate_positive_int("num_layers", num_layers)

    state_names: list[str] = []
    for layer_index in range(num_layers):
        state_names.append(f"context_k_l{layer_index:02d}")
        state_names.append(f"context_v_l{layer_index:02d}")
    state_names.append("valid_mask_state")
    return tuple(state_names)


def per_layer_state_shapes(
    bucket: CoreMLConditionBucket,
    branch_layout: str = BRANCH_LAYOUT_COND1,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> dict[str, tuple[int, ...]]:
    batch_size = branch_count_for_layout(branch_layout)
    _validate_positive_int("num_layers", num_layers)
    _validate_positive_int("num_heads", num_heads)
    _validate_positive_int("head_dim", head_dim)

    kv_shape = (batch_size, bucket.c_ctx_bucket, num_heads, head_dim)
    shapes: dict[str, tuple[int, ...]] = {}
    for layer_index in range(num_layers):
        shapes[f"context_k_l{layer_index:02d}"] = kv_shape
        shapes[f"context_v_l{layer_index:02d}"] = kv_shape
    shapes["valid_mask_state"] = (batch_size, bucket.c_ctx_bucket)
    return shapes


def kv_bytes_per_branch(
    bucket: CoreMLConditionBucket,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    _validate_positive_int("num_layers", num_layers)
    _validate_positive_int("num_heads", num_heads)
    _validate_positive_int("head_dim", head_dim)
    _validate_positive_int("dtype_bytes", dtype_bytes)

    return 2 * num_layers * bucket.c_ctx_bucket * num_heads * head_dim * dtype_bytes


def branch_count_for_layout(branch_layout: str) -> int:
    if branch_layout == BRANCH_LAYOUT_COND1:
        return 1
    if branch_layout == BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3:
        return 3
    raise ValueError(f"unknown branch_layout: {branch_layout}")


def kv_memory_bytes(
    bucket: CoreMLConditionBucket,
    branch_layouts: tuple[str, ...] | list[str],
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    return sum(
        branch_count_for_layout(branch_layout)
        * kv_bytes_per_branch(
            bucket,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
        )
        for branch_layout in branch_layouts
    )


def estimate_cfg_active_steps(
    num_steps: int,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    init_scale: float = 0.999,
) -> int:
    _validate_positive_int("num_steps", num_steps)
    if (
        not math.isfinite(cfg_min_t)
        or not math.isfinite(cfg_max_t)
        or cfg_min_t < 0.0
        or cfg_max_t > 1.0
        or cfg_min_t > cfg_max_t
    ):
        raise ValueError("cfg_min_t and cfg_max_t must satisfy 0.0 <= min <= max <= 1.0")
    if not math.isfinite(init_scale) or init_scale <= 0.0:
        raise ValueError("init_scale must be positive")

    active_steps = 0
    for step_index in range(num_steps):
        t = (1.0 - step_index / num_steps) * init_scale
        if cfg_min_t <= t <= cfg_max_t:
            active_steps += 1
    return active_steps


def _normalise_branch_layouts(branch_layouts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(branch_layouts, tuple | list) or not branch_layouts:
        raise CacheValidationError("branch_layouts must be a non-empty tuple or list")

    normalised = tuple(branch_layouts)
    for branch_layout in normalised:
        _validate_non_empty_string("branch_layout", branch_layout)
        try:
            branch_count_for_layout(branch_layout)
        except ValueError as exc:
            raise CacheValidationError(str(exc)) from exc
    if normalised not in ALLOWED_CONDITION_BRANCH_LAYOUTS:
        raise CacheValidationError("branch_layouts must be a supported canonical layout tuple")
    return normalised


def _content_id(prefix: str, payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:32]}"


def _normalise_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise CacheValidationError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _expires_at(created_at: datetime, ttl_seconds: float | None) -> datetime | None:
    if ttl_seconds is None:
        return None
    return created_at + timedelta(seconds=float(ttl_seconds))


@dataclass(frozen=True)
class ReferenceCacheRequest:
    model_fingerprint: str
    codec_fingerprint: str
    reference_fingerprint: str
    speaker_context_len: int
    memory_bytes: int = 0
    ttl_seconds: float | None = None
    metadata: Mapping[str, object] | None = None
    model: str | None = None
    ref_len: int | None = None
    speaker_dim: int | None = None
    memory_bytes_estimated: bool = True
    resident_layers: tuple[str, ...] | list[str] = ("ref_latent", "speaker_state")
    resident_buckets: tuple[str, ...] | list[str] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_string("model_fingerprint", self.model_fingerprint)
        _validate_non_empty_string("codec_fingerprint", self.codec_fingerprint)
        _validate_non_empty_string("reference_fingerprint", self.reference_fingerprint)
        if self.model is not None:
            _validate_non_empty_string("model", self.model)
        _validate_request_positive_int("speaker_context_len", self.speaker_context_len)
        _validate_optional_request_positive_int("ref_len", self.ref_len)
        _validate_optional_request_positive_int("speaker_dim", self.speaker_dim)
        _validate_request_non_negative_int("memory_bytes", self.memory_bytes)
        _validate_bool("memory_bytes_estimated", self.memory_bytes_estimated)
        _validate_ttl_seconds(self.ttl_seconds)
        _validate_metadata(self.metadata)
        object.__setattr__(
            self,
            "resident_layers",
            _normalise_string_tuple("resident_layers", self.resident_layers),
        )
        object.__setattr__(
            self,
            "resident_buckets",
            _normalise_string_tuple("resident_buckets", self.resident_buckets),
        )


@dataclass(kw_only=True)
class ReferenceCacheHandle:
    id: str
    status: str = "ready"
    model_fingerprint: str
    codec_fingerprint: str
    reference_fingerprint: str
    speaker_context_len: int
    model: str | None = None
    ref_len: int | None = None
    speaker_dim: int | None = None
    created_at: datetime
    expires_at: datetime | None
    memory_bytes: int
    memory_bytes_estimated: bool = True
    resident_layers: tuple[str, ...] = ("ref_latent", "speaker_state")
    resident_buckets: tuple[str, ...] = ()
    metadata: dict[str, object]
    hit_count: int = 0
    last_used_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.model is not None:
            _validate_non_empty_string("model", self.model)
        _validate_optional_request_positive_int("ref_len", self.ref_len)
        _validate_optional_request_positive_int("speaker_dim", self.speaker_dim)
        _validate_bool("memory_bytes_estimated", self.memory_bytes_estimated)
        self.resident_layers = _normalise_string_tuple(
            "resident_layers",
            self.resident_layers,
        )
        self.resident_buckets = _normalise_string_tuple(
            "resident_buckets",
            self.resident_buckets,
        )


@dataclass(frozen=True)
class ConditionCacheRequest:
    reference_cache_id: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    condition_fingerprint: str
    bucket: CoreMLConditionBucket
    speaker_context_len: int
    branch_layouts: tuple[str, ...] | list[str]
    ttl_seconds: float | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string("reference_cache_id", self.reference_cache_id)
        _validate_non_empty_string("model_fingerprint", self.model_fingerprint)
        _validate_non_empty_string("tokenizer_fingerprint", self.tokenizer_fingerprint)
        _validate_non_empty_string("condition_fingerprint", self.condition_fingerprint)
        if not isinstance(self.bucket, CoreMLConditionBucket):
            raise CacheValidationError("bucket must be a CoreMLConditionBucket")
        _validate_request_positive_int("speaker_context_len", self.speaker_context_len)
        if self.speaker_context_len > self.bucket.speaker_context_len_bucket:
            raise CacheValidationError(
                "speaker_context_len must not exceed speaker_context_len_bucket",
            )
        object.__setattr__(
            self,
            "branch_layouts",
            _normalise_branch_layouts(self.branch_layouts),
        )
        _validate_ttl_seconds(self.ttl_seconds)
        _validate_metadata(self.metadata)


@dataclass(kw_only=True)
class ConditionCacheHandle:
    id: str
    status: str = "ready"
    reference_cache_id: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    condition_fingerprint: str
    bucket_id: str
    state_layout: str = STATE_LAYOUT_PER_LAYER
    sequence_length: int
    text_len: int
    speaker_context_len: int
    speaker_context_len_bucket: int
    c_ctx_bucket: int
    branch_layouts: tuple[str, ...]
    mlstate_keys: tuple[str, ...]
    created_at: datetime
    expires_at: datetime | None
    memory_bytes: int
    metadata: dict[str, object]
    hit_count: int = 0
    last_used_at: datetime | None = None


@dataclass(frozen=True)
class CacheCreateResult:
    handle: ReferenceCacheHandle | ConditionCacheHandle
    reused: bool


def reference_cache_response(
    handle_or_result: ReferenceCacheHandle | CacheCreateResult,
    *,
    include_reused: bool = True,
) -> dict[str, object]:
    if isinstance(handle_or_result, CacheCreateResult):
        return reference_cache_create_response(
            handle_or_result,
            include_reused=include_reused,
        )
    return reference_cache_get_response(handle_or_result)


def reference_cache_create_response(
    result: CacheCreateResult,
    *,
    include_reused: bool = True,
) -> dict[str, object]:
    if not isinstance(result, CacheCreateResult) or not isinstance(
        result.handle,
        ReferenceCacheHandle,
    ):
        raise CacheValidationError("result must contain a reference cache handle")

    handle = result.handle
    response = _reference_cache_common_response(handle)
    response.update(
        {
            "model": handle.model,
            "layers": _reference_layers_response(handle),
            "shapes": _reference_shapes_response(handle),
            "memory_bytes_estimated": handle.memory_bytes_estimated,
        },
    )
    if include_reused:
        response["reused"] = result.reused
    return response


def reference_cache_get_response(handle: ReferenceCacheHandle) -> dict[str, object]:
    if not isinstance(handle, ReferenceCacheHandle):
        raise CacheValidationError("handle must be a reference cache handle")

    response = _reference_cache_common_response(handle)
    response.update(
        {
            "model": handle.model,
            "shapes": _reference_shapes_response(handle),
            "memory_bytes_estimated": handle.memory_bytes_estimated,
            "resident_layers": list(handle.resident_layers),
            "resident_buckets": list(handle.resident_buckets),
        },
    )
    return response


def condition_cache_response(
    handle_or_result: ConditionCacheHandle | CacheCreateResult,
    *,
    include_reused: bool = True,
) -> dict[str, object]:
    if isinstance(handle_or_result, CacheCreateResult):
        return condition_cache_create_response(
            handle_or_result,
            include_reused=include_reused,
        )
    return condition_cache_get_response(handle_or_result)


def condition_cache_create_response(
    result: CacheCreateResult,
    *,
    include_reused: bool = True,
) -> dict[str, object]:
    if not isinstance(result, CacheCreateResult) or not isinstance(
        result.handle,
        ConditionCacheHandle,
    ):
        raise CacheValidationError("result must contain a condition cache handle")

    handle = result.handle
    branch_layouts = _normalise_branch_layouts(handle.branch_layouts)
    response: dict[str, object] = {
        "id": handle.id,
        "status": handle.status,
        "reference_cache_id": handle.reference_cache_id,
        "model_fingerprint": handle.model_fingerprint,
        "tokenizer_fingerprint": handle.tokenizer_fingerprint,
        "condition_fingerprint": handle.condition_fingerprint,
        "bucket_id": handle.bucket_id,
        "state_layout": handle.state_layout,
        "branch_layouts": _condition_branch_layouts_response(branch_layouts),
        "shapes": {
            "sequence_length": handle.sequence_length,
            "text_len": handle.text_len,
            "speaker_context_len": handle.speaker_context_len,
            "speaker_context_len_bucket": handle.speaker_context_len_bucket,
            "c_ctx_bucket": handle.c_ctx_bucket,
            "branches_active": branch_count_for_layout(branch_layouts[-1]),
        },
        "memory_bytes": handle.memory_bytes,
        "expires_at": cache_datetime_to_iso(handle.expires_at),
        "created_at": cache_datetime_to_iso(handle.created_at),
        "last_used_at": cache_datetime_to_iso(handle.last_used_at),
        "hit_count": handle.hit_count,
        "metadata": _json_safe_mapping(handle.metadata),
    }
    if include_reused:
        response["reused"] = result.reused
    return response


def condition_cache_get_response(handle: ConditionCacheHandle) -> dict[str, object]:
    if not isinstance(handle, ConditionCacheHandle):
        raise CacheValidationError("handle must be a condition cache handle")

    branch_layouts = _normalise_branch_layouts(handle.branch_layouts)
    response: dict[str, object] = {
        "id": handle.id,
        "status": handle.status,
        "reference_cache_id": handle.reference_cache_id,
        "bucket_id": handle.bucket_id,
        "cfg": {
            "mode": _condition_cfg_mode(branch_layouts),
            "active_steps_estimate": None,
            "branches_active": branch_count_for_layout(branch_layouts[-1]),
        },
        "resident_states": list(branch_layouts),
        "created_at": cache_datetime_to_iso(handle.created_at),
        "last_used_at": cache_datetime_to_iso(handle.last_used_at),
        "hit_count": handle.hit_count,
        "metadata": _json_safe_mapping(handle.metadata),
    }
    return response


def cache_create_status_code(result: CacheCreateResult) -> int:
    if not isinstance(result, CacheCreateResult):
        raise CacheValidationError("result must be a CacheCreateResult")
    return 200 if result.reused else 201


def _reference_cache_common_response(handle: ReferenceCacheHandle) -> dict[str, object]:
    return {
        "id": handle.id,
        "status": handle.status,
        "model_fingerprint": handle.model_fingerprint,
        "codec_fingerprint": handle.codec_fingerprint,
        "reference_fingerprint": handle.reference_fingerprint,
        "memory_bytes": handle.memory_bytes,
        "expires_at": cache_datetime_to_iso(handle.expires_at),
        "created_at": cache_datetime_to_iso(handle.created_at),
        "last_used_at": cache_datetime_to_iso(handle.last_used_at),
        "hit_count": handle.hit_count,
        "metadata": _json_safe_mapping(handle.metadata),
    }


def _reference_shapes_response(handle: ReferenceCacheHandle) -> dict[str, int | None]:
    return {
        "ref_len": handle.ref_len,
        "speaker_context_len": handle.speaker_context_len,
        "speaker_dim": handle.speaker_dim,
    }


def _reference_layers_response(handle: ReferenceCacheHandle) -> dict[str, object]:
    layers: dict[str, object] = dict.fromkeys(handle.resident_layers, True)
    layers.setdefault("speaker_kv", "lazy")
    return layers


def _condition_branch_layouts_response(branch_layouts: tuple[str, ...]) -> dict[str, str]:
    if branch_layouts == (BRANCH_LAYOUT_COND1,):
        return {"cond": BRANCH_LAYOUT_COND1}
    if branch_layouts == (BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3):
        return {
            "cond": BRANCH_LAYOUT_COND1,
            "cfg_active": BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
        }
    raise CacheValidationError("branch_layouts must be a supported canonical layout tuple")


def _condition_cfg_mode(branch_layouts: tuple[str, ...]) -> str:
    if BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3 in branch_layouts:
        return "independent"
    return "cond"


def _json_safe_mapping(metadata: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_safe_value(value) for key, value in metadata.items()}


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, datetime):
        return cache_datetime_to_iso(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _reference_cache_id(request: ReferenceCacheRequest) -> str:
    return _content_id(
        "ref",
        {
            "kind": "reference_cache_v1",
            "model_fingerprint": request.model_fingerprint,
            "codec_fingerprint": request.codec_fingerprint,
            "reference_fingerprint": request.reference_fingerprint,
            "speaker_context_len": request.speaker_context_len,
        },
    )


def _condition_cache_id(request: ConditionCacheRequest) -> str:
    return _content_id(
        "cond",
        {
            "kind": "condition_cache_v1",
            "reference_cache_id": request.reference_cache_id,
            "model_fingerprint": request.model_fingerprint,
            "tokenizer_fingerprint": request.tokenizer_fingerprint,
            "condition_fingerprint": request.condition_fingerprint,
            "sequence_length": request.bucket.sequence_length,
            "text_len": request.bucket.text_len,
            "speaker_context_len": request.speaker_context_len,
            "speaker_context_len_bucket": request.bucket.speaker_context_len_bucket,
            "branch_layouts": request.branch_layouts,
        },
    )


class InMemoryCoreMLCacheManager:
    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._reference_caches: dict[str, ReferenceCacheHandle] = {}
        self._condition_caches: dict[str, ConditionCacheHandle] = {}

    def prepare_reference_cache(
        self,
        request: ReferenceCacheRequest,
        cache_mode: str = "create_or_reuse",
    ) -> CacheCreateResult:
        self._validate_reference_request(request)
        self._validate_cache_mode(cache_mode)

        now = self._now()
        cache_id = _reference_cache_id(request)
        existing = self._reference_caches.get(cache_id)
        if (
            cache_mode == "create_or_reuse"
            and existing is not None
            and not self._is_expired(existing, now)
        ):
            return CacheCreateResult(handle=existing, reused=True)

        if existing is not None:
            self._delete_conditions_for_reference(cache_id)

        handle = ReferenceCacheHandle(
            id=cache_id,
            model_fingerprint=request.model_fingerprint,
            codec_fingerprint=request.codec_fingerprint,
            reference_fingerprint=request.reference_fingerprint,
            speaker_context_len=request.speaker_context_len,
            model=request.model,
            ref_len=request.ref_len,
            speaker_dim=request.speaker_dim,
            created_at=now,
            expires_at=_expires_at(now, request.ttl_seconds),
            memory_bytes=request.memory_bytes,
            memory_bytes_estimated=request.memory_bytes_estimated,
            resident_layers=tuple(request.resident_layers),
            resident_buckets=tuple(request.resident_buckets),
            metadata=_copy_metadata(request.metadata),
        )
        self._reference_caches[cache_id] = handle
        return CacheCreateResult(handle=handle, reused=False)

    def prepare_condition_cache(
        self,
        request: ConditionCacheRequest,
        cache_mode: str = "create_or_reuse",
    ) -> CacheCreateResult:
        self._validate_condition_request(request)
        self._validate_cache_mode(cache_mode)

        now = self._now()
        reference = self._reference_caches.get(request.reference_cache_id)
        if reference is None:
            raise CacheNotFoundError(f"reference cache not found: {request.reference_cache_id}")
        self._raise_if_expired(reference, now, "reference")
        if request.model_fingerprint != reference.model_fingerprint:
            raise CacheConflictError("model_fingerprint conflicts with reference cache")
        if request.speaker_context_len != reference.speaker_context_len:
            raise CacheConflictError("speaker_context_len conflicts with reference cache")

        cache_id = _condition_cache_id(request)
        existing = self._condition_caches.get(cache_id)
        if (
            cache_mode == "create_or_reuse"
            and existing is not None
            and not self._is_expired(existing, now)
        ):
            self._record_resident_bucket(reference, existing.bucket_id)
            return CacheCreateResult(handle=existing, reused=True)

        branch_layouts = tuple(request.branch_layouts)
        handle = ConditionCacheHandle(
            id=cache_id,
            reference_cache_id=request.reference_cache_id,
            model_fingerprint=request.model_fingerprint,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
            condition_fingerprint=request.condition_fingerprint,
            bucket_id=request.bucket.bucket_id(branch_layouts[-1]),
            sequence_length=request.bucket.sequence_length,
            text_len=request.bucket.text_len,
            speaker_context_len=request.speaker_context_len,
            speaker_context_len_bucket=request.bucket.speaker_context_len_bucket,
            c_ctx_bucket=request.bucket.c_ctx_bucket,
            branch_layouts=branch_layouts,
            mlstate_keys=expected_per_layer_state_names(),
            created_at=now,
            expires_at=_expires_at(now, request.ttl_seconds),
            memory_bytes=kv_memory_bytes(request.bucket, branch_layouts),
            metadata=_copy_metadata(request.metadata),
        )
        self._condition_caches[cache_id] = handle
        self._record_resident_bucket(reference, handle.bucket_id)
        return CacheCreateResult(handle=handle, reused=False)

    def get_reference_cache(self, cache_id: str) -> ReferenceCacheHandle:
        _validate_non_empty_string("cache_id", cache_id)
        now = self._now()
        handle = self._reference_caches.get(cache_id)
        if handle is None:
            raise CacheNotFoundError(f"reference cache not found: {cache_id}")
        self._raise_if_expired(handle, now, "reference")
        self._record_hit(handle, now)
        return handle

    def get_condition_cache(self, cache_id: str) -> ConditionCacheHandle:
        _validate_non_empty_string("cache_id", cache_id)
        now = self._now()
        handle = self._condition_caches.get(cache_id)
        if handle is None:
            raise CacheNotFoundError(f"condition cache not found: {cache_id}")
        self._raise_if_expired(handle, now, "condition")
        reference = self._reference_caches.get(handle.reference_cache_id)
        if reference is None:
            raise CacheNotFoundError(f"reference cache not found: {handle.reference_cache_id}")
        self._raise_if_expired(reference, now, "reference")
        self._record_hit(handle, now)
        return handle

    def delete_reference_cache(self, cache_id: str, cascade: bool = True) -> bool:
        _validate_non_empty_string("cache_id", cache_id)
        if cache_id not in self._reference_caches:
            return False

        dependent_condition_ids = self._condition_ids_for_reference(cache_id)
        if dependent_condition_ids and not cascade:
            raise CacheConflictError("reference cache has dependent condition caches")

        for condition_id in dependent_condition_ids:
            del self._condition_caches[condition_id]
        del self._reference_caches[cache_id]
        return True

    def delete_condition_cache(self, cache_id: str) -> bool:
        _validate_non_empty_string("cache_id", cache_id)
        handle = self._condition_caches.get(cache_id)
        if handle is None:
            return False
        del self._condition_caches[cache_id]
        self._prune_resident_bucket_if_unused(handle.reference_cache_id, handle.bucket_id)
        return True

    def _prune_resident_bucket_if_unused(self, reference_cache_id: str, bucket_id: str) -> None:
        reference = self._reference_caches.get(reference_cache_id)
        if reference is None:
            return
        if any(
            condition.reference_cache_id == reference_cache_id and condition.bucket_id == bucket_id
            for condition in self._condition_caches.values()
        ):
            return
        reference.resident_buckets = tuple(
            resident_bucket
            for resident_bucket in reference.resident_buckets
            if resident_bucket != bucket_id
        )

    def require_condition_cache(
        self,
        cache_id: str,
        expected_request: ConditionCacheRequest,
    ) -> ConditionCacheHandle:
        _validate_non_empty_string("cache_id", cache_id)
        self._validate_condition_request(expected_request)

        now = self._now()
        handle = self._condition_caches.get(cache_id)
        if handle is None:
            raise CacheNotFoundError(f"condition cache not found: {cache_id}")
        self._raise_if_expired(handle, now, "condition")

        reference = self._reference_caches.get(handle.reference_cache_id)
        if reference is None:
            raise CacheNotFoundError(f"reference cache not found: {handle.reference_cache_id}")
        self._raise_if_expired(reference, now, "reference")

        self._raise_condition_conflict(
            "reference_cache_id",
            handle.reference_cache_id,
            expected_request.reference_cache_id,
        )
        self._raise_condition_conflict(
            "model_fingerprint",
            handle.model_fingerprint,
            expected_request.model_fingerprint,
        )
        self._raise_condition_conflict(
            "tokenizer_fingerprint",
            handle.tokenizer_fingerprint,
            expected_request.tokenizer_fingerprint,
        )
        self._raise_condition_conflict(
            "condition_fingerprint",
            handle.condition_fingerprint,
            expected_request.condition_fingerprint,
        )
        self._raise_condition_conflict(
            "sequence_length",
            handle.sequence_length,
            expected_request.bucket.sequence_length,
            "bucket",
        )
        self._raise_condition_conflict(
            "text_len",
            handle.text_len,
            expected_request.bucket.text_len,
            "bucket",
        )
        self._raise_condition_conflict(
            "speaker_context_len_bucket",
            handle.speaker_context_len_bucket,
            expected_request.bucket.speaker_context_len_bucket,
            "bucket",
        )
        self._raise_condition_conflict(
            "speaker_context_len",
            handle.speaker_context_len,
            expected_request.speaker_context_len,
        )
        self._raise_condition_conflict(
            "branch_layouts",
            handle.branch_layouts,
            tuple(expected_request.branch_layouts),
        )
        self._raise_condition_conflict(
            "state_layout",
            handle.state_layout,
            STATE_LAYOUT_PER_LAYER,
        )
        return handle

    def prune_expired(self) -> dict[str, int]:
        now = self._now()
        expired_reference_ids = {
            cache_id
            for cache_id, handle in self._reference_caches.items()
            if self._is_expired(handle, now)
        }
        expired_condition_ids = {
            cache_id
            for cache_id, handle in self._condition_caches.items()
            if self._is_expired(handle, now) or handle.reference_cache_id in expired_reference_ids
        }
        expired_condition_handles = {
            cache_id: self._condition_caches[cache_id] for cache_id in expired_condition_ids
        }

        for cache_id in expired_condition_ids:
            del self._condition_caches[cache_id]
        for handle in expired_condition_handles.values():
            if handle.reference_cache_id not in expired_reference_ids:
                self._prune_resident_bucket_if_unused(
                    handle.reference_cache_id,
                    handle.bucket_id,
                )
        for cache_id in expired_reference_ids:
            del self._reference_caches[cache_id]

        return {
            "reference": len(expired_reference_ids),
            "condition": len(expired_condition_ids),
        }

    def _now(self) -> datetime:
        return _normalise_datetime(self._clock())

    @staticmethod
    def _validate_cache_mode(cache_mode: str) -> None:
        if cache_mode not in {"create_or_reuse", "refresh"}:
            raise CacheValidationError("cache_mode must be create_or_reuse or refresh")

    @staticmethod
    def _validate_reference_request(request: ReferenceCacheRequest) -> None:
        if not isinstance(request, ReferenceCacheRequest):
            raise CacheValidationError("request must be a ReferenceCacheRequest")

    @staticmethod
    def _validate_condition_request(request: ConditionCacheRequest) -> None:
        if not isinstance(request, ConditionCacheRequest):
            raise CacheValidationError("request must be a ConditionCacheRequest")

    @staticmethod
    def _is_expired(
        handle: ReferenceCacheHandle | ConditionCacheHandle,
        now: datetime,
    ) -> bool:
        return handle.expires_at is not None and now >= handle.expires_at

    def _raise_if_expired(
        self,
        handle: ReferenceCacheHandle | ConditionCacheHandle,
        now: datetime,
        cache_type: str,
    ) -> None:
        if self._is_expired(handle, now):
            raise CacheExpiredError(f"{cache_type} cache expired: {handle.id}")

    @staticmethod
    def _record_hit(
        handle: ReferenceCacheHandle | ConditionCacheHandle,
        now: datetime,
    ) -> None:
        handle.hit_count += 1
        handle.last_used_at = now

    @staticmethod
    def _record_resident_bucket(handle: ReferenceCacheHandle, bucket_id: str) -> None:
        if bucket_id not in handle.resident_buckets:
            handle.resident_buckets = (*handle.resident_buckets, bucket_id)

    def _condition_ids_for_reference(self, reference_cache_id: str) -> list[str]:
        return [
            cache_id
            for cache_id, handle in self._condition_caches.items()
            if handle.reference_cache_id == reference_cache_id
        ]

    def _delete_conditions_for_reference(self, reference_cache_id: str) -> None:
        for cache_id in self._condition_ids_for_reference(reference_cache_id):
            del self._condition_caches[cache_id]

    @staticmethod
    def _raise_condition_conflict(
        field_name: str,
        actual: object,
        expected: object,
        message_field_name: str | None = None,
    ) -> None:
        if actual != expected:
            label = message_field_name or field_name
            raise CacheConflictError(f"{label} conflicts with condition cache")
