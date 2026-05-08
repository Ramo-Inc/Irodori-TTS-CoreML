from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "irodori_tts" / "coreml_cache.py"
MODULE_SPEC = importlib.util.spec_from_file_location("coreml_cache_api_schema", MODULE_PATH)
assert MODULE_SPEC is not None
coreml_cache = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = coreml_cache
MODULE_SPEC.loader.exec_module(coreml_cache)


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


def reference_request(**overrides: object) -> coreml_cache.ReferenceCacheRequest:
    kwargs: dict[str, object] = {
        "model_fingerprint": "model-a",
        "codec_fingerprint": "codec-a",
        "reference_fingerprint": "reference-a",
        "speaker_context_len": 120,
    }
    kwargs.update(overrides)
    return coreml_cache.ReferenceCacheRequest(**kwargs)


def condition_bucket() -> coreml_cache.CoreMLConditionBucket:
    return coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )


def condition_request(
    reference_cache_id: str,
    **overrides: object,
) -> coreml_cache.ConditionCacheRequest:
    kwargs: dict[str, object] = {
        "reference_cache_id": reference_cache_id,
        "model_fingerprint": "model-a",
        "tokenizer_fingerprint": "tokenizer-a",
        "condition_fingerprint": "condition-a",
        "bucket": condition_bucket(),
        "speaker_context_len": 120,
        "branch_layouts": (
            coreml_cache.BRANCH_LAYOUT_COND1,
            coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
        ),
    }
    kwargs.update(overrides)
    return coreml_cache.ConditionCacheRequest(**kwargs)


def assert_json_serializable(payload: dict[str, object]) -> None:
    json.dumps(payload)


def test_reference_cache_response_from_create_result_includes_api_fields() -> None:
    clock = Clock()
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=clock)
    result = manager.prepare_reference_cache(
        reference_request(
            model="irodori-tts-500m-v2",
            ref_len=137,
            speaker_dim=768,
            ttl_seconds=60,
            memory_bytes=512,
            memory_bytes_estimated=True,
            metadata={"speaker": "alpha", "tags": ("warm", "dry")},
        ),
    )
    clock.advance(5)
    manager.get_reference_cache(result.handle.id)

    response = coreml_cache.reference_cache_response(result)

    assert response == {
        "id": result.handle.id,
        "status": "ready",
        "model_fingerprint": "model-a",
        "codec_fingerprint": "codec-a",
        "reference_fingerprint": "reference-a",
        "memory_bytes": 512,
        "expires_at": "2026-01-01T00:01:00Z",
        "created_at": "2026-01-01T00:00:00Z",
        "last_used_at": "2026-01-01T00:00:05Z",
        "hit_count": 1,
        "metadata": {"speaker": "alpha", "tags": ["warm", "dry"]},
        "model": "irodori-tts-500m-v2",
        "layers": {
            "ref_latent": True,
            "speaker_state": True,
            "speaker_kv": "lazy",
        },
        "shapes": {
            "ref_len": 137,
            "speaker_context_len": 120,
            "speaker_dim": 768,
        },
        "memory_bytes_estimated": True,
        "reused": False,
    }
    assert_json_serializable(response)


def test_bare_reference_cache_response_uses_get_schema_and_tracks_buckets() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    result = manager.prepare_reference_cache(
        reference_request(
            model="irodori-tts-500m-v2",
            ref_len=137,
            speaker_dim=768,
            memory_bytes=512,
        ),
    )
    condition = manager.prepare_condition_cache(condition_request(result.handle.id)).handle
    manager.prepare_condition_cache(condition_request(result.handle.id))

    response = coreml_cache.reference_cache_response(result.handle)

    assert "reused" not in response
    assert "layers" not in response
    assert response == {
        "id": result.handle.id,
        "status": "ready",
        "model_fingerprint": "model-a",
        "codec_fingerprint": "codec-a",
        "reference_fingerprint": "reference-a",
        "memory_bytes": 512,
        "expires_at": None,
        "created_at": "2026-01-01T00:00:00Z",
        "last_used_at": None,
        "hit_count": 0,
        "metadata": {},
        "model": "irodori-tts-500m-v2",
        "shapes": {
            "ref_len": 137,
            "speaker_context_len": 120,
            "speaker_dim": 768,
        },
        "memory_bytes_estimated": True,
        "resident_layers": ["ref_latent", "speaker_state"],
        "resident_buckets": [condition.bucket_id],
    }
    assert_json_serializable(response)


def test_condition_cache_create_response_includes_branch_and_shape_fields() -> None:
    clock = Clock()
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=clock)
    reference = manager.prepare_reference_cache(reference_request()).handle
    result = manager.prepare_condition_cache(
        condition_request(
            reference.id,
            ttl_seconds=60,
            metadata={"route": "speech", "priority": 1},
        ),
    )
    clock.advance(7)
    manager.get_condition_cache(result.handle.id)

    response = coreml_cache.condition_cache_response(result)

    assert response["id"] == result.handle.id
    assert response["status"] == "ready"
    assert response["reference_cache_id"] == reference.id
    assert response["model_fingerprint"] == "model-a"
    assert response["tokenizer_fingerprint"] == "tokenizer-a"
    assert response["condition_fingerprint"] == "condition-a"
    assert response["bucket_id"] == "S100_T256_R160_independent_text_speaker3"
    assert response["state_layout"] == "per_layer_text_speaker_context_v1"
    assert response["branch_layouts"] == {
        "cond": "cond1",
        "cfg_active": "independent_text_speaker3",
    }
    assert response["shapes"] == {
        "sequence_length": 100,
        "text_len": 256,
        "speaker_context_len": 120,
        "speaker_context_len_bucket": 160,
        "c_ctx_bucket": 416,
        "branches_active": 3,
    }
    assert response["memory_bytes"] == 102_236_160
    assert response["expires_at"] == "2026-01-01T00:01:00Z"
    assert response["created_at"] == "2026-01-01T00:00:00Z"
    assert response["last_used_at"] == "2026-01-01T00:00:07Z"
    assert response["hit_count"] == 1
    assert response["metadata"] == {"route": "speech", "priority": 1}
    assert response["reused"] is False
    assert_json_serializable(response)


def test_cond1_only_condition_cache_response_has_one_active_branch() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    result = manager.prepare_condition_cache(
        condition_request(
            reference.id,
            branch_layouts=(coreml_cache.BRANCH_LAYOUT_COND1,),
            condition_fingerprint="condition-cond1",
        ),
    )

    response = coreml_cache.condition_cache_response(result.handle)

    assert response == {
        "id": result.handle.id,
        "status": "ready",
        "reference_cache_id": reference.id,
        "bucket_id": "S100_T256_R160_cond1",
        "cfg": {
            "mode": "cond",
            "active_steps_estimate": None,
            "branches_active": 1,
        },
        "resident_states": ["cond1"],
        "created_at": "2026-01-01T00:00:00Z",
        "last_used_at": None,
        "hit_count": 0,
        "metadata": {},
    }
    assert "reused" not in response
    assert "branch_layouts" not in response
    assert "shapes" not in response
    assert "memory_bytes" not in response
    assert_json_serializable(response)


def test_bare_condition_cache_response_uses_get_schema_for_independent_cfg() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    result = manager.prepare_condition_cache(
        condition_request(reference.id, metadata={"route": "speech"}),
    )

    response = coreml_cache.condition_cache_response(result.handle)

    assert response == {
        "id": result.handle.id,
        "status": "ready",
        "reference_cache_id": reference.id,
        "bucket_id": "S100_T256_R160_independent_text_speaker3",
        "cfg": {
            "mode": "independent",
            "active_steps_estimate": None,
            "branches_active": 3,
        },
        "resident_states": ["cond1", "independent_text_speaker3"],
        "created_at": "2026-01-01T00:00:00Z",
        "last_used_at": None,
        "hit_count": 0,
        "metadata": {"route": "speech"},
    }
    assert "reused" not in response
    assert "branch_layouts" not in response
    assert "shapes" not in response
    assert "memory_bytes" not in response
    assert_json_serializable(response)


def test_cache_create_status_code_distinguishes_created_and_reused() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    created = manager.prepare_reference_cache(reference_request())
    reused = manager.prepare_reference_cache(reference_request())

    assert coreml_cache.cache_create_status_code(created) == 201
    assert coreml_cache.cache_create_status_code(reused) == 200


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "off"),
        ("require", "require"),
        (" refresh ", "refresh"),
    ],
)
def test_normalize_speech_cache_mode_accepts_known_modes(
    value: str | None,
    expected: str,
) -> None:
    assert coreml_cache.normalize_speech_cache_mode(value) == expected


def test_normalize_speech_cache_mode_rejects_unknown_mode() -> None:
    with pytest.raises(coreml_cache.CacheValidationError, match="cache_mode"):
        coreml_cache.normalize_speech_cache_mode("reuse")


@pytest.mark.parametrize("value", ["", "   "])
def test_normalize_speech_cache_mode_rejects_blank_mode(value: str) -> None:
    with pytest.raises(coreml_cache.CacheValidationError, match="cache_mode"):
        coreml_cache.normalize_speech_cache_mode(value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ref_len": 0},
        {"speaker_dim": 0},
        {"memory_bytes_estimated": "yes"},
    ],
)
def test_reference_cache_request_rejects_invalid_api_metadata(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(coreml_cache.CacheValidationError):
        reference_request(**overrides)


@pytest.mark.parametrize(
    ("exc", "status_code", "error_type"),
    [
        (coreml_cache.CacheNotFoundError("missing cache"), 404, "cache_not_found"),
        (coreml_cache.CacheExpiredError("expired cache"), 410, "cache_expired"),
        (coreml_cache.CacheConflictError("mismatched cache"), 409, "cache_mismatch"),
        (
            coreml_cache.CacheValidationError("invalid cache request"),
            400,
            "cache_validation_error",
        ),
        (Exception("unexpected failure"), 500, "cache_internal_error"),
    ],
)
def test_cache_exception_to_http_error_mappings(
    exc: Exception,
    status_code: int,
    error_type: str,
) -> None:
    api_error = coreml_cache.cache_exception_to_http_error(
        exc,
        cache_id="cond_test",
        cache_mode="require",
    )

    assert api_error.status_code == status_code
    assert api_error.payload == {
        "error": {
            "type": error_type,
            "message": str(exc),
            "cache_id": "cond_test",
            "cache_mode": "require",
        },
    }
    assert_json_serializable(api_error.payload)
