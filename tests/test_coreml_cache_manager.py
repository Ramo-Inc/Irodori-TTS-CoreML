from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "irodori_tts" / "coreml_cache.py"
MODULE_SPEC = importlib.util.spec_from_file_location("coreml_cache_manager", MODULE_PATH)
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


def test_reference_create_reuse_refresh_and_ttl_expiry() -> None:
    clock = Clock()
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=clock)
    request = reference_request(ttl_seconds=10, memory_bytes=512, metadata={"speaker": "a"})

    created = manager.prepare_reference_cache(request)

    assert created.reused is False
    handle = created.handle
    assert handle.id.startswith("ref_")
    assert handle.created_at == clock.current
    assert handle.expires_at == clock.current + timedelta(seconds=10)
    assert handle.memory_bytes == 512
    assert handle.metadata == {"speaker": "a"}

    reused = manager.prepare_reference_cache(request)

    assert reused.reused is True
    assert reused.handle is handle
    assert handle.hit_count == 0
    assert handle.last_used_at is None

    clock.advance(10)
    with pytest.raises(coreml_cache.CacheExpiredError):
        manager.get_reference_cache(handle.id)

    recreated = manager.prepare_reference_cache(request)

    assert recreated.reused is False
    assert recreated.handle.id == handle.id
    assert recreated.handle is not handle
    assert recreated.handle.created_at == clock.current

    clock.advance(1)
    refreshed = manager.prepare_reference_cache(request, cache_mode="refresh")

    assert refreshed.reused is False
    assert refreshed.handle.id == handle.id
    assert refreshed.handle is not recreated.handle
    assert refreshed.handle.created_at == clock.current


def test_condition_create_requires_reference_and_populates_contract_fields() -> None:
    clock = Clock()
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=clock)

    with pytest.raises(coreml_cache.CacheNotFoundError):
        manager.prepare_condition_cache(condition_request("ref_missing"))

    reference = manager.prepare_reference_cache(reference_request()).handle
    request = condition_request(reference.id)

    created = manager.prepare_condition_cache(request)

    assert created.reused is False
    handle = created.handle
    assert handle.id.startswith("cond_")
    assert handle.reference_cache_id == reference.id
    assert handle.memory_bytes == 102_236_160
    assert handle.bucket_id == "S100_T256_R160_independent_text_speaker3"
    assert handle.state_layout == "per_layer_text_speaker_context_v1"
    assert handle.sequence_length == 100
    assert handle.text_len == 256
    assert handle.speaker_context_len == 120
    assert handle.speaker_context_len_bucket == 160
    assert handle.c_ctx_bucket == 416
    assert handle.branch_layouts == (
        coreml_cache.BRANCH_LAYOUT_COND1,
        coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    )
    assert handle.mlstate_keys == coreml_cache.expected_per_layer_state_names()

    reused = manager.prepare_condition_cache(request)

    assert reused.reused is True
    assert reused.handle is handle
    assert handle.hit_count == 0
    assert handle.last_used_at is None


def test_condition_cache_request_rejects_speaker_context_past_bucket() -> None:
    with pytest.raises(coreml_cache.CacheValidationError, match="speaker_context_len"):
        condition_request("ref_existing", speaker_context_len=161)


def test_prepare_condition_cache_rejects_speaker_context_mismatch_with_reference() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(
        reference_request(speaker_context_len=100),
    ).handle
    request = condition_request(reference.id, speaker_context_len=120)

    with pytest.raises(coreml_cache.CacheConflictError, match="speaker_context_len"):
        manager.prepare_condition_cache(request)


def test_get_increments_hit_count_and_last_used_at() -> None:
    clock = Clock()
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=clock)
    reference = manager.prepare_reference_cache(reference_request()).handle
    condition = manager.prepare_condition_cache(condition_request(reference.id)).handle

    clock.advance(1)
    first_reference_hit = manager.get_reference_cache(reference.id)

    assert first_reference_hit.hit_count == 1
    assert first_reference_hit.last_used_at == clock.current

    clock.advance(1)
    second_reference_hit = manager.get_reference_cache(reference.id)

    assert second_reference_hit.hit_count == 2
    assert second_reference_hit.last_used_at == clock.current

    clock.advance(1)
    condition_hit = manager.get_condition_cache(condition.id)

    assert condition_hit.hit_count == 1
    assert condition_hit.last_used_at == clock.current


def test_require_condition_cache_matches_without_incrementing_hits() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    request = condition_request(reference.id)
    condition = manager.prepare_condition_cache(request).handle

    required = manager.require_condition_cache(condition.id, request)

    assert required is condition
    assert condition.hit_count == 0
    assert condition.last_used_at is None


def test_require_condition_cache_rejects_changed_condition_fingerprint() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    request = condition_request(reference.id)
    condition = manager.prepare_condition_cache(request).handle
    changed_request = replace(request, condition_fingerprint="condition-b")

    with pytest.raises(coreml_cache.CacheConflictError, match="condition_fingerprint"):
        manager.require_condition_cache(condition.id, changed_request)


def test_require_condition_cache_rejects_changed_bucket_or_branch_layout() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    request = condition_request(reference.id)
    condition = manager.prepare_condition_cache(request).handle
    changed_bucket_request = replace(
        request,
        bucket=coreml_cache.CoreMLConditionBucket(
            sequence_length=100,
            text_len=128,
            speaker_context_len_bucket=160,
        ),
    )
    changed_layout_request = replace(
        request,
        branch_layouts=(coreml_cache.BRANCH_LAYOUT_COND1,),
    )

    with pytest.raises(coreml_cache.CacheConflictError, match="bucket"):
        manager.require_condition_cache(condition.id, changed_bucket_request)
    with pytest.raises(coreml_cache.CacheConflictError, match="branch_layouts"):
        manager.require_condition_cache(condition.id, changed_layout_request)


def test_delete_reference_cache_cascade_removes_dependent_conditions() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    condition = manager.prepare_condition_cache(condition_request(reference.id)).handle

    with pytest.raises(coreml_cache.CacheConflictError):
        manager.delete_reference_cache(reference.id, cascade=False)

    assert manager.delete_reference_cache(reference.id, cascade=True) is True

    with pytest.raises(coreml_cache.CacheNotFoundError):
        manager.get_reference_cache(reference.id)
    with pytest.raises(coreml_cache.CacheNotFoundError):
        manager.get_condition_cache(condition.id)
    assert manager.delete_reference_cache(reference.id) is False


def test_delete_condition_cache_prunes_unused_reference_bucket_only_after_last_user() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    first_condition = manager.prepare_condition_cache(
        condition_request(reference.id, condition_fingerprint="condition-a"),
    ).handle
    second_condition = manager.prepare_condition_cache(
        condition_request(reference.id, condition_fingerprint="condition-b"),
    ).handle

    assert reference.resident_buckets == (first_condition.bucket_id,)

    assert manager.delete_condition_cache(first_condition.id) is True
    assert reference.resident_buckets == (first_condition.bucket_id,)

    assert manager.delete_condition_cache(second_condition.id) is True
    assert reference.resident_buckets == ()


def test_prune_expired_removes_expired_references_and_dependent_conditions() -> None:
    clock = Clock()
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=clock)
    expired_reference = manager.prepare_reference_cache(
        reference_request(reference_fingerprint="reference-expiring", ttl_seconds=5),
    ).handle
    dependent_condition = manager.prepare_condition_cache(
        condition_request(expired_reference.id),
    ).handle
    live_reference = manager.prepare_reference_cache(
        reference_request(reference_fingerprint="reference-live"),
    ).handle
    expiring_condition = manager.prepare_condition_cache(
        condition_request(
            live_reference.id,
            condition_fingerprint="condition-expiring",
            ttl_seconds=5,
        ),
    ).handle

    clock.advance(5)

    assert manager.prune_expired() == {"reference": 1, "condition": 2}

    with pytest.raises(coreml_cache.CacheNotFoundError):
        manager.get_reference_cache(expired_reference.id)
    with pytest.raises(coreml_cache.CacheNotFoundError):
        manager.get_condition_cache(dependent_condition.id)
    assert manager.get_reference_cache(live_reference.id).id == live_reference.id
    assert live_reference.resident_buckets == ()
    with pytest.raises(coreml_cache.CacheNotFoundError):
        manager.get_condition_cache(expiring_condition.id)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_fingerprint": ""},
        {"codec_fingerprint": "  "},
        {"reference_fingerprint": ""},
        {"speaker_context_len": 0},
        {"memory_bytes": -1},
        {"ttl_seconds": 0},
        {"ttl_seconds": float("nan")},
        {"metadata": {1: "bad"}},
    ],
)
def test_reference_cache_request_rejects_invalid_inputs(overrides: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "model_fingerprint": "model-a",
        "codec_fingerprint": "codec-a",
        "reference_fingerprint": "reference-a",
        "speaker_context_len": 120,
    }
    kwargs.update(overrides)

    with pytest.raises(coreml_cache.CacheValidationError):
        coreml_cache.ReferenceCacheRequest(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"reference_cache_id": ""},
        {"model_fingerprint": ""},
        {"tokenizer_fingerprint": ""},
        {"condition_fingerprint": ""},
        {"bucket": object()},
        {"speaker_context_len": 0},
        {"branch_layouts": ()},
        {"branch_layouts": "cond1"},
        {"branch_layouts": ("split",)},
        {"branch_layouts": (coreml_cache.BRANCH_LAYOUT_COND1, coreml_cache.BRANCH_LAYOUT_COND1)},
        {
            "branch_layouts": (
                coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
                coreml_cache.BRANCH_LAYOUT_COND1,
            ),
        },
        {"branch_layouts": (coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,)},
        {"ttl_seconds": 0},
        {"metadata": {1: "bad"}},
    ],
)
def test_condition_cache_request_rejects_invalid_inputs(overrides: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "reference_cache_id": "ref_existing",
        "model_fingerprint": "model-a",
        "tokenizer_fingerprint": "tokenizer-a",
        "condition_fingerprint": "condition-a",
        "bucket": condition_bucket(),
        "speaker_context_len": 120,
        "branch_layouts": (coreml_cache.BRANCH_LAYOUT_COND1,),
    }
    kwargs.update(overrides)

    with pytest.raises(coreml_cache.CacheValidationError):
        coreml_cache.ConditionCacheRequest(**kwargs)
