from __future__ import annotations

import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "irodori_tts" / "coreml_cache.py"
MODULE_SPEC = importlib.util.spec_from_file_location("coreml_cache_concurrency", MODULE_PATH)
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


def _run_concurrently(worker_count: int, fn):
    barrier = threading.Barrier(worker_count)

    def wrapped(worker_index: int):
        barrier.wait(timeout=5.0)
        return fn(worker_index)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(wrapped, worker_index) for worker_index in range(worker_count)]
        return [future.result(timeout=10.0) for future in futures]


def test_concurrent_same_key_reference_prepare_reuses_single_handle() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    request = reference_request(memory_bytes=512)

    results = _run_concurrently(
        16,
        lambda _worker_index: manager.prepare_reference_cache(request),
    )

    assert {result.handle.id for result in results} == {
        manager.reference_cache_id_for_request(request)
    }
    assert len({id(result.handle) for result in results}) == 1
    assert sum(not result.reused for result in results) == 1
    assert manager.metrics_snapshot()["reference_count"] == 1


def test_concurrent_same_key_condition_prepare_reuses_single_handle() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    request = condition_request(reference.id)

    results = _run_concurrently(
        16,
        lambda _worker_index: manager.prepare_condition_cache(request),
    )

    assert {result.handle.id for result in results} == {
        manager.condition_cache_id_for_request(request)
    }
    assert len({id(result.handle) for result in results}) == 1
    assert sum(not result.reused for result in results) == 1
    snapshot = manager.metrics_snapshot()
    assert snapshot["reference_count"] == 1
    assert snapshot["condition_count"] == 1


def test_prepare_lock_stripes_do_not_grow_for_many_unique_cache_keys() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference_locks = manager._reference_prepare_locks
    condition_locks = manager._condition_prepare_locks

    assert isinstance(reference_locks, tuple)
    assert isinstance(condition_locks, tuple)
    assert len(reference_locks) > 1
    assert len(condition_locks) > 1

    for index in range(len(reference_locks) * 3):
        manager.prepare_reference_cache(
            reference_request(reference_fingerprint=f"reference-stripe-{index}")
        )

    reference = manager.prepare_reference_cache(
        reference_request(reference_fingerprint="condition-stripe-anchor")
    ).handle
    for index in range(len(condition_locks) * 3):
        manager.prepare_condition_cache(
            condition_request(reference.id, condition_fingerprint=f"condition-stripe-{index}")
        )

    assert manager._reference_prepare_locks is reference_locks
    assert manager._condition_prepare_locks is condition_locks
    assert len(manager._reference_prepare_locks) == len(reference_locks)
    assert len(manager._condition_prepare_locks) == len(condition_locks)
    assert not isinstance(manager._reference_prepare_locks, dict)
    assert not isinstance(manager._condition_prepare_locks, dict)


def test_metrics_snapshot_runs_during_concurrent_prepare_and_delete() -> None:
    manager = coreml_cache.InMemoryCoreMLCacheManager(clock=Clock())
    reference = manager.prepare_reference_cache(reference_request()).handle
    worker_count = 5
    iterations = 200
    barrier = threading.Barrier(worker_count)

    def wait_for_start() -> None:
        barrier.wait(timeout=5.0)

    def snapshot_worker() -> None:
        wait_for_start()
        for _ in range(iterations * 4):
            snapshot = manager.metrics_snapshot()
            assert snapshot["reference_count"] >= 1
            assert snapshot["condition_count"] >= 0
            assert snapshot["total_memory_bytes"] >= 0

    def reference_worker(worker_index: int) -> None:
        wait_for_start()
        for iteration in range(iterations):
            result = manager.prepare_reference_cache(
                reference_request(
                    reference_fingerprint=f"reference-{worker_index}-{iteration}",
                )
            )
            if iteration % 2 == 0:
                manager.delete_reference_cache(result.handle.id)

    def condition_worker(worker_index: int) -> None:
        wait_for_start()
        for iteration in range(iterations):
            result = manager.prepare_condition_cache(
                condition_request(
                    reference.id,
                    condition_fingerprint=f"condition-{worker_index}-{iteration}",
                )
            )
            if iteration % 2 == 0:
                manager.delete_condition_cache(result.handle.id)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(snapshot_worker),
            executor.submit(reference_worker, 0),
            executor.submit(reference_worker, 1),
            executor.submit(condition_worker, 0),
            executor.submit(condition_worker, 1),
        ]
        for future in futures:
            future.result(timeout=10.0)


def test_runtime_resident_prepare_lock_helper_is_per_key() -> None:
    from irodori_tts import inference_runtime

    runtime = inference_runtime.InferenceRuntime.__new__(inference_runtime.InferenceRuntime)

    first = runtime._resident_prepare_lock(("reference", "same"))
    second = runtime._resident_prepare_lock(("reference", "same"))
    prepare_locks = runtime._resident_prepare_locks
    stripe_count = len(prepare_locks)

    for index in range(stripe_count * 4):
        runtime._resident_prepare_lock(("reference", index))

    assert first is second
    assert isinstance(runtime._resident_prepare_locks, tuple)
    assert runtime._resident_prepare_locks is prepare_locks
    assert len(runtime._resident_prepare_locks) == stripe_count
    assert not isinstance(runtime._resident_prepare_locks, dict)
    with first:
        with second:
            pass
