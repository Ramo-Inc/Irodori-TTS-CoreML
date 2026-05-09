from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "openai_api_server.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "openai_api_server_segment_cache",
    MODULE_PATH,
)
assert MODULE_SPEC is not None
openai_api_server = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = openai_api_server
MODULE_SPEC.loader.exec_module(openai_api_server)

ServerSettings = openai_api_server.ServerSettings
create_app = openai_api_server.create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.legacy_requests: list[Any] = []
        self.fast_requests: list[tuple[Any, Any]] = []
        self.codec = SimpleNamespace(
            sample_rate=24_000,
            model=SimpleNamespace(hop_length=512),
        )
        self.model_cfg = SimpleNamespace(
            latent_patch_size=2,
            text_tokenizer_repo="fake-tokenizer",
            text_add_bos=True,
        )
        self.tokenizer_fingerprint = "tokenizer:fake"

    def estimate_patched_steps(self, seconds: float) -> int:
        target_samples = int(float(seconds) * int(self.codec.sample_rate))
        latent_steps = (target_samples + int(self.codec.model.hop_length) - 1) // int(
            self.codec.model.hop_length,
        )
        return (latent_steps + int(self.model_cfg.latent_patch_size) - 1) // int(
            self.model_cfg.latent_patch_size,
        )

    def tokenize_for_bucket(self, normalized_text: str) -> tuple[int, str]:
        return len(normalized_text.encode("utf-8")) + 1, "sha256:fake-token-ids"

    def synthesize(self, request: Any, log_fn: Any = None) -> SimpleNamespace:
        del log_fn
        self.legacy_requests.append(request)
        return SimpleNamespace(
            audio=torch.zeros((1, 8), dtype=torch.float32),
            sample_rate=16_000,
        )

    def synthesize_with_condition_cache(
        self,
        request: Any,
        *,
        condition_cache: Any,
        log_fn: Any = None,
    ) -> SimpleNamespace:
        del log_fn
        self.fast_requests.append((request, condition_cache))
        return SimpleNamespace(
            audio=torch.zeros((1, 8), dtype=torch.float32),
            sample_rate=16_000,
        )


@pytest.fixture
def settings() -> ServerSettings:
    return ServerSettings(
        host="127.0.0.1",
        port=0,
        checkpoint="unit-test-checkpoint",
        reference_wav=PROJECT_ROOT / "rem.wav",
        api_model_id="irodori-test-model",
        model_device="cpu",
        codec_device="cpu",
        model_precision="fp32",
        codec_precision="fp32",
        codec_repo="unit-test-codec",
        default_num_steps=4,
        max_num_steps=8,
        seconds=None,
        min_seconds=1.0,
        max_seconds=3.0,
        chars_per_second=4.0,
        seconds_padding=0.0,
        max_ref_seconds=None,
        preload=False,
        log_timings=False,
    )


@pytest.fixture
def client_runtime(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> Iterator[tuple[TestClient, FakeRuntime]]:
    runtime = FakeRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        del self
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)
    with TestClient(create_app(settings)) as test_client:
        yield test_client, runtime


def speech_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "input": "segment cache test",
        "instructions": "neutral",
        "seconds": 2.0,
        "response_format": "pcm",
    }
    payload.update(overrides)
    return payload


def segment_plan(count: int) -> openai_api_server.SpeechSegmentPlan:
    segments = tuple(
        openai_api_server.SpeechSegment(text=f"segment {index}", seconds=1.0)
        for index in range(count)
    )
    return openai_api_server.SpeechSegmentPlan(
        segments=segments,
        total_seconds=float(count),
        seconds_mode="auto-chunked" if count > 1 else "auto",
    )


def force_segment_plan(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    plan = segment_plan(count)

    def build_speech_segment_plan(
        payload: dict[str, Any],
        text: str,
        settings: ServerSettings,
    ) -> openai_api_server.SpeechSegmentPlan:
        del payload, text, settings
        return plan

    monkeypatch.setattr(
        openai_api_server,
        "_build_speech_segment_plan",
        build_speech_segment_plan,
    )


def bucket() -> openai_api_server.CoreMLConditionBucket:
    return openai_api_server.CoreMLConditionBucket(
        sequence_length=100,
        text_len=64,
        speaker_context_len_bucket=160,
    )


def condition_handle(cache_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=cache_id)


def test_resolve_speech_cache_returns_per_segment_off_resolutions(
    settings: ServerSettings,
) -> None:
    plan = segment_plan(3)
    resolutions = openai_api_server._resolve_speech_cache(
        None,
        openai_api_server.CACHE_MODE_OFF,
        None,
        plan,
        "neutral",
        settings,
        openai_api_server.InMemoryCoreMLCacheManager(),
    )

    assert len(resolutions) == 3
    assert [resolution.condition_handle for resolution in resolutions] == [None] * 3
    assert [resolution.auto_status for resolution in resolutions] == ["off"] * 3


def test_resolve_speech_cache_returns_per_segment_auto_miss_resolutions(
    settings: ServerSettings,
) -> None:
    plan = segment_plan(2)
    resolutions = openai_api_server._resolve_speech_cache(
        None,
        openai_api_server.CACHE_MODE_AUTO,
        None,
        plan,
        "neutral",
        settings,
        openai_api_server.InMemoryCoreMLCacheManager(),
    )

    assert len(resolutions) == 2
    assert [resolution.condition_handle for resolution in resolutions] == [None, None]
    assert [resolution.auto_status for resolution in resolutions] == [
        "miss-fallback",
        "miss-fallback",
    ]


def test_single_cache_id_with_multi_segment_require_remains_conflict(
    settings: ServerSettings,
) -> None:
    with pytest.raises(openai_api_server.CacheConflictError):
        openai_api_server._resolve_speech_cache(
            {"cache_mode": "require", "cache_id": "cond_any"},
            openai_api_server.CACHE_MODE_REQUIRE,
            "cond_any",
            segment_plan(2),
            "neutral",
            settings,
            openai_api_server.InMemoryCoreMLCacheManager(),
        )


def create_reference_cache(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/v1/tts/reference-caches",
        json={"source": {"type": "server_default"}},
    )
    assert response.status_code == 201
    return response.json()


def create_matching_condition_cache(client: TestClient) -> dict[str, object]:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "segment cache test",
            "caption": "neutral",
            "seconds": 2.0,
            "cfg": {"mode": "independent"},
        },
    )
    assert response.status_code == 201
    return response.json()


def test_single_segment_cache_headers_keep_legacy_and_add_aliases(
    client_runtime: tuple[TestClient, FakeRuntime],
) -> None:
    client, _runtime = client_runtime
    condition = create_matching_condition_cache(client)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == condition["id"]
    assert response.headers["X-Irodori-Cache-Condition-Id"] == condition["id"]
    assert response.headers["X-Irodori-Cache-Condition-Count"] == "1"
    assert response.headers["X-Irodori-Reference-Cache-Id"] == condition["reference_cache_id"]
    assert response.headers["X-Irodori-Cache-Reference-Id"] == condition["reference_cache_id"]


def test_mixed_backend_headers_for_multi_segment_auto(
    client_runtime: tuple[TestClient, FakeRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime = client_runtime
    force_segment_plan(monkeypatch, 2)

    def auto_prepare_speech_caches(**_kwargs: object) -> list[object]:
        return [
            openai_api_server._SpeechCacheResolution(
                condition_handle=condition_handle("cond_0"),
                reference_cache_id="ref_0",
                reference_created=False,
                condition_created=True,
                bucket=bucket(),
                auto_status="prepared",
            ),
            openai_api_server._SpeechCacheResolution(
                condition_handle=None,
                reference_cache_id=None,
                reference_created=False,
                condition_created=False,
                auto_status="miss-fallback",
            ),
        ]

    monkeypatch.setattr(
        openai_api_server,
        "_auto_prepare_speech_caches",
        auto_prepare_speech_caches,
    )

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(seconds=None),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "mixed"
    assert response.headers["X-Irodori-Backend"] == "mixed"
    assert response.headers["X-Irodori-Backend-Per-Segment"] == "coreml-stateful,pytorch"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == "cond_0"
    assert response.headers["X-Irodori-Cache-Condition-Id"] == "cond_0"
    assert response.headers["X-Irodori-Cache-Condition-Count"] == "1"
    assert response.headers["X-Irodori-Reference-Cache-Id"] == "ref_0"
    assert response.headers["X-Irodori-Cache-Auto"] == "mixed"
    assert len(runtime.fast_requests) == 1
    assert len(runtime.legacy_requests) == 1


def test_multi_segment_all_coreml_condition_ids_are_comma_separated(
    client_runtime: tuple[TestClient, FakeRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime = client_runtime
    force_segment_plan(monkeypatch, 3)

    def auto_prepare_speech_caches(**_kwargs: object) -> list[object]:
        return [
            openai_api_server._SpeechCacheResolution(
                condition_handle=condition_handle(f"cond_{index}"),
                reference_cache_id="ref_shared",
                reference_created=False,
                condition_created=True,
                bucket=bucket(),
                auto_status="prepared",
            )
            for index in range(3)
        ]

    monkeypatch.setattr(
        openai_api_server,
        "_auto_prepare_speech_caches",
        auto_prepare_speech_caches,
    )

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(seconds=None),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == "cond_0,cond_1,cond_2"
    assert response.headers["X-Irodori-Cache-Condition-Id"] == "cond_0,cond_1,cond_2"
    assert response.headers["X-Irodori-Cache-Condition-Count"] == "3"
    assert response.headers["X-Irodori-Reference-Cache-Id"] == "ref_shared"
    assert response.headers["X-Irodori-Cache-Auto"] == "prepared"
    assert len(runtime.fast_requests) == 3
    assert runtime.legacy_requests == []


def test_many_condition_ids_are_summarized_with_count(
    client_runtime: tuple[TestClient, FakeRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime = client_runtime
    force_segment_plan(monkeypatch, 6)

    def auto_prepare_speech_caches(**_kwargs: object) -> list[object]:
        return [
            openai_api_server._SpeechCacheResolution(
                condition_handle=condition_handle(f"cond_{index}"),
                reference_cache_id="ref_shared",
                reference_created=False,
                condition_created=True,
                bucket=bucket(),
                auto_status="prepared",
            )
            for index in range(6)
        ]

    monkeypatch.setattr(
        openai_api_server,
        "_auto_prepare_speech_caches",
        auto_prepare_speech_caches,
    )

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(seconds=None),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Condition-Cache-Id"] == "cond_0,+5 more"
    assert response.headers["X-Irodori-Cache-Condition-Id"] == "cond_0,+5 more"
    assert response.headers["X-Irodori-Cache-Condition-Count"] == "6"
    assert len(runtime.fast_requests) == 6
    assert runtime.legacy_requests == []


def test_multi_segment_auto_bucket_planning_runs_off_event_loop(
    client_runtime: tuple[TestClient, FakeRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime = client_runtime
    force_segment_plan(monkeypatch, 2)
    planned_segments: list[str] = []

    def resolve_auto_bucket_resolution(
        irodori: dict[str, Any] | None,
        segment: openai_api_server.SpeechSegment,
        runtime: FakeRuntime,
    ) -> openai_api_server.AutoBucketResolution:
        del irodori, runtime
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AssertionError("AUTO bucket planning ran on the event loop")
        planned_segments.append(segment.text)
        return openai_api_server.AutoBucketResolution(
            bucket=bucket(),
            reason=None,
            attempted=None,
        )

    monkeypatch.setattr(
        openai_api_server,
        "_resolve_auto_bucket_resolution",
        resolve_auto_bucket_resolution,
    )

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(seconds=None),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Backend-Per-Segment"] == "pytorch,pytorch"
    assert response.headers["X-Irodori-Cache-Auto"] == "miss-fallback"
    assert response.headers["X-Irodori-Bucket"] == "S100_T64_R160,S100_T64_R160"
    assert planned_segments == ["segment 0", "segment 1"]
    assert len(runtime.legacy_requests) == 2
    assert runtime.fast_requests == []
