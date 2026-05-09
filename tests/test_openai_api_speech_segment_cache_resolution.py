from __future__ import annotations

import asyncio
import dataclasses
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


def test_split_text_for_auto_chunks_recursively_splits_oversize_text(
    settings: ServerSettings,
) -> None:
    text = "あ" * 257
    chunks = openai_api_server._split_text_for_auto_chunks(text, settings)
    assert chunks
    assert "".join(chunks) == text
    char_budget = openai_api_server._speech_chunk_char_budget(settings)
    for chunk in chunks:
        assert openai_api_server._count_non_whitespace_chars(chunk) <= char_budget


def test_split_text_recursive_prefers_sentence_then_phrase_then_space() -> None:
    sentence_split = openai_api_server._split_chunk_at_priority_boundary(
        "あいうえお。かきくけこ、さしすせそ ",
    )
    assert sentence_split is not None
    left, right = sentence_split
    assert left.endswith("。")
    assert right.strip()

    phrase_split = openai_api_server._split_chunk_at_priority_boundary(
        "あいうえお、かきくけこ さしすせそ",
    )
    assert phrase_split is not None
    left, right = phrase_split
    assert left.endswith("、")

    space_split = openai_api_server._split_chunk_at_priority_boundary(
        "abcdefg hijklmn",
    )
    assert space_split is not None
    left, right = space_split
    assert left.endswith(" ")


def test_split_text_recursive_handles_full_width_space() -> None:
    text = "abcdefghij　klmnopqrst"
    split = openai_api_server._split_chunk_at_priority_boundary(text)
    assert split is not None
    left, right = split
    assert left.endswith("　")
    assert right.strip() == "klmnopqrst"


def test_build_speech_segment_plan_keeps_long_input_under_256_as_single_segment(
    settings: ServerSettings,
) -> None:
    long_settings = openai_api_server.ServerSettings(
        host=settings.host,
        port=settings.port,
        checkpoint=settings.checkpoint,
        reference_wav=settings.reference_wav,
        api_model_id=settings.api_model_id,
        model_device=settings.model_device,
        codec_device=settings.codec_device,
        model_precision=settings.model_precision,
        codec_precision=settings.codec_precision,
        codec_repo=settings.codec_repo,
        default_num_steps=settings.default_num_steps,
        max_num_steps=settings.max_num_steps,
        seconds=None,
        min_seconds=settings.min_seconds,
        max_seconds=70.0,
        chars_per_second=4.0,
        seconds_padding=1.5,
        max_ref_seconds=settings.max_ref_seconds,
        preload=settings.preload,
        log_timings=settings.log_timings,
    )
    text = "あ" * 256
    plan = openai_api_server._build_speech_segment_plan({}, text, long_settings)
    assert len(plan.segments) == 1
    assert plan.segments[0].text == text


def test_build_speech_segment_plan_splits_text_above_256_chars(
    settings: ServerSettings,
) -> None:
    long_settings = openai_api_server.ServerSettings(
        host=settings.host,
        port=settings.port,
        checkpoint=settings.checkpoint,
        reference_wav=settings.reference_wav,
        api_model_id=settings.api_model_id,
        model_device=settings.model_device,
        codec_device=settings.codec_device,
        model_precision=settings.model_precision,
        codec_precision=settings.codec_precision,
        codec_repo=settings.codec_repo,
        default_num_steps=settings.default_num_steps,
        max_num_steps=settings.max_num_steps,
        seconds=None,
        min_seconds=settings.min_seconds,
        max_seconds=70.0,
        chars_per_second=4.0,
        seconds_padding=1.5,
        max_ref_seconds=settings.max_ref_seconds,
        preload=settings.preload,
        log_timings=settings.log_timings,
    )
    text = ("あ" * 200) + "。" + ("い" * 200)
    plan = openai_api_server._build_speech_segment_plan({}, text, long_settings)
    assert len(plan.segments) >= 2
    rejoined = "".join(segment.text for segment in plan.segments)
    assert rejoined == text
    char_budget = openai_api_server._speech_chunk_char_budget(long_settings)
    for segment in plan.segments:
        assert openai_api_server._count_non_whitespace_chars(segment.text) <= char_budget


def test_runtime_refine_splits_segments_until_token_len_fits(
    settings: ServerSettings,
) -> None:
    class TokenCappedRuntime:
        def __init__(self) -> None:
            self.codec = SimpleNamespace(
                sample_rate=24_000,
                model=SimpleNamespace(hop_length=512),
            )
            self.model_cfg = SimpleNamespace(latent_patch_size=2)

        def estimate_patched_steps(self, seconds: float) -> int:
            target_samples = int(float(seconds) * int(self.codec.sample_rate))
            latent_steps = (target_samples + int(self.codec.model.hop_length) - 1) // int(
                self.codec.model.hop_length
            )
            return (latent_steps + int(self.model_cfg.latent_patch_size) - 1) // int(
                self.model_cfg.latent_patch_size
            )

        def tokenize_for_bucket(self, normalized_text: str) -> tuple[int, str]:
            return len(normalized_text), "sha256:fake"

    runtime = TokenCappedRuntime()
    plan = openai_api_server.SpeechSegmentPlan(
        segments=(
            openai_api_server.SpeechSegment(
                text="A" * 300, seconds=10.0,
            ),
        ),
        total_seconds=10.0,
        seconds_mode="auto",
    )
    refined = openai_api_server._runtime_refine_segment_plan_for_auto(plan, runtime, settings)
    assert refined is not plan
    assert len(refined.segments) >= 2
    for segment in refined.segments:
        normalized = openai_api_server.normalize_text(segment.text).strip()
        token_len, _ = runtime.tokenize_for_bucket(normalized)
        assert token_len <= openai_api_server.AUTO_TEXT_LEN_MAX
    assert "".join(segment.text for segment in refined.segments) == "A" * 300


def test_runtime_refine_keeps_segment_when_split_does_not_reduce_tokens(
    settings: ServerSettings,
) -> None:
    class ConstantTokenRuntime:
        def __init__(self) -> None:
            self.codec = SimpleNamespace(
                sample_rate=24_000,
                model=SimpleNamespace(hop_length=512),
            )
            self.model_cfg = SimpleNamespace(latent_patch_size=2)

        def estimate_patched_steps(self, seconds: float) -> int:
            del seconds
            return 50

        def tokenize_for_bucket(self, normalized_text: str) -> tuple[int, str]:
            del normalized_text
            return 999, "sha256:fake"

    runtime = ConstantTokenRuntime()
    segment = openai_api_server.SpeechSegment(text="abc def ghi", seconds=2.0)
    plan = openai_api_server.SpeechSegmentPlan(
        segments=(segment,),
        total_seconds=2.0,
        seconds_mode="auto",
    )
    refined = openai_api_server._runtime_refine_segment_plan_for_auto(plan, runtime, settings)
    assert refined is plan


def test_audio_speech_strict_coreml_with_oversize_t_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    import dataclasses

    runtime = FakeRuntime()
    strict_settings = dataclasses.replace(settings, strict_coreml=True)

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    def oversize_tokens(text: str) -> tuple[int, str]:
        del text
        return 999, "sha256:oversize"

    monkeypatch.setattr(runtime, "tokenize_for_bucket", oversize_tokens)

    with TestClient(create_app(strict_settings)) as client:
        response = client.post("/v1/audio/speech", json=speech_payload())

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "coreml_backend_unavailable"
    assert "strict_coreml" in body["error"]["message"]
    assert runtime.legacy_requests == []
    assert runtime.fast_requests == []


def test_audio_speech_strict_coreml_off_mode_still_uses_pytorch(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    import dataclasses

    runtime = FakeRuntime()
    strict_settings = dataclasses.replace(settings, strict_coreml=True)

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    with TestClient(create_app(strict_settings)) as client:
        response = client.post(
            "/v1/audio/speech",
            json=speech_payload(irodori={"cache_mode": "off"}),
        )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert "X-Irodori-Cache-Auto" not in response.headers
    assert len(runtime.legacy_requests) == 1
    assert runtime.fast_requests == []


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


def test_request_seconds_plan_skips_runtime_refine_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    import dataclasses

    runtime = FakeRuntime()
    strict_settings = dataclasses.replace(settings, strict_coreml=True)

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    refine_calls: list[str] = []

    def spy_refine(
        plan: openai_api_server.SpeechSegmentPlan,
        _runtime: Any,
        _settings: ServerSettings,
        *,
        speed: float = 1.0,
    ) -> openai_api_server.SpeechSegmentPlan:
        del speed
        refine_calls.append(plan.seconds_mode)
        return plan

    monkeypatch.setattr(
        openai_api_server,
        "_runtime_refine_segment_plan_for_auto",
        spy_refine,
    )

    def oversize_tokens(text: str) -> tuple[int, str]:
        del text
        return 999, "sha256:oversize"

    monkeypatch.setattr(runtime, "tokenize_for_bucket", oversize_tokens)

    with TestClient(create_app(strict_settings)) as client:
        response = client.post("/v1/audio/speech", json=speech_payload())

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "coreml_backend_unavailable"
    assert refine_calls == []
    assert runtime.legacy_requests == []
    assert runtime.fast_requests == []


def test_runtime_refine_runs_for_auto_seconds_plan(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    runtime = FakeRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    refine_calls: list[str] = []

    def spy_refine(
        plan: openai_api_server.SpeechSegmentPlan,
        _runtime: Any,
        _settings: ServerSettings,
        *,
        speed: float = 1.0,
    ) -> openai_api_server.SpeechSegmentPlan:
        del speed
        refine_calls.append(plan.seconds_mode)
        return plan

    monkeypatch.setattr(
        openai_api_server,
        "_runtime_refine_segment_plan_for_auto",
        spy_refine,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/audio/speech",
            json=speech_payload(seconds=None),
        )

    assert response.status_code == 200
    assert refine_calls == ["auto"]


def _service_default_settings(settings: ServerSettings) -> ServerSettings:
    return dataclasses.replace(
        settings,
        chars_per_second=5.5,
        seconds_padding=1.5,
        min_seconds=4.0,
        max_seconds=70.0,
    )


@pytest.mark.parametrize(
    ("char_count", "expected_seconds"),
    [
        (135, 26.5),
        (159, 30.5),
    ],
)
def test_default_chars_per_second_estimates_match_calibration(
    settings: ServerSettings,
    char_count: int,
    expected_seconds: float,
) -> None:
    default_settings = _service_default_settings(settings)
    text = "あ" * char_count
    estimated = openai_api_server._estimate_generation_seconds(text, default_settings)
    assert estimated == pytest.approx(expected_seconds)


def test_default_chars_per_second_keeps_chunk_below_old_4_0_default(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    old_settings = dataclasses.replace(default_settings, chars_per_second=4.0)
    for char_count in (135, 159):
        text = "あ" * char_count
        new_seconds = openai_api_server._estimate_generation_seconds(text, default_settings)
        old_seconds = openai_api_server._estimate_generation_seconds(text, old_settings)
        assert new_seconds < old_seconds


class _StaticBucketRuntime:
    def __init__(self, *, token_len: int) -> None:
        self._token_len = int(token_len)
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
        del normalized_text
        return self._token_len, "sha256:fake-token-ids"


def test_default_chars_per_second_avoids_s1536_for_159_char_segment(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    text = "あ" * 159
    plan = openai_api_server._build_speech_segment_plan(
        {"speed": 1.0}, text, default_settings,
    )
    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.seconds == pytest.approx(30.5)

    runtime = _StaticBucketRuntime(token_len=64)
    resolution = openai_api_server._resolve_auto_bucket_resolution(None, segment, runtime)

    assert resolution.bucket is not None
    assert resolution.bucket.sequence_length == 1024
    assert resolution.bucket.text_len == 128


def test_old_4_0_default_would_have_picked_s1536_for_oversize_chunk(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    old_settings = dataclasses.replace(default_settings, chars_per_second=4.0)

    runtime = _StaticBucketRuntime(token_len=64)
    text = "あ" * 175
    new_segment = openai_api_server.SpeechSegment(
        text=text,
        seconds=openai_api_server._estimate_generation_seconds(text, default_settings),
    )
    old_segment = openai_api_server.SpeechSegment(
        text=text,
        seconds=openai_api_server._estimate_generation_seconds(text, old_settings),
    )
    new_resolution = openai_api_server._resolve_auto_bucket_resolution(
        None, new_segment, runtime,
    )
    old_resolution = openai_api_server._resolve_auto_bucket_resolution(
        None, old_segment, runtime,
    )

    assert new_resolution.bucket is not None
    assert old_resolution.bucket is not None
    assert new_resolution.bucket.sequence_length == 1024
    assert old_resolution.bucket.sequence_length == 1536


def test_explicit_request_seconds_unaffected_by_default_chars_per_second(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    plan = openai_api_server._build_speech_segment_plan(
        {"seconds": 12.0}, "あ" * 159, default_settings,
    )
    assert plan.seconds_mode == "request"
    assert len(plan.segments) == 1
    assert plan.segments[0].seconds == pytest.approx(12.0)


def test_default_speed_setting_is_1_2(settings: ServerSettings) -> None:
    default_settings = _service_default_settings(settings)
    assert default_settings.default_speed == pytest.approx(1.2)


def test_default_speed_reduces_estimated_seconds_for_auto_plan(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    text = "あ" * 159
    auto_plan = openai_api_server._build_speech_segment_plan(
        {}, text, default_settings,
    )
    speed_one_plan = openai_api_server._build_speech_segment_plan(
        {"speed": 1.0}, text, default_settings,
    )
    assert auto_plan.segments[0].seconds < speed_one_plan.segments[0].seconds
    assert auto_plan.segments[0].seconds == pytest.approx(26.0)
    assert speed_one_plan.segments[0].seconds == pytest.approx(30.5)


def test_explicit_payload_speed_overrides_default_speed(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    text = "あ" * 159
    plan = openai_api_server._build_speech_segment_plan(
        {"speed": 1.0}, text, default_settings,
    )
    assert plan.seconds_mode == "auto"
    assert plan.segments[0].seconds == pytest.approx(30.5)


def test_request_seconds_unaffected_by_default_speed(
    settings: ServerSettings,
) -> None:
    default_settings = _service_default_settings(settings)
    plan = openai_api_server._build_speech_segment_plan(
        {"seconds": 12.0}, "あ" * 159, default_settings,
    )
    assert plan.seconds_mode == "request"
    assert plan.segments[0].seconds == pytest.approx(12.0)
