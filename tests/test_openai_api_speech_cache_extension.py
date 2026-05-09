from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
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
MODULE_SPEC = importlib.util.spec_from_file_location("openai_api_server", MODULE_PATH)
assert MODULE_SPEC is not None
openai_api_server = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = openai_api_server
MODULE_SPEC.loader.exec_module(openai_api_server)

ServerSettings = openai_api_server.ServerSettings
create_app = openai_api_server.create_app

SPEECH_TEXT = "cache this utterance"
CAPTION = "neutral"


class FakeRuntime:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.fast_requests: list[tuple[Any, Any]] = []
        self.fast_resident_references: list[Any | None] = []
        self.resident_store: dict[str, openai_api_server.ResidentReferenceTensors] = {}
        self.deleted_resident_ids: list[str] = []
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
            self.codec.model.hop_length
        )
        return (latent_steps + int(self.model_cfg.latent_patch_size) - 1) // int(
            self.model_cfg.latent_patch_size
        )

    def tokenize_for_bucket(self, normalized_text: str) -> tuple[int, str]:
        return len(normalized_text.encode("utf-8")) + 1, "sha256:fake-token-ids"

    def synthesize(self, request: Any, log_fn: Any = None) -> SimpleNamespace:
        del log_fn
        self.requests.append(request)
        return SimpleNamespace(
            audio=torch.zeros((1, 8), dtype=torch.float32),
            sample_rate=16_000,
        )

    def synthesize_with_condition_cache(
        self,
        request: Any,
        *,
        condition_cache: Any,
        resident_reference_tensors: Any | None = None,
        log_fn: Any = None,
    ) -> SimpleNamespace:
        del log_fn
        self.fast_requests.append((request, condition_cache))
        self.fast_resident_references.append(resident_reference_tensors)
        return SimpleNamespace(
            audio=torch.zeros((1, 8), dtype=torch.float32),
            sample_rate=16_000,
        )

    def prepare_default_reference_tensors(
        self,
        ref_wav: str | Path,
        max_ref_seconds: float | None,
        *,
        reference_cache_id: str | None = None,
        reference_fingerprint: str | None = None,
        force_refresh: bool = False,
    ) -> openai_api_server.ResidentReferenceTensors:
        del max_ref_seconds, force_refresh
        cache_id = str(reference_cache_id or "resident_default")
        ref_latent = torch.ones((1, 2, 2), dtype=torch.float32)
        ref_mask = torch.ones((1, 2), dtype=torch.bool)
        speaker_state = torch.ones((1, 3, 4), dtype=torch.float32)
        speaker_mask = torch.ones((1, 3), dtype=torch.bool)
        memory_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (ref_latent, ref_mask, speaker_state, speaker_mask)
        )
        now = datetime.now(timezone.utc)
        resident = openai_api_server.ResidentReferenceTensors(
            reference_cache_id=cache_id,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            reference_fingerprint=reference_fingerprint,
            source=str(Path(ref_wav).expanduser().resolve(strict=False)),
            ref_len=2,
            speaker_context_len=3,
            speaker_dim=4,
            memory_bytes=int(memory_bytes),
            device="cpu",
            dtype="torch.float32",
            created_at=now,
            last_used_at=now,
        )
        self.resident_store[cache_id] = resident
        return resident

    def get_resident_reference_tensors(
        self,
        reference_cache_id: str,
    ) -> openai_api_server.ResidentReferenceTensors | None:
        return self.resident_store.get(reference_cache_id)

    def delete_resident_reference_tensors(self, reference_cache_id: str) -> bool:
        self.deleted_resident_ids.append(reference_cache_id)
        return self.resident_store.pop(reference_cache_id, None) is not None


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
) -> Iterator[tuple[TestClient, FakeRuntime, dict[str, int]]]:
    runtime = FakeRuntime()
    calls = {"get_runtime": 0}

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        calls["get_runtime"] += 1
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)
    with TestClient(create_app(settings)) as test_client:
        yield test_client, runtime, calls


def speech_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "input": SPEECH_TEXT,
        "instructions": CAPTION,
        "seconds": 2.0,
        "response_format": "pcm",
    }
    payload.update(overrides)
    return payload


def assert_cache_error(
    response: Any,
    status_code: int,
    error_type: str,
    *,
    cache_id: str | None = None,
    cache_mode: str | None = None,
) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert "detail" not in body
    assert body["error"]["type"] == error_type
    assert isinstance(body["error"]["message"], str)
    if cache_id is not None:
        assert body["error"]["cache_id"] == cache_id
    if cache_mode is not None:
        assert body["error"]["cache_mode"] == cache_mode


def create_reference_cache(client: TestClient, **overrides: object) -> dict[str, object]:
    payload = {"source": {"type": "server_default"}}
    payload.update(overrides)
    response = client.post(
        "/v1/tts/reference-caches",
        json=payload,
    )
    assert response.status_code == 201
    return response.json()


def create_matching_condition_cache(
    client: TestClient,
    *,
    input_text: str = SPEECH_TEXT,
    caption: str = CAPTION,
    seconds: float = 2.0,
    reference_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    reference = create_reference_cache(client, **(reference_overrides or {}))
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": input_text,
            "caption": caption,
            "seconds": seconds,
            "cfg": {"mode": "independent"},
        },
    )
    assert response.status_code == 201
    return response.json()


def assert_legacy_pcm_success(response: Any) -> None:
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/L16"
    assert response.headers["X-Irodori-Voice-Resolved"] == "server-default"
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.content == b"\x00" * 16


def test_no_irodori_defaults_to_auto_prepare_with_coreml_stateful(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post("/v1/audio/speech", json=speech_payload())

    assert openai_api_server._speech_cache_mode(None) == openai_api_server.CACHE_MODE_AUTO
    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Cache-Auto"] == "prepared"
    assert response.headers["X-Irodori-Bucket"] == "S256_T32_R160"
    assert response.headers["X-Irodori-Condition-Cache-Id"].startswith("cond_")
    assert response.headers["X-Irodori-Reference-Cache-Id"].startswith("ref_")
    assert "X-Irodori-Fallback-Reason" not in response.headers
    assert response.headers["X-Irodori-Speed"] == "1"
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 1
    assert runtime.fast_requests[0][0].text == SPEECH_TEXT
    assert runtime.fast_requests[0][0].caption == CAPTION


def test_default_speed_header_reflects_settings_default(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, _runtime, _calls = client_runtime
    response = client.post("/v1/audio/speech", json=speech_payload())
    assert response.status_code == 200
    assert response.headers["X-Irodori-Speed"] == "1"


def test_explicit_payload_speed_reflected_in_header(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, _runtime, _calls = client_runtime
    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(speed=1.2),
    )
    assert response.status_code == 200
    assert response.headers["X-Irodori-Speed"] == "1.2"


def test_invalid_payload_speed_rejected(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, _runtime, _calls = client_runtime
    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(speed=10.0),
    )
    assert response.status_code == 400


def test_auto_bucket_planning_runs_off_event_loop(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _runtime, _calls = client_runtime

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
        return openai_api_server.AutoBucketResolution(
            bucket=openai_api_server.CoreMLConditionBucket(
                sequence_length=100,
                text_len=64,
                speaker_context_len_bucket=160,
            ),
            reason=None,
            attempted=None,
            planning=openai_api_server.AutoSpeechPlanningContext(
                segment_text=segment.text,
                normalized_text=segment.text,
                seconds=float(segment.seconds),
                sample_rate=24_000,
                hop_length=512,
                latent_patch_size=2,
                tokenizer_fingerprint="tokenizer:fake",
                token_len=8,
                token_ids_hash="sha256:fake-token-ids",
                patched_steps=10,
            ),
        )

    monkeypatch.setattr(
        openai_api_server,
        "_resolve_auto_bucket_resolution",
        resolve_auto_bucket_resolution,
    )

    response = client.post("/v1/audio/speech", json=speech_payload())

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Bucket"] == "S100_T64_R160"
    assert response.headers["X-Irodori-Cache-Auto"] == "prepared"


def test_irodori_without_cache_mode_defaults_to_auto_prepare(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post("/v1/audio/speech", json=speech_payload(irodori={}))

    assert openai_api_server._speech_cache_mode({}) == openai_api_server.CACHE_MODE_AUTO
    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Cache-Auto"] == "prepared"
    assert response.headers["X-Irodori-Condition-Cache-Id"].startswith("cond_")
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 1


def test_explicit_cache_mode_off_preserves_legacy_pytorch_path(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "off"}),
    )

    assert (
        openai_api_server._speech_cache_mode({"cache_mode": "off"})
        == openai_api_server.CACHE_MODE_OFF
    )
    assert_legacy_pcm_success(response)
    assert "X-Irodori-Cache-Auto" not in response.headers
    assert calls["get_runtime"] == 1
    assert len(runtime.requests) == 1


def test_reference_cache_endpoint_is_metadata_only_after_runtime_loaded(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    speech_response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "off"}),
    )
    assert_legacy_pcm_success(speech_response)
    assert calls["get_runtime"] == 1
    assert not runtime.resident_store

    response = client.post(
        "/v1/tts/reference-caches",
        json={"source": {"type": "server_default"}},
    )

    assert response.status_code == 201
    reference = response.json()
    assert reference["memory_bytes"] == 0
    assert reference["memory_bytes_estimated"] is True
    assert reference["metadata"] == {}
    assert reference["shapes"] == {
        "ref_len": None,
        "speaker_context_len": 1,
        "speaker_dim": None,
    }
    assert calls["get_runtime"] == 1
    assert not runtime.resident_store


@pytest.mark.parametrize("voice", [None, "default", "alloy", "nova", "garbage"])
def test_openai_voice_field_is_ignored_and_reports_server_default(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    voice: str | None,
) -> None:
    client, _runtime, _calls = client_runtime
    payload = speech_payload(irodori={"cache_mode": "off"})
    if voice is not None:
        payload["voice"] = voice

    response = client.post("/v1/audio/speech", json=payload)

    assert_legacy_pcm_success(response)
    assert response.headers["X-Irodori-Voice-Resolved"] == "server-default"


def test_require_without_cache_id_returns_validation_error_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "require"}),
    )

    assert_cache_error(response, 400, "cache_validation_error", cache_mode="require")
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def test_require_with_missing_cache_id_returns_not_found_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": "cond_missing"},
        ),
    )

    assert_cache_error(
        response,
        404,
        "cache_not_found",
        cache_id="cond_missing",
        cache_mode="require",
    )
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def test_require_with_mismatched_condition_returns_conflict_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client, input_text="different text")

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )

    assert_cache_error(
        response,
        409,
        "cache_mismatch",
        cache_id=str(condition["id"]),
        cache_mode="require",
    )
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


@pytest.mark.parametrize(
    "reference_overrides",
    [
        {"speaker_context_len": 2},
        {"reference_fingerprint": "server_default:/different/reference.wav"},
    ],
)
def test_require_rejects_condition_built_for_different_reference_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    reference_overrides: dict[str, object],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(
        client,
        reference_overrides=reference_overrides,
    )

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )

    assert_cache_error(
        response,
        409,
        "cache_mismatch",
        cache_id=str(condition["id"]),
        cache_mode="require",
    )
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def test_require_with_matching_condition_cache_uses_fast_path(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/L16"
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == condition["id"]
    assert response.content == b"\x00" * 16
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 1


def test_auto_without_cache_id_auto_prepares_condition_cache(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "auto"}),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Cache-Auto"] == "prepared"
    assert response.headers["X-Irodori-Condition-Cache-Id"].startswith("cond_")
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 1


def test_auto_second_identical_request_reuses_condition_cache(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, _runtime, _calls = client_runtime

    first = client.post("/v1/audio/speech", json=speech_payload())
    second = client.post("/v1/audio/speech", json=speech_payload())

    assert first.status_code == 200
    assert second.status_code == 200
    first_condition_id = first.headers["X-Irodori-Condition-Cache-Id"]
    second_condition_id = second.headers["X-Irodori-Condition-Cache-Id"]
    assert first_condition_id == second_condition_id
    assert first.headers["X-Irodori-Cache-Auto"] == "prepared"
    assert second.headers["X-Irodori-Cache-Auto"] == "reused"
    assert second.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"


def test_auto_seconds_within_same_bucket_reuse_condition_id(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, _runtime, _calls = client_runtime

    first = client.post("/v1/audio/speech", json=speech_payload(seconds=2.0))
    second = client.post("/v1/audio/speech", json=speech_payload(seconds=2.5))

    assert first.status_code == 200
    assert second.status_code == 200
    assert (
        first.headers["X-Irodori-Condition-Cache-Id"]
        == second.headers["X-Irodori-Condition-Cache-Id"]
    )
    assert first.headers["X-Irodori-Cache-Auto"] == "prepared"
    assert second.headers["X-Irodori-Cache-Auto"] == "reused"


def test_auto_oversize_text_falls_back_to_pytorch_with_reason(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime, _calls = client_runtime

    def tokenize_for_bucket(text: str) -> tuple[int, str]:
        del text
        return 999, "sha256:oversize"

    monkeypatch.setattr(runtime, "tokenize_for_bucket", tokenize_for_bucket)

    response = client.post("/v1/audio/speech", json=speech_payload())

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Cache-Auto"] == "miss-fallback"
    assert response.headers["X-Irodori-Fallback-Reason"] == "oversize_t"
    assert "X-Irodori-Condition-Cache-Id" not in response.headers
    assert len(runtime.requests) == 1
    assert runtime.fast_requests == []


def test_cache_mode_off_does_not_prepare_condition_cache(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, _calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "off"}),
    )
    metrics = client.get("/v1/tts/cache-metrics").json()["cache_manager"]

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert "X-Irodori-Cache-Auto" not in response.headers
    assert "X-Irodori-Condition-Cache-Id" not in response.headers
    assert metrics["condition_count"] == 0
    assert metrics["reference_count"] == 0
    assert len(runtime.requests) == 1
    assert runtime.fast_requests == []


def test_auto_with_missing_condition_cache_falls_back_to_legacy_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "auto", "cache_id": "cond_missing"},
        ),
    )

    assert_legacy_pcm_success(response)
    assert calls["get_runtime"] == 1
    assert len(runtime.requests) == 1


@pytest.mark.parametrize(
    "exception_type",
    [
        openai_api_server.CacheNotFoundError,
        openai_api_server.CacheExpiredError,
    ],
)
def test_auto_falls_back_when_cache_disappears_during_final_validation(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[Exception],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client)

    def require_condition_cache(cache_id: str, expected_request: object) -> None:
        del expected_request
        raise exception_type(f"condition cache unavailable: {cache_id}")

    monkeypatch.setattr(
        client.app.state.coreml_cache_manager,
        "require_condition_cache",
        require_condition_cache,
    )

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "auto", "cache_id": condition["id"]},
        ),
    )

    assert_legacy_pcm_success(response)
    assert calls["get_runtime"] == 1
    assert len(runtime.requests) == 1


def test_auto_with_mismatched_condition_returns_conflict_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client, input_text="different text")

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "auto", "cache_id": condition["id"]},
        ),
    )

    assert_cache_error(
        response,
        409,
        "cache_mismatch",
        cache_id=str(condition["id"]),
        cache_mode="auto",
    )
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def test_invalid_irodori_object_uses_cache_error_envelope(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori="not-an-object"),
    )

    assert_cache_error(response, 400, "cache_validation_error")
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


@pytest.mark.parametrize("cache_mode", ["prepare", "refresh"])
def test_prepare_and_refresh_modes_use_fast_path_with_cache_headers(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    cache_mode: str,
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": cache_mode}),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/L16"
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"].startswith("cond_")
    assert response.headers["X-Irodori-Reference-Cache-Id"].startswith("ref_")
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 1


@pytest.mark.parametrize("cache_mode", ["prepare", "refresh"])
def test_prepare_and_refresh_use_resident_reference_metadata(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    cache_mode: str,
) -> None:
    client, runtime, _calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": cache_mode}),
    )

    assert response.status_code == 200
    reference_id = response.headers["X-Irodori-Reference-Cache-Id"]
    condition_id = response.headers["X-Irodori-Condition-Cache-Id"]
    reference_response = client.get(f"/v1/tts/reference-caches/{reference_id}")
    assert reference_response.status_code == 200
    reference = reference_response.json()
    resident = runtime.resident_store[reference_id]
    assert reference["memory_bytes"] == resident.memory_bytes
    assert reference["memory_bytes_estimated"] is False
    assert reference["shapes"] == {
        "ref_len": 2,
        "speaker_context_len": 3,
        "speaker_dim": 4,
    }

    condition = client.app.state.coreml_cache_manager.peek_condition_cache(condition_id)
    assert condition.reference_cache_id == reference_id
    assert condition.speaker_context_len == 3
    assert runtime.fast_resident_references == [resident]


@pytest.mark.parametrize("hit_mode", ["require", "auto"])
def test_prepared_resident_speech_cache_is_accepted_by_hit_modes(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    hit_mode: str,
) -> None:
    client, runtime, calls = client_runtime

    prepare_response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "prepare"}),
    )
    assert prepare_response.status_code == 200
    condition_id = prepare_response.headers["X-Irodori-Condition-Cache-Id"]
    reference_id = prepare_response.headers["X-Irodori-Reference-Cache-Id"]
    resident = runtime.resident_store[reference_id]
    assert runtime.fast_resident_references == [resident]

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": hit_mode, "cache_id": condition_id}),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/L16"
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == condition_id
    assert response.headers["X-Irodori-Reference-Cache-Id"] == reference_id
    if hit_mode == "auto":
        assert response.headers["X-Irodori-Cache-Auto"] == "hit"
    assert calls["get_runtime"] == 2
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 2
    assert runtime.fast_resident_references == [resident, resident]


@pytest.mark.parametrize("hit_mode", ["require", "auto"])
def test_prepare_upgrades_same_id_metadata_reference_to_resident_metadata(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    settings: ServerSettings,
    hit_mode: str,
) -> None:
    client, runtime, _calls = client_runtime
    metadata_response = client.post(
        "/v1/tts/reference-caches",
        json={
            "source": {"type": "server_default"},
            "reference_fingerprint": openai_api_server._server_reference_fingerprint(
                settings,
            ),
            "speaker_context_len": 3,
            "memory_bytes": 1,
            "metadata": {"source": {"type": "server_default"}},
        },
    )
    assert metadata_response.status_code == 201
    metadata_reference = metadata_response.json()
    reference_id = metadata_reference["id"]
    assert metadata_reference["memory_bytes_estimated"] is True
    assert metadata_reference["metadata"].get("resident") is None

    prepare_response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "prepare"}),
    )

    assert prepare_response.status_code == 200
    assert prepare_response.headers["X-Irodori-Reference-Cache-Id"] == reference_id
    condition_id = prepare_response.headers["X-Irodori-Condition-Cache-Id"]
    upgraded_response = client.get(f"/v1/tts/reference-caches/{reference_id}")
    assert upgraded_response.status_code == 200
    upgraded_reference = upgraded_response.json()
    resident = runtime.resident_store[reference_id]
    assert upgraded_reference["memory_bytes"] == resident.memory_bytes
    assert upgraded_reference["memory_bytes_estimated"] is False
    assert upgraded_reference["metadata"]["resident"] is True
    assert upgraded_reference["shapes"] == {
        "ref_len": 2,
        "speaker_context_len": 3,
        "speaker_dim": 4,
    }

    hit_response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": hit_mode, "cache_id": condition_id}),
    )
    assert hit_response.status_code == 200
    assert hit_response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert hit_response.headers["X-Irodori-Reference-Cache-Id"] == reference_id
    if hit_mode == "auto":
        assert hit_response.headers["X-Irodori-Cache-Auto"] == "hit"
    assert runtime.requests == []
    assert runtime.fast_resident_references == [resident, resident]


def test_prepare_with_multi_segment_speech_returns_conflict(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    long_text = " ".join(["longtext"] * 40)

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": long_text,
            "response_format": "pcm",
            "irodori": {"cache_mode": "prepare"},
        },
    )

    assert_cache_error(response, 409, "cache_mismatch", cache_mode="prepare")
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def test_prepare_rejects_mismatched_cache_id(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "prepare", "cache_id": "cond_unrelated"},
        ),
    )

    assert_cache_error(
        response,
        409,
        "cache_mismatch",
        cache_id="cond_unrelated",
        cache_mode="prepare",
    )
    assert calls["get_runtime"] == 1
    assert runtime.requests == []


def test_prepare_with_mismatched_cache_id_does_not_mutate_cache(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    metrics_before = client.get("/v1/tts/cache-metrics").json()
    assert metrics_before["cache_manager"]["reference_count"] == 0
    assert metrics_before["cache_manager"]["condition_count"] == 0

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "prepare", "cache_id": "cond_unrelated"},
        ),
    )
    assert_cache_error(response, 409, "cache_mismatch")

    metrics_after = client.get("/v1/tts/cache-metrics").json()
    assert metrics_after["cache_manager"]["reference_count"] == 0
    assert metrics_after["cache_manager"]["condition_count"] == 0
    assert metrics_after["cache_manager"]["evictions"] == 0
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert not runtime.resident_store


def test_prepare_condition_failure_cleans_new_reference_and_resident(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime, calls = client_runtime
    manager = client.app.state.coreml_cache_manager

    def fail_prepare_condition_cache(request: object, cache_mode: str = "create_or_reuse") -> None:
        del request, cache_mode
        raise openai_api_server.CacheValidationError("condition prepare failed")

    monkeypatch.setattr(manager, "prepare_condition_cache", fail_prepare_condition_cache)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "prepare"}),
    )

    assert_cache_error(response, 400, "cache_validation_error", cache_mode="prepare")
    metrics = client.get("/v1/tts/cache-metrics").json()["cache_manager"]
    assert metrics["reference_count"] == 0
    assert metrics["condition_count"] == 0
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert not runtime.resident_store


def test_refresh_condition_failure_cleans_partial_reference_and_resident(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, runtime, calls = client_runtime
    manager = client.app.state.coreml_cache_manager

    def fail_prepare_condition_cache(request: object, cache_mode: str = "create_or_reuse") -> None:
        del request, cache_mode
        raise openai_api_server.CacheValidationError("condition prepare failed")

    monkeypatch.setattr(manager, "prepare_condition_cache", fail_prepare_condition_cache)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "refresh"}),
    )

    assert_cache_error(response, 400, "cache_validation_error", cache_mode="refresh")
    metrics = client.get("/v1/tts/cache-metrics").json()["cache_manager"]
    assert metrics["reference_count"] == 0
    assert metrics["condition_count"] == 0
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert not runtime.resident_store
    assert runtime.deleted_resident_ids


def test_prepare_does_not_commit_metadata_if_final_resident_id_changes(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    class MismatchedFinalResidentRuntime(FakeRuntime):
        def prepare_default_reference_tensors(
            self,
            ref_wav: str | Path,
            max_ref_seconds: float | None,
            *,
            reference_cache_id: str | None = None,
            reference_fingerprint: str | None = None,
            force_refresh: bool = False,
        ) -> openai_api_server.ResidentReferenceTensors:
            resident = super().prepare_default_reference_tensors(
                ref_wav,
                max_ref_seconds,
                reference_cache_id=reference_cache_id,
                reference_fingerprint=reference_fingerprint,
                force_refresh=force_refresh,
            )
            if str(reference_cache_id or "").startswith("ref_"):
                speaker_state = torch.ones((1, 4, 4), dtype=torch.float32)
                speaker_mask = torch.ones((1, 4), dtype=torch.bool)
                memory_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        resident.ref_latent,
                        resident.ref_mask,
                        speaker_state,
                        speaker_mask,
                    )
                )
                resident = openai_api_server.ResidentReferenceTensors(
                    reference_cache_id=resident.reference_cache_id,
                    ref_latent=resident.ref_latent,
                    ref_mask=resident.ref_mask,
                    speaker_state=speaker_state,
                    speaker_mask=speaker_mask,
                    reference_fingerprint=resident.reference_fingerprint,
                    source=resident.source,
                    ref_len=resident.ref_len,
                    speaker_context_len=4,
                    speaker_dim=4,
                    memory_bytes=int(memory_bytes),
                    device=resident.device,
                    dtype=resident.dtype,
                    created_at=resident.created_at,
                    last_used_at=resident.last_used_at,
                )
                self.resident_store[resident.reference_cache_id] = resident
            return resident

    runtime = MismatchedFinalResidentRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> MismatchedFinalResidentRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/audio/speech",
            json=speech_payload(irodori={"cache_mode": "prepare"}),
        )

        assert_cache_error(response, 409, "cache_mismatch", cache_mode="prepare")
        metrics = client.get("/v1/tts/cache-metrics").json()["cache_manager"]
        assert metrics["reference_count"] == 0
        assert metrics["condition_count"] == 0
        assert not runtime.resident_store


def test_refresh_with_mismatched_cache_id_does_not_delete_existing_caches(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    prepare_response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "prepare"}),
    )
    assert prepare_response.status_code == 200
    prepared_condition_id = prepare_response.headers["X-Irodori-Condition-Cache-Id"]
    prepared_reference_id = prepare_response.headers["X-Irodori-Reference-Cache-Id"]

    metrics_before = client.get("/v1/tts/cache-metrics").json()["cache_manager"]
    assert metrics_before["reference_count"] == 1
    assert metrics_before["condition_count"] == 1

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "refresh", "cache_id": "cond_unrelated"},
        ),
    )
    assert_cache_error(response, 409, "cache_mismatch")

    # The previously prepared caches must still be intact.
    assert client.get(f"/v1/tts/condition-caches/{prepared_condition_id}").status_code == 200
    assert client.get(f"/v1/tts/reference-caches/{prepared_reference_id}").status_code == 200

    metrics_after = client.get("/v1/tts/cache-metrics").json()["cache_manager"]
    assert metrics_after["reference_count"] == 1
    assert metrics_after["condition_count"] == 1


def test_required_cache_is_marked_recently_used(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client)
    assert calls["get_runtime"] == 0

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )
    assert response.status_code == 200

    get_response = client.get(f"/v1/tts/condition-caches/{condition['id']}")
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["hit_count"] >= 1
    assert body["last_used_at"] is not None


def _create_condition_cache(
    client: TestClient,
    *,
    cfg: dict[str, object],
    input_text: str = SPEECH_TEXT,
    caption: str = CAPTION,
    seconds: float = 2.0,
) -> dict[str, object]:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": input_text,
            "caption": caption,
            "seconds": seconds,
            "cfg": cfg,
        },
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.parametrize(
    "cfg",
    [
        {"mode": "joint", "scale_text": 4.0, "scale_speaker": 4.0, "scale_caption": 0.0},
        {"mode": "alternating", "scale_text": 3.0, "scale_speaker": 5.0},
    ],
)
def test_speech_require_with_joint_or_alternating_cfg_uses_fast_path(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    cfg: dict[str, object],
) -> None:
    client, runtime, calls = client_runtime
    condition = _create_condition_cache(client, cfg=cfg)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"], "cfg": cfg},
        ),
    )

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == condition["id"]
    assert calls["get_runtime"] == 1
    assert runtime.requests == []
    assert len(runtime.fast_requests) == 1
    request, _handle = runtime.fast_requests[0]
    assert request.cfg_guidance_mode == cfg["mode"]


def test_speech_require_with_speaker_kv_scale_passes_to_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    cfg = {
        "mode": "independent",
        "speaker_kv_scale": 2.5,
        "speaker_kv_min_t": 0.7,
        "speaker_kv_max_layers": 4,
    }
    condition = _create_condition_cache(client, cfg=cfg)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"], "cfg": cfg},
        ),
    )

    assert response.status_code == 200
    assert calls["get_runtime"] == 1
    assert len(runtime.fast_requests) == 1
    request, _handle = runtime.fast_requests[0]
    assert request.speaker_kv_scale == 2.5
    assert request.speaker_kv_min_t == 0.7
    assert request.speaker_kv_max_layers == 4


def test_speech_rejects_positive_caption_cfg_for_speech_cache_validation(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={
                "cache_mode": "prepare",
                "cfg": {"mode": "independent", "scale_caption": 2.0},
            },
        ),
    )

    assert_cache_error(response, 400, "cache_validation_error", cache_mode="prepare")
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def test_cache_id_with_multi_segment_speech_returns_conflict_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    long_text = " ".join(["longtext"] * 40)

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": long_text,
            "response_format": "pcm",
            "irodori": {"cache_mode": "require", "cache_id": "cond_any"},
        },
    )

    assert_cache_error(
        response,
        409,
        "cache_mismatch",
        cache_id="cond_any",
        cache_mode="require",
    )
    assert calls["get_runtime"] == 0
    assert runtime.requests == []


def _client_for_settings(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
    runtime: FakeRuntime,
) -> TestClient:
    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)
    return TestClient(create_app(settings))


def test_auto_prepare_default_cache_false_falls_back_without_condition_cache(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    from dataclasses import replace as _dc_replace

    custom_settings = _dc_replace(settings, auto_prepare_default_cache=False)
    runtime = FakeRuntime()
    with _client_for_settings(monkeypatch, custom_settings, runtime) as client:
        response = client.post("/v1/audio/speech", json=speech_payload())

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Cache-Auto"] == "miss-fallback"
    assert response.headers["X-Irodori-Fallback-Reason"] == "auto-prepare-disabled"
    assert "X-Irodori-Condition-Cache-Id" not in response.headers
    assert "X-Irodori-Reference-Cache-Id" not in response.headers
    assert len(runtime.fast_requests) == 0
    assert len(runtime.requests) == 1


def test_enable_resident_reference_cache_false_falls_back_for_auto(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    from dataclasses import replace as _dc_replace

    custom_settings = _dc_replace(settings, enable_resident_reference_cache=False)
    runtime = FakeRuntime()
    with _client_for_settings(monkeypatch, custom_settings, runtime) as client:
        response = client.post("/v1/audio/speech", json=speech_payload())

    assert response.status_code == 200
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Cache-Auto"] == "miss-fallback"
    assert response.headers["X-Irodori-Fallback-Reason"] == "resident_reference_disabled"
    assert "X-Irodori-Condition-Cache-Id" not in response.headers
    assert len(runtime.fast_requests) == 0


def test_auto_internal_condition_cache_request_uses_default_ttl(
    settings: ServerSettings,
) -> None:
    from dataclasses import replace as _dc_replace

    custom_settings = _dc_replace(settings, condition_cache_default_ttl_seconds=300.0)
    runtime = FakeRuntime()
    bucket = openai_api_server.CoreMLConditionBucket(
        sequence_length=100,
        text_len=64,
        speaker_context_len_bucket=160,
    )
    planning = openai_api_server.AutoSpeechPlanningContext(
        segment_text="hi",
        normalized_text="hi",
        seconds=2.0,
        sample_rate=24_000,
        hop_length=512,
        latent_patch_size=2,
        tokenizer_fingerprint="tokenizer:fake",
        token_len=4,
        token_ids_hash="sha256:fake",
        patched_steps=10,
    )
    fake_handle = SimpleNamespace(id="ref_test", speaker_context_len=3)
    request = openai_api_server._auto_internal_condition_cache_request(
        irodori=None,
        settings=custom_settings,
        runtime=runtime,
        reference_handle=fake_handle,
        bucket=bucket,
        planning=planning,
        caption=None,
    )
    assert request.ttl_seconds == pytest.approx(300.0)


def test_auto_internal_condition_cache_request_omits_ttl_when_none(
    settings: ServerSettings,
) -> None:
    from dataclasses import replace as _dc_replace

    custom_settings = _dc_replace(settings, condition_cache_default_ttl_seconds=None)
    runtime = FakeRuntime()
    bucket = openai_api_server.CoreMLConditionBucket(
        sequence_length=100,
        text_len=64,
        speaker_context_len_bucket=160,
    )
    planning = openai_api_server.AutoSpeechPlanningContext(
        segment_text="hi",
        normalized_text="hi",
        seconds=2.0,
        sample_rate=24_000,
        hop_length=512,
        latent_patch_size=2,
        tokenizer_fingerprint="tokenizer:fake",
        token_len=4,
        token_ids_hash="sha256:fake",
        patched_steps=10,
    )
    fake_handle = SimpleNamespace(id="ref_test", speaker_context_len=3)
    request = openai_api_server._auto_internal_condition_cache_request(
        irodori=None,
        settings=custom_settings,
        runtime=runtime,
        reference_handle=fake_handle,
        bucket=bucket,
        planning=planning,
        caption=None,
    )
    assert request.ttl_seconds is None


def test_default_auto_response_emits_alias_headers(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, _runtime, _calls = client_runtime
    response = client.post("/v1/audio/speech", json=speech_payload())
    assert response.status_code == 200
    assert (
        response.headers["X-Irodori-Backend"]
        == response.headers["X-Irodori-Denoiser-Backend"]
    )
    assert (
        response.headers["X-Irodori-Cache-Condition-Id"]
        == response.headers["X-Irodori-Condition-Cache-Id"]
    )
    assert (
        response.headers["X-Irodori-Cache-Reference-Id"]
        == response.headers["X-Irodori-Reference-Cache-Id"]
    )


class _ConfigureRecordingRuntime:
    def __init__(self) -> None:
        self.configure_calls: list[dict[str, Any]] = []

    def configure_resident_speaker_kv(
        self,
        *,
        enabled: bool | None = None,
        max_entries: int | None = None,
    ) -> None:
        self.configure_calls.append({"enabled": enabled, "max_entries": max_entries})


def test_runtime_state_get_runtime_applies_resident_speaker_kv_settings(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    settings = dataclasses.replace(
        settings,
        enable_resident_speaker_kv=False,
        max_resident_speaker_kv_buckets=1,
    )
    fake_runtime = _ConfigureRecordingRuntime()

    def fake_get_cached_runtime(_key: Any) -> tuple[Any, bool]:
        return fake_runtime, True

    monkeypatch.setattr(
        openai_api_server, "get_cached_runtime", fake_get_cached_runtime
    )
    monkeypatch.setattr(
        openai_api_server.RuntimeState,
        "runtime_key",
        lambda self: "fake-runtime-key",
    )

    state = openai_api_server.RuntimeState(settings)
    first = state.get_runtime()
    assert first is fake_runtime
    assert fake_runtime.configure_calls == [
        {"enabled": False, "max_entries": 1},
    ]

    state.get_runtime()
    assert len(fake_runtime.configure_calls) == 1


def test_lifespan_warmup_default_condition_cache_prepares_condition_cache(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = FakeRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    warmup_settings = dataclasses.replace(
        settings,
        preload=True,
        default_condition_cache_prepare_text="テスト。",
    )
    app = create_app(warmup_settings)
    with TestClient(app):
        cache_manager = app.state.coreml_cache_manager
        snap = cache_manager.metrics_snapshot()
        assert snap["reference_count"] >= 1
        assert snap["condition_count"] >= 1

    out = capsys.readouterr().out
    assert "default condition cache warmup" in out
    assert "prepared=" in out


def test_lifespan_strict_coreml_warmup_raises_when_condition_handle_missing(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    runtime = FakeRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    def all_oversize(
        irodori: dict[str, Any] | None,
        segments: tuple[openai_api_server.SpeechSegment, ...],
        runtime: Any,
    ) -> list[openai_api_server.AutoBucketResolution]:
        del irodori, runtime
        return [
            openai_api_server.AutoBucketResolution(
                bucket=None,
                reason="oversize_t",
                attempted=">2048_>256",
                planning=None,
            )
            for _ in segments
        ]

    monkeypatch.setattr(
        openai_api_server,
        "_resolve_auto_bucket_resolutions",
        all_oversize,
    )

    warmup_settings = dataclasses.replace(
        settings,
        preload=True,
        strict_coreml=True,
        default_condition_cache_prepare_text="テスト。",
    )
    app = create_app(warmup_settings)
    with pytest.raises(openai_api_server.CoreMLStatefulUnavailableError):
        with TestClient(app):
            pass
