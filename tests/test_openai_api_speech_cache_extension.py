from __future__ import annotations

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
) -> Iterator[tuple[TestClient, FakeRuntime, dict[str, int]]]:
    runtime = FakeRuntime()
    calls = {"get_runtime": 0}

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeRuntime:
        del self
        calls["get_runtime"] += 1
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
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.content == b"\x00" * 16


def test_no_irodori_uses_legacy_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post("/v1/audio/speech", json=speech_payload())

    assert_legacy_pcm_success(response)
    assert calls["get_runtime"] == 1
    assert len(runtime.requests) == 1
    assert runtime.requests[0].text == SPEECH_TEXT
    assert runtime.requests[0].caption == CAPTION


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


def test_auto_without_cache_id_falls_back_to_legacy_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori={"cache_mode": "auto"}),
    )

    assert_legacy_pcm_success(response)
    assert calls["get_runtime"] == 1
    assert len(runtime.requests) == 1


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


@pytest.mark.parametrize(
    "irodori",
    [
        "not-an-object",
        {"cache_mode": "prepare"},
        {"cache_mode": "refresh"},
    ],
)
def test_invalid_irodori_or_unsupported_modes_use_cache_error_envelope(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
    irodori: object,
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(irodori=irodori),
    )

    assert_cache_error(response, 400, "cache_validation_error")
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
