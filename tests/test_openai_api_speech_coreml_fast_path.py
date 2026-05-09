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
MODULE_SPEC = importlib.util.spec_from_file_location("openai_api_server_fast_path", MODULE_PATH)
assert MODULE_SPEC is not None
openai_api_server = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = openai_api_server
MODULE_SPEC.loader.exec_module(openai_api_server)

from irodori_tts.coreml_stateful import CoreMLStatefulUnavailableError  # noqa: E402

ServerSettings = openai_api_server.ServerSettings
create_app = openai_api_server.create_app

SPEECH_TEXT = "cache this utterance"
CAPTION = "neutral"


class FakeRuntime:
    def __init__(self) -> None:
        self.legacy_requests: list[Any] = []
        self.fast_requests: list[tuple[Any, Any]] = []
        self.fast_exception: Exception | None = None
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
        if self.fast_exception is not None:
            raise self.fast_exception
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
            "input": SPEECH_TEXT,
            "caption": CAPTION,
            "seconds": 2.0,
            "cfg": {"mode": "independent"},
        },
    )
    assert response.status_code == 201
    return response.json()


def assert_pcm_success(response: Any) -> None:
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/L16"
    assert response.headers["X-Irodori-Voice-Resolved"] == "server-default"
    assert response.content == b"\x00" * 16


def assert_cache_error(response: Any, status_code: int, error_type: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert "detail" not in body
    assert body["error"]["type"] == error_type


def test_no_irodori_uses_legacy_synthesize_and_pytorch_header(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post("/v1/audio/speech", json=speech_payload())

    assert_pcm_success(response)
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert response.headers["X-Irodori-Cache-Auto"] == "miss-fallback"
    assert "X-Irodori-Condition-Cache-Id" not in response.headers
    assert calls["get_runtime"] == 1
    assert len(runtime.legacy_requests) == 1
    assert runtime.fast_requests == []


def test_require_matching_cache_uses_coreml_stateful_fast_path(
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

    assert_pcm_success(response)
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == condition["id"]
    assert calls["get_runtime"] == 1
    assert runtime.legacy_requests == []
    assert len(runtime.fast_requests) == 1
    request, handle = runtime.fast_requests[0]
    assert request.text == SPEECH_TEXT
    assert handle.id == condition["id"]


def test_auto_matching_cache_uses_coreml_stateful_fast_path(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client)

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "auto", "cache_id": condition["id"]},
        ),
    )

    assert_pcm_success(response)
    assert response.headers["X-Irodori-Denoiser-Backend"] == "coreml-stateful"
    assert response.headers["X-Irodori-Condition-Cache-Id"] == condition["id"]
    assert calls["get_runtime"] == 1
    assert runtime.legacy_requests == []
    assert len(runtime.fast_requests) == 1


def test_require_matching_cache_coreml_unavailable_returns_503_without_legacy(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client)
    runtime.fast_exception = CoreMLStatefulUnavailableError("coremltools is unavailable")

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )

    assert_cache_error(response, 503, "coreml_backend_unavailable")
    assert calls["get_runtime"] == 1
    assert runtime.legacy_requests == []
    assert runtime.fast_requests == []


def test_auto_matching_cache_coreml_unavailable_falls_back_to_legacy(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    condition = create_matching_condition_cache(client)
    runtime.fast_exception = CoreMLStatefulUnavailableError("coremltools is unavailable")

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "auto", "cache_id": condition["id"]},
        ),
    )

    assert_pcm_success(response)
    assert response.headers["X-Irodori-Denoiser-Backend"] == "pytorch"
    assert "X-Irodori-Condition-Cache-Id" not in response.headers
    assert calls["get_runtime"] == 1
    assert len(runtime.legacy_requests) == 1
    assert runtime.fast_requests == []


def test_require_missing_cache_stops_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime

    response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": "cond_missing"},
        ),
    )

    assert_cache_error(response, 404, "cache_not_found")
    assert calls["get_runtime"] == 0
    assert runtime.legacy_requests == []
    assert runtime.fast_requests == []


def test_require_mismatched_cache_stops_before_runtime(
    client_runtime: tuple[TestClient, FakeRuntime, dict[str, int]],
) -> None:
    client, runtime, calls = client_runtime
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "different text",
            "caption": CAPTION,
            "seconds": 2.0,
            "cfg": {"mode": "independent"},
        },
    )
    assert response.status_code == 201
    condition = response.json()

    speech_response = client.post(
        "/v1/audio/speech",
        json=speech_payload(
            irodori={"cache_mode": "require", "cache_id": condition["id"]},
        ),
    )

    assert_cache_error(speech_response, 409, "cache_mismatch")
    assert calls["get_runtime"] == 0
    assert runtime.legacy_requests == []
    assert runtime.fast_requests == []
