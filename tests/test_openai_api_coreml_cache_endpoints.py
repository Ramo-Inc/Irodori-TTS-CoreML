from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
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
def client(monkeypatch: pytest.MonkeyPatch, settings: ServerSettings) -> Iterator[TestClient]:
    def fail_get_runtime(self: openai_api_server.RuntimeState) -> None:
        raise AssertionError("cache metadata endpoints must not load the synthesis runtime")

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", fail_get_runtime)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def assert_cache_error(response, status_code: int, error_type: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert "detail" not in body
    assert body["error"]["type"] == error_type
    assert isinstance(body["error"]["message"], str)


def create_reference_cache(client: TestClient) -> dict[str, object]:
    response = client.post(
        "/v1/tts/reference-caches",
        json={"source": {"type": "server_default"}},
    )
    assert response.status_code == 201
    return response.json()


def create_independent_condition_cache(
    client: TestClient,
    reference_cache_id: str,
) -> dict[str, object]:
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference_cache_id,
            "input": "cache this utterance",
            "caption": "neutral",
            "seconds": 2.0,
            "cfg": {"mode": "independent"},
        },
    )
    assert response.status_code == 201
    return response.json()


def test_reference_cache_create_reuse_get_and_resident_buckets(
    client: TestClient,
    settings: ServerSettings,
) -> None:
    create_response = client.post(
        "/v1/tts/reference-caches",
        json={"source": {"type": "server_default"}},
    )

    assert create_response.status_code == 201
    reference = create_response.json()
    assert reference["id"].startswith("ref_")
    assert reference["reused"] is False
    assert reference["model"] == settings.api_model_id
    assert reference["shapes"] == {
        "ref_len": None,
        "speaker_context_len": 1,
        "speaker_dim": None,
    }
    assert reference["layers"] == {
        "ref_latent": True,
        "speaker_state": True,
        "speaker_kv": "lazy",
    }
    assert reference["memory_bytes"] == 0
    assert reference["memory_bytes_estimated"] is True

    reuse_response = client.post(
        "/v1/tts/reference-caches",
        json={"source": {"type": "server_default"}},
    )
    assert reuse_response.status_code == 200
    assert reuse_response.json()["id"] == reference["id"]
    assert reuse_response.json()["reused"] is True

    get_response = client.get(f"/v1/tts/reference-caches/{reference['id']}")
    assert get_response.status_code == 200
    reference_get = get_response.json()
    assert "reused" not in reference_get
    assert reference_get["resident_buckets"] == []

    condition = create_independent_condition_cache(client, str(reference["id"]))

    get_after_condition_response = client.get(f"/v1/tts/reference-caches/{reference['id']}")
    assert get_after_condition_response.status_code == 200
    assert get_after_condition_response.json()["resident_buckets"] == [condition["bucket_id"]]


def test_reference_delete_rejects_dependents_without_cascade_then_cascades(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)
    condition = create_independent_condition_cache(client, str(reference["id"]))

    conflict_response = client.delete(
        f"/v1/tts/reference-caches/{reference['id']}",
        params={"cascade": "false"},
    )
    assert_cache_error(conflict_response, 409, "cache_mismatch")

    delete_response = client.delete(f"/v1/tts/reference-caches/{reference['id']}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""

    missing_condition_response = client.get(f"/v1/tts/condition-caches/{condition['id']}")
    assert_cache_error(missing_condition_response, 404, "cache_not_found")


def test_reference_cache_rejects_base64_source_without_detail_wrapper(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/tts/reference-caches",
        json={"source": {"type": "base64", "data": "AAAA"}},
    )

    assert_cache_error(response, 400, "cache_validation_error")


@pytest.mark.parametrize(
    "path",
    [
        "/v1/tts/reference-caches",
        "/v1/tts/condition-caches",
    ],
)
def test_cache_post_missing_body_uses_cache_error_envelope(
    client: TestClient,
    path: str,
) -> None:
    response = client.post(path)

    assert_cache_error(response, 400, "cache_validation_error")


def test_reference_delete_invalid_cascade_uses_cache_error_envelope(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)

    response = client.delete(
        f"/v1/tts/reference-caches/{reference['id']}",
        params={"cascade": "maybe"},
    )

    assert_cache_error(response, 400, "cache_validation_error")


def test_condition_cache_requires_existing_reference(client: TestClient) -> None:
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": "ref_missing",
            "input": "missing reference",
            "cfg": {"mode": "independent"},
        },
    )

    assert_cache_error(response, 404, "cache_not_found")


@pytest.mark.parametrize(
    "payload_overrides",
    [
        {},
        {"input": 123},
        {"input": "valid input", "seconds": -1},
        {"input": "valid input", "caption": "calm", "instructions": "bright"},
    ],
)
def test_condition_cache_rejects_invalid_condition_metadata(
    client: TestClient,
    payload_overrides: dict[str, object],
) -> None:
    reference = create_reference_cache(client)
    payload = {
        "reference_cache_id": reference["id"],
        "cfg": {"mode": "independent"},
    }
    payload.update(payload_overrides)

    response = client.post("/v1/tts/condition-caches", json=payload)

    assert_cache_error(response, 400, "cache_validation_error")


def test_independent_condition_cache_create_reuse_and_get(client: TestClient) -> None:
    reference = create_reference_cache(client)
    payload = {
        "reference_cache_id": reference["id"],
        "input": "cache this utterance",
        "caption": "neutral",
        "seconds": 2.0,
        "cfg": {"mode": "independent"},
    }

    create_response = client.post("/v1/tts/condition-caches", json=payload)

    assert create_response.status_code == 201
    condition = create_response.json()
    assert condition["id"].startswith("cond_")
    assert condition["reused"] is False
    assert condition["bucket_id"] == "S100_T256_R160_independent_text_speaker3"
    assert condition["state_layout"] == "per_layer_text_speaker_context_v1"
    assert condition["memory_bytes"] == 102_236_160
    assert condition["shapes"]["branches_active"] == 3

    reuse_payload = {
        "reference_cache_id": reference["id"],
        "input": " cache this utterance ",
        "instruction": " neutral ",
        "seconds": 2,
        "cfg": {"mode": "independent"},
    }
    reuse_response = client.post("/v1/tts/condition-caches", json=reuse_payload)
    assert reuse_response.status_code == 200
    assert reuse_response.json()["id"] == condition["id"]
    assert reuse_response.json()["reused"] is True

    get_response = client.get(f"/v1/tts/condition-caches/{condition['id']}")
    assert get_response.status_code == 200
    condition_get = get_response.json()
    assert condition_get["cfg"]["mode"] == "independent"
    assert condition_get["cfg"]["branches_active"] == 3
    assert condition_get["resident_states"] == ["cond1", "independent_text_speaker3"]


def test_cond_only_condition_cache_uses_single_branch(client: TestClient) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "single branch",
            "cfg": {"mode": "cond"},
        },
    )

    assert response.status_code == 201
    condition = response.json()
    assert condition["bucket_id"] == "S100_T256_R160_cond1"
    assert condition["shapes"]["branches_active"] == 1


def test_condition_cache_omitted_cfg_reuses_explicit_cond_cfg(client: TestClient) -> None:
    reference = create_reference_cache(client)
    base_payload = {
        "reference_cache_id": reference["id"],
        "input": "canonical cond cfg",
    }

    create_response = client.post("/v1/tts/condition-caches", json=base_payload)
    assert create_response.status_code == 201
    condition = create_response.json()
    assert condition["bucket_id"] == "S100_T256_R160_cond1"

    reuse_response = client.post(
        "/v1/tts/condition-caches",
        json={**base_payload, "cfg": {"mode": "cond"}},
    )
    assert reuse_response.status_code == 200
    assert reuse_response.json()["id"] == condition["id"]
    assert reuse_response.json()["reused"] is True


def test_condition_cache_canonicalizes_independent_cfg_defaults(client: TestClient) -> None:
    reference = create_reference_cache(client)
    base_payload = {
        "reference_cache_id": reference["id"],
        "input": "canonical independent cfg",
        "cfg": {"mode": "independent"},
    }

    create_response = client.post("/v1/tts/condition-caches", json=base_payload)
    assert create_response.status_code == 201
    condition = create_response.json()
    assert condition["bucket_id"] == "S100_T256_R160_independent_text_speaker3"

    reuse_response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "canonical independent cfg",
            "cfg": {
                "mode": " Independent ",
                "scale_text": 3.0,
                "scale_speaker": 5,
                "scale_caption": 0,
                "min_t": 0.5,
                "max_t": 1.0,
            },
        },
    )
    assert reuse_response.status_code == 200
    assert reuse_response.json()["id"] == condition["id"]
    assert reuse_response.json()["reused"] is True


def test_condition_cache_rejects_stale_supplied_condition_fingerprint(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)
    payload = {
        "reference_cache_id": reference["id"],
        "input": "fingerprint source A",
        "cfg": {"mode": "independent"},
    }
    create_response = client.post("/v1/tts/condition-caches", json=payload)
    assert create_response.status_code == 201
    condition = create_response.json()
    canonical_fingerprint = condition["condition_fingerprint"]

    exact_response = client.post(
        "/v1/tts/condition-caches",
        json={**payload, "condition_fingerprint": canonical_fingerprint},
    )
    assert exact_response.status_code == 200
    assert exact_response.json()["id"] == condition["id"]
    assert exact_response.json()["reused"] is True

    stale_response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "fingerprint source B",
            "cfg": {"mode": "independent"},
            "condition_fingerprint": canonical_fingerprint,
        },
    )
    assert_cache_error(stale_response, 400, "cache_validation_error")


@pytest.mark.parametrize(
    "cfg",
    [
        {"mode": "independent", "scale_text": -0.1},
        {"mode": "independent", "scale_speaker": "5"},
        {"mode": "independent", "scale_caption": -1},
        {"mode": "independent", "min_t": -0.1},
        {"mode": "independent", "max_t": 1.1},
        {"mode": "independent", "min_t": 0.8, "max_t": 0.2},
    ],
)
def test_condition_cache_rejects_invalid_cfg_numbers(
    client: TestClient,
    cfg: dict[str, object],
) -> None:
    reference = create_reference_cache(client)

    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "invalid cfg",
            "cfg": cfg,
        },
    )

    assert_cache_error(response, 400, "cache_validation_error")


def test_condition_cache_delete_then_get_returns_not_found(client: TestClient) -> None:
    reference = create_reference_cache(client)
    condition = create_independent_condition_cache(client, str(reference["id"]))

    delete_response = client.delete(f"/v1/tts/condition-caches/{condition['id']}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""

    get_response = client.get(f"/v1/tts/condition-caches/{condition['id']}")
    assert_cache_error(get_response, 404, "cache_not_found")

    reference_response = client.get(f"/v1/tts/reference-caches/{reference['id']}")
    assert reference_response.status_code == 200
    assert reference_response.json()["resident_buckets"] == []
