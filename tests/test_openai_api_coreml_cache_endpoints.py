from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

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


class FakeResidentRuntime:
    def __init__(self) -> None:
        self.prepare_calls: list[str | None] = []
        self.deleted: list[str] = []
        self.store: dict[str, openai_api_server.ResidentReferenceTensors] = {}

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
        self.prepare_calls.append(reference_cache_id)
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
            source=str(ref_wav),
            ref_len=2,
            speaker_context_len=3,
            speaker_dim=4,
            memory_bytes=int(memory_bytes),
            device="cpu",
            dtype="torch.float32",
            created_at=now,
            last_used_at=now,
        )
        self.store[cache_id] = resident
        return resident

    def get_resident_reference_tensors(
        self,
        reference_cache_id: str,
    ) -> openai_api_server.ResidentReferenceTensors | None:
        return self.store.get(reference_cache_id)

    def delete_resident_reference_tensors(self, reference_cache_id: str) -> bool:
        self.deleted.append(reference_cache_id)
        return self.store.pop(reference_cache_id, None) is not None

    def coreml_stateful_metrics_snapshot(self) -> list[dict[str, object]]:
        return []

    def resident_reference_metrics_snapshot(self) -> dict[str, object]:
        return {
            "count": len(self.store),
            "memory_bytes": sum(resident.memory_bytes for resident in self.store.values()),
            "entries": [
                {
                    "reference_cache_id": resident.reference_cache_id,
                    "memory_bytes": resident.memory_bytes,
                }
                for resident in self.store.values()
            ],
        }


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


def test_server_reference_fingerprint_changes_with_reference_wav_contents(
    tmp_path: Path,
    settings: ServerSettings,
) -> None:
    reference_wav = tmp_path / "rem.wav"
    reference_wav.write_bytes(b"first reference bytes")
    first_settings = replace(settings, reference_wav=reference_wav, max_ref_seconds=12.5)

    first = openai_api_server._server_reference_fingerprint(first_settings)
    reference_wav.write_bytes(b"second reference bytes")
    second = openai_api_server._server_reference_fingerprint(first_settings)
    third = openai_api_server._server_reference_fingerprint(
        replace(first_settings, max_ref_seconds=5.0),
    )

    assert first.startswith(f"server_default:{reference_wav}:sha256:")
    assert ":max_ref_seconds:12.5" in first
    assert second.startswith(f"server_default:{reference_wav}:sha256:")
    assert first != second
    assert second != third


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
            "branch_layouts": ["cond1"],
        },
    )

    assert response.status_code == 201
    condition = response.json()
    assert condition["bucket_id"] == "S100_T256_R160_cond1"
    assert condition["shapes"]["branches_active"] == 1


def test_independent_condition_cache_rejects_cond_only_branch_layout(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)

    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "invalid branch layout",
            "cfg": {"mode": "independent"},
            "branch_layouts": ["cond1"],
        },
    )

    assert_cache_error(response, 400, "cache_validation_error")


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


@pytest.mark.parametrize(
    ("cfg", "expected_bucket_id", "expected_active_branches"),
    [
        (
            {"mode": "joint", "scale_text": 4.0, "scale_speaker": 4.0, "scale_caption": 0.0},
            "S100_T256_R160_joint2",
            2,
        ),
        (
            {"mode": "alternating", "scale_text": 3.0, "scale_speaker": 5.0},
            "S100_T256_R160_alternating_speaker2",
            2,
        ),
        (
            {"mode": "alternating", "scale_text": 3.0, "scale_speaker": 0.0},
            "S100_T256_R160_alternating_text2",
            2,
        ),
    ],
)
def test_condition_cache_supports_joint_and_alternating_cfg(
    client: TestClient,
    cfg: dict[str, object],
    expected_bucket_id: str,
    expected_active_branches: int,
) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "joint or alternating cache",
            "cfg": cfg,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["bucket_id"] == expected_bucket_id
    assert body["shapes"]["branches_active"] == expected_active_branches


def test_condition_cache_supports_speaker_kv_scale_in_cfg(client: TestClient) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "speaker kv scale cache",
            "cfg": {
                "mode": "independent",
                "speaker_kv_scale": 2.5,
                "speaker_kv_min_t": 0.7,
                "speaker_kv_max_layers": 4,
            },
        },
    )

    assert response.status_code == 201


@pytest.mark.parametrize(
    "cfg",
    [
        {"mode": "independent", "speaker_kv_scale": 0.0},
        {"mode": "independent", "speaker_kv_scale": -1.0},
        {"mode": "independent", "speaker_kv_scale": 2.0, "speaker_kv_min_t": -0.1},
        {"mode": "independent", "speaker_kv_scale": 2.0, "speaker_kv_min_t": 1.5},
        {"mode": "independent", "speaker_kv_scale": 2.0, "speaker_kv_max_layers": -1},
    ],
)
def test_condition_cache_rejects_invalid_speaker_kv_settings(
    client: TestClient,
    cfg: dict[str, object],
) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "invalid speaker kv",
            "cfg": cfg,
        },
    )

    assert_cache_error(response, 400, "cache_validation_error")


def test_cache_metrics_endpoint_does_not_load_runtime(client: TestClient) -> None:
    response = client.get("/v1/tts/cache-metrics")
    assert response.status_code == 200
    body = response.json()
    assert "cache_manager" in body
    assert body["coreml_stateful_backends"] == []
    assert body["resident_references"] == {}
    assert body["resident_speaker_kv"] == {}
    snapshot = body["cache_manager"]
    assert "reference_hits" in snapshot
    assert "evictions" in snapshot


def test_cache_metrics_endpoint_exposes_resident_speaker_kv_from_loaded_runtime(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    speaker_kv_snapshot = {
        "count": 1,
        "memory_bytes": 4096,
        "max_entries": 4,
        "resident_speaker_kv_hits": 2,
        "resident_speaker_kv_misses": 1,
        "resident_speaker_kv_evictions": 0,
        "resident_speaker_kv_deletes": 0,
        "entries": [
            {
                "reference_cache_id": "ref_runtime",
                "reference_fingerprint": "fp",
                "sequence_length_bucket": 100,
                "speaker_context_len_bucket": 160,
                "branch_layout": "joint2",
                "scale_signature": "joint2:s4.0",
                "memory_bytes": 4096,
                "created_at": "2026-01-01T00:00:00+00:00",
                "last_used_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    }

    class SpeakerKvRuntime:
        def coreml_stateful_metrics_snapshot(self) -> list[dict[str, object]]:
            return []

        def resident_reference_metrics_snapshot(self) -> dict[str, object]:
            return {"count": 0, "memory_bytes": 0, "entries": []}

        def resident_speaker_kv_metrics_snapshot(self) -> dict[str, object]:
            return speaker_kv_snapshot

    runtime = SpeakerKvRuntime()

    monkeypatch.setattr(
        openai_api_server.RuntimeState,
        "runtime_if_loaded",
        property(lambda self: runtime),
    )

    def fail_get_runtime(self: openai_api_server.RuntimeState) -> None:
        raise AssertionError("cache-metrics must not load the synthesis runtime")

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", fail_get_runtime)
    with TestClient(create_app(settings)) as test_client:
        response = test_client.get("/v1/tts/cache-metrics")
        assert response.status_code == 200
        body = response.json()
        assert body["resident_speaker_kv"] == speaker_kv_snapshot
        for entry in body["resident_speaker_kv"]["entries"]:
            assert "speaker_kv" not in entry
            assert "tensors" not in entry
            assert "tensor" not in entry


def test_lifespan_prepares_default_reference_resident_metadata(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    runtime = FakeResidentRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeResidentRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)
    with TestClient(create_app(replace(settings, preload=True))) as test_client:
        resident_ids = [cache_id for cache_id in runtime.store if cache_id.startswith("ref_")]
        assert len(resident_ids) == 1
        reference_id = resident_ids[0]

        response = test_client.get(f"/v1/tts/reference-caches/{reference_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == reference_id
        assert body["memory_bytes"] == runtime.store[reference_id].memory_bytes
        assert body["memory_bytes_estimated"] is False
        assert body["shapes"] == {
            "ref_len": 2,
            "speaker_context_len": 3,
            "speaker_dim": 4,
        }
        assert body["resident_layers"] == [
            "ref_latent",
            "ref_mask",
            "speaker_state",
            "speaker_mask",
        ]

        metrics = test_client.get("/v1/tts/cache-metrics").json()
        assert metrics["resident_references"]["count"] == 1
        assert "ref_latent" not in metrics["resident_references"]["entries"][0]


def test_delete_reference_cache_prunes_loaded_runtime_resident_reference(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    runtime = FakeResidentRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> FakeResidentRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)
    with TestClient(create_app(replace(settings, preload=True))) as test_client:
        reference_id = next(cache_id for cache_id in runtime.store if cache_id.startswith("ref_"))

        response = test_client.delete(f"/v1/tts/reference-caches/{reference_id}")

        assert response.status_code == 204
        assert reference_id in runtime.deleted
        assert reference_id not in runtime.store


def test_condition_cache_rejects_bare_joint_with_unequal_default_scales(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "joint with unequal defaults",
            "cfg": {"mode": "joint"},
        },
    )

    assert_cache_error(response, 400, "cache_validation_error")


def test_condition_cache_accepts_joint_with_equal_scales(client: TestClient) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "joint with equal scales",
            "cfg": {"mode": "joint", "scale_text": 4.0, "scale_speaker": 4.0},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["bucket_id"] == "S100_T256_R160_joint2"
    assert body["state_copies"] == 1


def test_condition_cache_rejects_positive_scale_caption_universally(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "caption cfg",
            "cfg": {"mode": "independent", "scale_caption": 2.0},
        },
    )

    assert_cache_error(response, 400, "cache_validation_error")


def test_condition_cache_rejects_alternating_with_positive_scale_caption(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)
    response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "alternating with caption",
            "cfg": {
                "mode": "alternating",
                "scale_text": 3.0,
                "scale_speaker": 5.0,
                "scale_caption": 2.0,
            },
        },
    )

    assert_cache_error(response, 400, "cache_validation_error")


def test_condition_cache_state_copies_doubled_for_speaker_kv_scale(
    client: TestClient,
) -> None:
    reference = create_reference_cache(client)
    base_cfg = {"mode": "independent"}
    base_response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "memory baseline",
            "cfg": base_cfg,
        },
    )
    assert base_response.status_code == 201
    base = base_response.json()

    scaled_response = client.post(
        "/v1/tts/condition-caches",
        json={
            "reference_cache_id": reference["id"],
            "input": "memory baseline",
            "cfg": {**base_cfg, "speaker_kv_scale": 2.5},
        },
    )
    assert scaled_response.status_code == 201
    scaled = scaled_response.json()

    assert base["state_copies"] == 1
    assert scaled["state_copies"] == 2
    assert scaled["memory_bytes"] == 2 * base["memory_bytes"]
    assert scaled["id"] != base["id"]


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


class _PrecompileFailingRuntime(FakeResidentRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.precompile_calls: list[list[openai_api_server.CoreMLConditionBucket]] = []

    def precompile_coreml_stateful_buckets(
        self,
        buckets: list[openai_api_server.CoreMLConditionBucket],
    ) -> None:
        self.precompile_calls.append(list(buckets))
        raise openai_api_server.CoreMLStatefulUnavailableError("coremltools not installed")


def _warmup_settings(settings: ServerSettings, *, strict: bool) -> ServerSettings:
    bucket = openai_api_server.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )
    return replace(
        settings,
        preload=True,
        warmup_buckets=(bucket,),
        strict_coreml=strict,
    )


def test_lifespan_warmup_continues_when_coreml_unavailable_and_not_strict(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _PrecompileFailingRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> _PrecompileFailingRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    with TestClient(create_app(_warmup_settings(settings, strict=False))) as test_client:
        response = test_client.get("/v1/health")
        assert response.status_code == 200

    assert len(runtime.precompile_calls) == 1
    captured = capsys.readouterr()
    assert "CoreML stateful warmup unavailable" in captured.out


def test_lifespan_warmup_strict_reraises_when_coreml_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    settings: ServerSettings,
) -> None:
    runtime = _PrecompileFailingRuntime()

    def get_runtime(self: openai_api_server.RuntimeState) -> _PrecompileFailingRuntime:
        with self._lock:
            self._runtime = runtime
        return runtime

    monkeypatch.setattr(openai_api_server.RuntimeState, "get_runtime", get_runtime)

    with pytest.raises(openai_api_server.CoreMLStatefulUnavailableError):
        with TestClient(create_app(_warmup_settings(settings, strict=True))):
            pass

    assert len(runtime.precompile_calls) == 1
