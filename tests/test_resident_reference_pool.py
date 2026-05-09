from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from irodori_tts import inference_runtime  # noqa: E402
from irodori_tts.coreml_cache import (  # noqa: E402
    BRANCH_LAYOUT_COND1,
    BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    STATE_LAYOUT_PER_LAYER,
    ConditionCacheHandle,
    expected_per_layer_state_names,
)  # noqa: E402


class TinyTokenizer:
    def batch_encode(
        self,
        texts: list[str],
        *,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del texts
        ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        mask = torch.ones((1, max_length), dtype=torch.bool)
        return ids, mask


class TinyCodec:
    sample_rate = 2
    model = SimpleNamespace(hop_length=1)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros((z.shape[0], 1, z.shape[1]), dtype=z.dtype)


class TinyResidentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls: dict[str, int] = {
            "encode_conditions": 0,
            "speaker_encoder": 0,
            "text_encoder": 0,
            "build_context_kv_cache": 0,
            "build_text_context_kv_cache": 0,
            "build_speaker_context_kv_cache": 0,
        }
        self.fail_encode_conditions = False
        self.cfg = SimpleNamespace(
            use_caption_condition=False,
            use_speaker_condition=True,
            patched_latent_dim=2,
            latent_patch_size=1,
            latent_dim=2,
            speaker_patch_size=1,
            num_layers=1,
        )

    def text_encoder(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        del text_mask
        self.calls["text_encoder"] += 1
        return torch.ones((text_ids.shape[0], text_ids.shape[1], 3), dtype=self.weight.dtype)

    def text_norm(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def speaker_encoder(self, ref_latent: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        del ref_mask
        self.calls["speaker_encoder"] += 1
        return torch.ones((ref_latent.shape[0], ref_latent.shape[1], 3), dtype=self.weight.dtype)

    def speaker_norm(self, state: torch.Tensor) -> torch.Tensor:
        return state

    @staticmethod
    def _prepend_masked_mean_token(
        state: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.unsqueeze(-1).to(dtype=state.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_token = (state * mask_f).sum(dim=1, keepdim=True) / denom
        has_any = mask.any(dim=1, keepdim=True)
        return torch.cat([mean_token, state], dim=1), torch.cat([has_any, mask], dim=1)

    def encode_conditions(
        self,
        *,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        caption_input_ids: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        del caption_input_ids, caption_mask
        self.calls["encode_conditions"] += 1
        if self.fail_encode_conditions:
            raise AssertionError("resident path must not call encode_conditions")
        text_state = self.text_encoder(text_input_ids, text_mask)
        if ref_latent is None or ref_mask is None:
            raise AssertionError("fallback path must provide reference tensors")
        speaker_state = self.speaker_encoder(ref_latent, ref_mask)
        speaker_state, speaker_mask = self._prepend_masked_mean_token(speaker_state, ref_mask)
        return text_state, text_mask, speaker_state, speaker_mask, None, None

    def build_context_kv_cache(
        self,
        *,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
        caption_state: torch.Tensor | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        del caption_state
        self.calls["build_context_kv_cache"] += 1
        batch = int(text_state.shape[0])
        text_len = int(text_state.shape[1])
        speaker_len = int(speaker_state.shape[1])
        return [
            (
                torch.ones((batch, text_len, 1, 1), dtype=text_state.dtype),
                torch.ones((batch, text_len, 1, 1), dtype=text_state.dtype),
                torch.ones((batch, speaker_len, 1, 1), dtype=text_state.dtype),
                torch.ones((batch, speaker_len, 1, 1), dtype=text_state.dtype),
            )
        ]

    def build_text_context_kv_cache(
        self,
        *,
        text_state: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.calls["build_text_context_kv_cache"] += 1
        batch = int(text_state.shape[0])
        text_len = int(text_state.shape[1])
        return [
            (
                torch.ones((batch, text_len, 1, 1), dtype=text_state.dtype),
                torch.ones((batch, text_len, 1, 1), dtype=text_state.dtype),
            )
        ]

    def build_speaker_context_kv_cache(
        self,
        *,
        speaker_state: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.calls["build_speaker_context_kv_cache"] += 1
        batch = int(speaker_state.shape[0])
        speaker_len = int(speaker_state.shape[1])
        return [
            (
                torch.ones((batch, speaker_len, 1, 1), dtype=speaker_state.dtype),
                torch.ones((batch, speaker_len, 1, 1), dtype=speaker_state.dtype),
            )
        ]


class FakeBackend:
    def prepare_state(self, payload: Any) -> SimpleNamespace:
        return SimpleNamespace(batch_size=payload.batch_size)

    def predict_step(
        self,
        prepared_state: Any,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        del prepared_state, t, latent_mask
        return torch.zeros_like(x_t)


def make_runtime() -> inference_runtime.InferenceRuntime:
    model = TinyResidentModel()
    runtime = inference_runtime.InferenceRuntime.__new__(inference_runtime.InferenceRuntime)
    runtime.key = inference_runtime.RuntimeKey(checkpoint="tiny", model_device="cpu")
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = model.cfg
    runtime.train_cfg = None
    runtime.model = model
    runtime.tokenizer = TinyTokenizer()
    runtime.caption_tokenizer = None
    runtime.codec = TinyCodec()
    runtime.default_text_max_len = 2
    runtime.default_caption_max_len = 2
    runtime._infer_lock = threading.Lock()
    runtime._coreml_stateful_backends = {}
    runtime._coreml_stateful_backends_lock = threading.Lock()
    runtime._runtime_resident_lock = threading.RLock()
    runtime._resident_prepare_locks = tuple(threading.RLock() for _ in range(8))
    runtime._resident_reference_tensors = {}
    runtime._resident_reference_metrics = {
        "reference_encode_calls": 0,
        "speaker_state_encode_calls": 0,
        "resident_reference_hits": 0,
        "resident_reference_misses": 0,
        "resident_reference_prepares": 0,
        "resident_reference_deletes": 0,
    }
    runtime._resident_speaker_kv_store = {}
    runtime._resident_speaker_kv_metrics = {
        "speaker_kv_projection_calls": 0,
        "resident_speaker_kv_hits": 0,
        "resident_speaker_kv_misses": 0,
        "resident_speaker_kv_prepares": 0,
        "resident_speaker_kv_evictions": 0,
        "resident_speaker_kv_deletes": 0,
    }
    runtime._resident_speaker_kv_max_entries = (
        inference_runtime._RESIDENT_SPEAKER_KV_DEFAULT_MAX_ENTRIES
    )
    return runtime


def condition_cache(reference_cache_id: str, speaker_context_len: int) -> ConditionCacheHandle:
    now = datetime.now(timezone.utc)
    return ConditionCacheHandle(
        id="cond_test",
        reference_cache_id=reference_cache_id,
        model_fingerprint="model:test",
        tokenizer_fingerprint="tokenizer:test",
        condition_fingerprint="condition:test",
        bucket_id="S4_T2_R4_independent_text_speaker3",
        state_layout=STATE_LAYOUT_PER_LAYER,
        sequence_length=4,
        text_len=2,
        speaker_context_len=int(speaker_context_len),
        speaker_context_len_bucket=4,
        c_ctx_bucket=6,
        branch_layouts=(BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3),
        mlstate_keys=expected_per_layer_state_names(),
        created_at=now,
        expires_at=None,
        memory_bytes=0,
        metadata={},
    )


def patch_reference_loader(
    monkeypatch: pytest.MonkeyPatch,
    runtime: inference_runtime.InferenceRuntime,
    calls: dict[str, int],
) -> None:
    def load_reference_latent(
        *,
        req: inference_runtime.SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del req, messages
        calls["load_reference_latent"] += 1
        return (
            torch.ones((batch_size, 2, 2), dtype=torch.float32),
            torch.ones((batch_size, 2), dtype=torch.bool),
        )

    monkeypatch.setattr(runtime, "_load_reference_latent", load_reference_latent)


def test_prepare_default_reference_tensors_stores_actual_tensors_and_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    calls = {"load_reference_latent": 0}
    patch_reference_loader(monkeypatch, runtime, calls)

    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
    )

    assert calls["load_reference_latent"] == 1
    assert runtime.model.calls["speaker_encoder"] == 1
    assert resident.reference_cache_id == "ref_default"
    assert resident.ref_latent.shape == (1, 2, 2)
    assert resident.ref_mask.shape == (1, 2)
    assert resident.speaker_state.shape == (1, 3, 3)
    assert resident.speaker_mask.shape == (1, 3)
    assert resident.ref_len == 2
    assert resident.speaker_context_len == 3
    assert resident.speaker_dim == 3
    expected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            resident.ref_latent,
            resident.ref_mask,
            resident.speaker_state,
            resident.speaker_mask,
        )
    )
    assert resident.memory_bytes == expected_bytes

    assert runtime.get_resident_reference_tensors("ref_default") is not None
    assert runtime.delete_resident_reference_tensors("ref_default") is True
    assert runtime.get_resident_reference_tensors("ref_default") is None


def test_unload_clears_resident_reference_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})

    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
    )

    assert runtime.resident_reference_metrics_snapshot()["count"] == 1
    stored = runtime.get_resident_reference_tensors("ref_default")
    assert stored is not None
    assert stored.ref_latent is resident.ref_latent

    backend_key = (1, 2, "cond1", "branch")
    runtime._coreml_stateful_backends[backend_key] = FakeBackend()
    assert backend_key in runtime._coreml_stateful_backends

    runtime.unload()

    snapshot = runtime.resident_reference_metrics_snapshot()
    assert snapshot["count"] == 0
    assert snapshot["memory_bytes"] == 0
    assert snapshot["entries"] == []
    assert runtime.get_resident_reference_tensors("ref_default") is None
    assert runtime._coreml_stateful_backends == {}


def test_prepare_default_reference_tensors_force_refresh_reencodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    calls = {"load_reference_latent": 0}

    def load_reference_latent(
        *,
        req: inference_runtime.SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del req, messages
        calls["load_reference_latent"] += 1
        value = float(calls["load_reference_latent"])
        return (
            torch.full((batch_size, 2, 2), value, dtype=torch.float32),
            torch.ones((batch_size, 2), dtype=torch.bool),
        )

    monkeypatch.setattr(runtime, "_load_reference_latent", load_reference_latent)

    first = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
    )
    reused = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
    )
    refreshed = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
        force_refresh=True,
    )

    assert calls["load_reference_latent"] == 2
    assert runtime.model.calls["speaker_encoder"] == 2
    assert reused.ref_latent is first.ref_latent
    assert refreshed.ref_latent is not first.ref_latent
    assert torch.equal(first.ref_latent, torch.ones_like(first.ref_latent))
    assert torch.equal(refreshed.ref_latent, torch.full_like(refreshed.ref_latent, 2.0))
    stored = runtime.get_resident_reference_tensors("ref_default")
    assert stored is not None
    assert stored.ref_latent is refreshed.ref_latent


def test_prepare_default_reference_tensors_reencodes_same_id_on_source_or_fingerprint_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    calls = {"load_reference_latent": 0}

    def load_reference_latent(
        *,
        req: inference_runtime.SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del req, messages
        calls["load_reference_latent"] += 1
        value = float(calls["load_reference_latent"])
        return (
            torch.full((batch_size, 2, 2), value, dtype=torch.float32),
            torch.ones((batch_size, 2), dtype=torch.bool),
        )

    monkeypatch.setattr(runtime, "_load_reference_latent", load_reference_latent)
    first_path = tmp_path / "a" / "rem.wav"
    second_path = tmp_path / "b" / "rem.wav"
    first_path.parent.mkdir()
    second_path.parent.mkdir()

    first = runtime.prepare_default_reference_tensors(
        first_path,
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fingerprint:a",
    )
    same = runtime.prepare_default_reference_tensors(
        first_path,
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fingerprint:a",
    )
    changed_fingerprint = runtime.prepare_default_reference_tensors(
        first_path,
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fingerprint:b",
    )
    changed_source = runtime.prepare_default_reference_tensors(
        second_path,
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fingerprint:b",
    )

    assert calls["load_reference_latent"] == 3
    assert same.ref_latent is first.ref_latent
    assert torch.equal(changed_fingerprint.ref_latent, torch.full_like(first.ref_latent, 2.0))
    assert torch.equal(changed_source.ref_latent, torch.full_like(first.ref_latent, 3.0))
    assert changed_source.source.endswith("/b/rem.wav")
    stored = runtime.get_resident_reference_tensors("ref_default")
    assert stored is not None
    assert stored.ref_latent is changed_source.ref_latent


def test_resident_reference_metrics_snapshot_has_no_tensor_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
    )

    snapshot = runtime.resident_reference_metrics_snapshot()

    assert snapshot["count"] == 1
    assert snapshot["memory_bytes"] > 0
    entry = snapshot["entries"][0]
    assert "ref_latent" not in entry
    assert "speaker_state" not in entry
    assert entry["reference_cache_id"] == "ref_default"


def test_synthesize_with_resident_reference_skips_reference_and_speaker_encode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    calls = {"load_reference_latent": 0}
    patch_reference_loader(monkeypatch, runtime, calls)
    monkeypatch.setattr(
        inference_runtime,
        "COREML_STATEFUL_BACKEND_FACTORY",
        lambda runtime, condition_cache, branch_layout: FakeBackend(),
    )
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="server_default:rem.wav",
    )
    runtime.model.fail_encode_conditions = True
    calls["load_reference_latent"] = 0
    runtime.model.calls["speaker_encoder"] = 0

    result = runtime.synthesize_with_condition_cache(
        inference_runtime.SamplingRequest(
            text="hello",
            ref_wav=str(tmp_path / "rem.wav"),
            seconds=2.0,
            num_steps=1,
            seed=123,
            trim_tail=False,
            cfg_min_t=0.0,
            cfg_max_t=1.0,
        ),
        condition_cache=condition_cache("ref_default", resident.speaker_context_len),
        resident_reference_tensors=resident,
    )

    assert result.sample_rate == 2
    assert calls["load_reference_latent"] == 0
    assert runtime.model.calls["encode_conditions"] == 0
    assert runtime.model.calls["speaker_encoder"] == 0
    assert runtime.model.calls["text_encoder"] >= 1
    # Active CFG (defaults) builds a speaker-uncond branch with zero speaker_state,
    # which keeps using build_context_kv_cache; the cond and text-uncond branches
    # exercise the resident speaker KV path instead.
    assert runtime.model.calls["build_context_kv_cache"] == 1
    assert runtime.model.calls["build_text_context_kv_cache"] >= 1
    assert runtime.model.calls["build_speaker_context_kv_cache"] == 1


def test_synthesize_without_resident_reference_keeps_existing_encode_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    calls = {"load_reference_latent": 0}
    patch_reference_loader(monkeypatch, runtime, calls)
    monkeypatch.setattr(
        inference_runtime,
        "COREML_STATEFUL_BACKEND_FACTORY",
        lambda runtime, condition_cache, branch_layout: FakeBackend(),
    )

    result = runtime.synthesize_with_condition_cache(
        inference_runtime.SamplingRequest(
            text="hello",
            ref_wav=str(tmp_path / "rem.wav"),
            seconds=2.0,
            num_steps=1,
            seed=123,
            trim_tail=False,
            cfg_min_t=0.0,
            cfg_max_t=1.0,
        ),
        condition_cache=condition_cache("ref_default", 3),
    )

    assert result.sample_rate == 2
    assert calls["load_reference_latent"] == 1
    assert runtime.model.calls["encode_conditions"] == 1
    assert runtime.model.calls["speaker_encoder"] == 1
