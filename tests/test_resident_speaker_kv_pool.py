from __future__ import annotations

import sys
import threading
import time
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
)
from irodori_tts.model import JointAttention  # noqa: E402


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
            num_layers=2,
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
            for _ in range(int(self.cfg.num_layers))
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
            for _ in range(int(self.cfg.num_layers))
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
                torch.full(
                    (batch, speaker_len, 1, 1),
                    float(layer_idx + 1),
                    dtype=speaker_state.dtype,
                ),
                torch.full(
                    (batch, speaker_len, 1, 1),
                    float(layer_idx + 1) * 0.5,
                    dtype=speaker_state.dtype,
                ),
            )
            for layer_idx in range(int(self.cfg.num_layers))
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


def _condition_cache(
    reference_cache_id: str,
    speaker_context_len: int,
    *,
    sequence_length: int = 4,
    speaker_context_len_bucket: int = 4,
) -> ConditionCacheHandle:
    now = datetime.now(timezone.utc)
    return ConditionCacheHandle(
        id="cond_test",
        reference_cache_id=reference_cache_id,
        model_fingerprint="model:test",
        tokenizer_fingerprint="tokenizer:test",
        condition_fingerprint="condition:test",
        bucket_id=f"S{sequence_length}_T2_R{speaker_context_len_bucket}_independent_text_speaker3",
        state_layout=STATE_LAYOUT_PER_LAYER,
        sequence_length=int(sequence_length),
        text_len=2,
        speaker_context_len=int(speaker_context_len),
        speaker_context_len_bucket=int(speaker_context_len_bucket),
        c_ctx_bucket=2 + int(speaker_context_len_bucket),
        branch_layouts=(BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3),
        mlstate_keys=expected_per_layer_state_names(),
        created_at=now,
        expires_at=None,
        memory_bytes=0,
        metadata={},
    )


def _patch_reference_loader(
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


def _make_request(*, ref_path: Path, num_steps: int = 1) -> inference_runtime.SamplingRequest:
    return inference_runtime.SamplingRequest(
        text="hello",
        ref_wav=str(ref_path),
        seconds=2.0,
        num_steps=num_steps,
        seed=123,
        trim_tail=False,
        cfg_min_t=0.0,
        cfg_max_t=1.0,
        cfg_scale_text=0.0,
        cfg_scale_speaker=0.0,
    )


def test_project_speaker_context_kv_does_not_touch_text_path() -> None:
    attn = JointAttention(
        dim=4,
        heads=2,
        text_ctx_dim=4,
        speaker_ctx_dim=4,
        caption_ctx_dim=None,
        norm_eps=1e-5,
    )
    counts: dict[str, int] = {
        "wk_text": 0,
        "wv_text": 0,
        "wk_speaker": 0,
        "wv_speaker": 0,
    }

    class _CountingLinear(torch.nn.Module):
        def __init__(self, base: torch.nn.Linear, name: str) -> None:
            super().__init__()
            self.base = base
            self.name = name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            counts[self.name] = counts.get(self.name, 0) + 1
            return self.base(x)

    attn.wk_text = _CountingLinear(attn.wk_text, "wk_text")
    attn.wv_text = _CountingLinear(attn.wv_text, "wv_text")
    attn.wk_speaker = _CountingLinear(attn.wk_speaker, "wk_speaker")
    attn.wv_speaker = _CountingLinear(attn.wv_speaker, "wv_speaker")

    speaker_input = torch.randn(1, 3, 4)
    k_speaker, v_speaker = attn.project_speaker_context_kv(speaker_input)
    assert k_speaker.shape == (1, 3, 2, 2)
    assert v_speaker.shape == (1, 3, 2, 2)
    assert counts["wk_speaker"] == 1
    assert counts["wv_speaker"] == 1
    assert counts["wk_text"] == 0
    assert counts["wv_text"] == 0

    text_input = torch.randn(1, 5, 4)
    k_text, v_text = attn.project_text_context_kv(text_input)
    assert k_text.shape == (1, 5, 2, 2)
    assert v_text.shape == (1, 5, 2, 2)
    assert counts["wk_text"] == 1
    assert counts["wv_text"] == 1
    assert counts["wk_speaker"] == 1
    assert counts["wv_speaker"] == 1


def test_resident_speaker_kv_key_separates_sequence_bucket_and_scale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )

    speaker_state = resident.speaker_state
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=100,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=160,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=100,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=(2.5, None),
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 3
    bucket_signatures = {
        (int(entry["sequence_length_bucket"]), entry["scale_signature"])
        for entry in snapshot["entries"]
    }
    assert bucket_signatures == {
        (100, None),
        (100, (2.5, None)),
        (160, None),
    }
    assert snapshot["resident_speaker_kv_prepares"] == 3
    assert snapshot["speaker_kv_projection_calls"] == 2

    hits_before = snapshot["resident_speaker_kv_hits"]
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=100,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["resident_speaker_kv_hits"] == hits_before + 1
    assert snapshot["speaker_kv_projection_calls"] == 2


def test_repeated_synthesize_reuses_resident_speaker_kv_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    monkeypatch.setattr(
        inference_runtime,
        "COREML_STATEFUL_BACKEND_FACTORY",
        lambda runtime, condition_cache, branch_layout: FakeBackend(),
    )
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )

    runtime.model.calls["build_text_context_kv_cache"] = 0
    runtime.model.calls["build_speaker_context_kv_cache"] = 0
    runtime.model.calls["build_context_kv_cache"] = 0

    cond_cache = _condition_cache("ref_default", resident.speaker_context_len)
    req = _make_request(ref_path=tmp_path / "rem.wav")

    runtime.synthesize_with_condition_cache(
        req,
        condition_cache=cond_cache,
        resident_reference_tensors=resident,
    )
    runtime.synthesize_with_condition_cache(
        req,
        condition_cache=cond_cache,
        resident_reference_tensors=resident,
    )

    assert runtime.model.calls["build_speaker_context_kv_cache"] == 1
    assert runtime.model.calls["build_text_context_kv_cache"] >= 2
    assert runtime.model.calls["build_context_kv_cache"] == 0
    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["speaker_kv_projection_calls"] == 1
    assert snapshot["resident_speaker_kv_prepares"] == 1
    assert snapshot["resident_speaker_kv_hits"] >= 1


def test_scaled_variant_does_not_mutate_unscaled_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )
    speaker_state = resident.speaker_state

    unscaled_cache = runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    unscaled_snapshot = [(k.detach().clone(), v.detach().clone()) for k, v in unscaled_cache]

    scaled_cache = runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=(2.5, None),
    )

    assert len(scaled_cache) == len(unscaled_cache)
    for layer_idx, ((k_unscaled, v_unscaled), (k_scaled, v_scaled)) in enumerate(
        zip(unscaled_cache, scaled_cache, strict=True)
    ):
        snap_k, snap_v = unscaled_snapshot[layer_idx]
        assert torch.equal(k_unscaled, snap_k)
        assert torch.equal(v_unscaled, snap_v)
        assert torch.allclose(k_scaled, snap_k * 2.5)
        assert torch.allclose(v_scaled, snap_v * 2.5)
        assert k_scaled.data_ptr() != k_unscaled.data_ptr()
        assert v_scaled.data_ptr() != v_unscaled.data_ptr()


def test_resident_speaker_kv_lru_evicts_oldest_bucket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    runtime._resident_speaker_kv_max_entries = 2
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )
    speaker_state = resident.speaker_state

    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=10,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    time.sleep(0.005)
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=20,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    time.sleep(0.005)
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=30,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 2
    assert snapshot["resident_speaker_kv_evictions"] == 1
    sequence_buckets = sorted(int(entry["sequence_length_bucket"]) for entry in snapshot["entries"])
    assert sequence_buckets == [20, 30]


def test_delete_resident_reference_cascades_speaker_kv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    resident_a = runtime.prepare_default_reference_tensors(
        tmp_path / "a.wav",
        None,
        reference_cache_id="ref_a",
        reference_fingerprint="fp:a",
    )
    resident_b = runtime.prepare_default_reference_tensors(
        tmp_path / "b.wav",
        None,
        reference_cache_id="ref_b",
        reference_fingerprint="fp:b",
    )

    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=resident_a.speaker_state,
        reference_cache_id="ref_a",
        reference_fingerprint="fp:a",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=resident_a.speaker_state,
        reference_cache_id="ref_a",
        reference_fingerprint="fp:a",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=(2.0, None),
    )
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=resident_b.speaker_state,
        reference_cache_id="ref_b",
        reference_fingerprint="fp:b",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 3

    deleted = runtime.delete_resident_reference_tensors("ref_a")
    assert deleted is True

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1
    assert snapshot["entries"][0]["reference_cache_id"] == "ref_b"
    assert snapshot["resident_speaker_kv_deletes"] == 2


def test_force_refresh_evicts_stale_resident_speaker_kv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=resident.speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1
    assert snapshot["speaker_kv_projection_calls"] == 1
    deletes_before = int(snapshot["resident_speaker_kv_deletes"])

    refreshed = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        force_refresh=True,
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 0
    assert snapshot["resident_speaker_kv_deletes"] == deletes_before + 1

    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=refreshed.speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1
    assert snapshot["speaker_kv_projection_calls"] == 2


def test_unload_clears_resident_speaker_kv_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )
    runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=resident.speaker_state,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
        sequence_length_bucket=4,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1

    runtime.unload()

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 0
    assert snapshot["entries"] == []


def test_disabled_resident_speaker_kv_uses_legacy_build_context_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime()
    _patch_reference_loader(monkeypatch, runtime, {"load_reference_latent": 0})
    monkeypatch.setattr(
        inference_runtime,
        "COREML_STATEFUL_BACKEND_FACTORY",
        lambda runtime, condition_cache, branch_layout: FakeBackend(),
    )
    resident = runtime.prepare_default_reference_tensors(
        tmp_path / "rem.wav",
        None,
        reference_cache_id="ref_default",
        reference_fingerprint="fp:1",
    )

    runtime.configure_resident_speaker_kv(enabled=False)
    assert runtime.resident_speaker_kv_enabled is False

    runtime.model.calls["build_text_context_kv_cache"] = 0
    runtime.model.calls["build_speaker_context_kv_cache"] = 0
    runtime.model.calls["build_context_kv_cache"] = 0

    cond_cache = _condition_cache("ref_default", resident.speaker_context_len)
    req = _make_request(ref_path=tmp_path / "rem.wav")

    runtime.synthesize_with_condition_cache(
        req,
        condition_cache=cond_cache,
        resident_reference_tensors=resident,
    )
    runtime.synthesize_with_condition_cache(
        req,
        condition_cache=cond_cache,
        resident_reference_tensors=resident,
    )

    assert runtime.model.calls["build_speaker_context_kv_cache"] == 0
    assert runtime.model.calls["build_text_context_kv_cache"] == 0
    assert runtime.model.calls["build_context_kv_cache"] >= 2
    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 0
    assert snapshot["resident_speaker_kv_prepares"] == 0


def test_configure_resident_speaker_kv_max_entries_evicts_excess() -> None:
    runtime = make_runtime()
    runtime._resident_speaker_kv_store[("speaker_kv", "tiny", "ref1", None, 4, 4, "cond1", None)] = (
        inference_runtime.ResidentSpeakerKVPayload(
            key=("speaker_kv", "tiny", "ref1", None, 4, 4, "cond1", None),
            reference_cache_id="ref1",
            reference_fingerprint=None,
            sequence_length_bucket=4,
            speaker_context_len_bucket=4,
            branch_layout="cond1",
            scale_signature=None,
            speaker_kv_cache=(),
            memory_bytes=0,
            created_at=datetime.now(timezone.utc),
            last_used_at=datetime.now(timezone.utc),
        )
    )
    runtime._resident_speaker_kv_store[("speaker_kv", "tiny", "ref2", None, 4, 4, "cond1", None)] = (
        inference_runtime.ResidentSpeakerKVPayload(
            key=("speaker_kv", "tiny", "ref2", None, 4, 4, "cond1", None),
            reference_cache_id="ref2",
            reference_fingerprint=None,
            sequence_length_bucket=4,
            speaker_context_len_bucket=4,
            branch_layout="cond1",
            scale_signature=None,
            speaker_kv_cache=(),
            memory_bytes=0,
            created_at=datetime.now(timezone.utc),
            last_used_at=datetime.now(timezone.utc),
        )
    )
    assert runtime.resident_speaker_kv_metrics_snapshot()["count"] == 2

    runtime.configure_resident_speaker_kv(max_entries=1)
    assert runtime.resident_speaker_kv_max_entries == 1

    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1
    assert snapshot["resident_speaker_kv_evictions"] == 1
    assert snapshot["memory_bytes"] == 0


def test_warmup_resident_speaker_kv_handles_inference_speaker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = make_runtime()

    projection = torch.nn.Linear(3, 3, bias=False)

    def _build(*, speaker_state: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # The bug: with autograd active, projecting an inference tensor through a
        # parameter raises "Inference tensors cannot be saved for backward".
        k = projection(speaker_state)
        v = projection(speaker_state)
        return [(k, v) for _ in range(int(runtime.model.cfg.num_layers))]

    monkeypatch.setattr(runtime.model, "build_speaker_context_kv_cache", _build)

    with torch.inference_mode():
        speaker_state = torch.ones((1, 4, 3), dtype=torch.float32)
        speaker_mask = torch.ones((1, 4), dtype=torch.bool)
        ref_latent = torch.ones((1, 2, 2), dtype=torch.float32)
        ref_mask = torch.ones((1, 2), dtype=torch.bool)
    assert speaker_state.is_inference()

    # Sanity check: outside inference_mode, projecting through a parameter must
    # raise on an inference tensor — proves the failure mode the fix protects against.
    with pytest.raises(RuntimeError, match="Inference tensors"):
        projection(speaker_state)

    now = datetime.now(timezone.utc)
    resident = inference_runtime.ResidentReferenceTensors(
        reference_cache_id="ref_inference",
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        speaker_state=speaker_state,
        speaker_mask=speaker_mask,
        reference_fingerprint="fp:inference",
        source="memory://inference",
        ref_len=int(ref_latent.shape[1]),
        speaker_context_len=int(speaker_mask.shape[1]),
        speaker_dim=int(speaker_state.shape[-1]),
        memory_bytes=0,
        device="cpu",
        dtype="float32",
        created_at=now,
        last_used_at=now,
    )

    assert not torch.is_inference_mode_enabled()
    payload = runtime.warmup_resident_speaker_kv(
        resident_reference=resident,
        sequence_length_bucket=100,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
    )

    assert isinstance(payload, inference_runtime.ResidentSpeakerKVPayload)
    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1
    assert snapshot["resident_speaker_kv_prepares"] == 1
    assert snapshot["speaker_kv_projection_calls"] == 1


def test_get_or_prepare_resident_speaker_kv_handles_inference_speaker_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = make_runtime()

    projection = torch.nn.Linear(3, 3, bias=False)

    def _build(*, speaker_state: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        k = projection(speaker_state)
        v = projection(speaker_state)
        return [(k, v) for _ in range(int(runtime.model.cfg.num_layers))]

    monkeypatch.setattr(runtime.model, "build_speaker_context_kv_cache", _build)

    with torch.inference_mode():
        speaker_state = torch.ones((1, 4, 3), dtype=torch.float32)
    assert speaker_state.is_inference()

    assert not torch.is_inference_mode_enabled()
    cache = runtime._get_or_prepare_resident_speaker_kv(
        speaker_state=speaker_state,
        reference_cache_id="ref_inference",
        reference_fingerprint="fp:inference",
        sequence_length_bucket=100,
        speaker_context_len_bucket=4,
        branch_layout=BRANCH_LAYOUT_COND1,
        scale_signature=None,
    )

    assert len(cache) == int(runtime.model.cfg.num_layers)
    snapshot = runtime.resident_speaker_kv_metrics_snapshot()
    assert snapshot["count"] == 1
    assert snapshot["speaker_kv_projection_calls"] == 1
