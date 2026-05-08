from __future__ import annotations

import importlib
import sys
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_coreml_stateful_module():
    sys.modules.pop("irodori_tts.coreml_stateful", None)
    return importlib.import_module("irodori_tts.coreml_stateful")


def test_module_import_does_not_require_coremltools() -> None:
    sys.modules.pop("coremltools", None)

    module = load_coreml_stateful_module()

    assert module.STATE_LAYOUT_PER_LAYER == "per_layer_text_speaker_context_v1"
    assert "coremltools" not in sys.modules


class RopeCacheLeaf(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_freqs_cis_cache",
            torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        )


class RopeCacheTree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._freqs_cis_cache = torch.tensor([3.0 + 4.0j], dtype=torch.complex64)
        self.leaf = RopeCacheLeaf()
        self.stack = torch.nn.Sequential(RopeCacheLeaf())


def test_reset_freqs_cis_caches_clears_nested_complex_rope_caches() -> None:
    coreml_stateful = load_coreml_stateful_module()
    module = RopeCacheTree()

    coreml_stateful._reset_freqs_cis_caches(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        module.to(dtype=torch.float32)
        module.to(dtype=torch.float16)
    coreml_stateful._reset_freqs_cis_caches(module)

    for child in module.modules():
        if not hasattr(child, "_freqs_cis_cache"):
            continue
        cache = child._freqs_cis_cache
        assert isinstance(cache, torch.Tensor)
        assert cache.dtype == torch.complex64
        assert cache.numel() == 0
        assert cache.requires_grad is False


def test_convert_and_reload_mlmodel_returns_coreml_framework_loaded_model(
    tmp_path: Path,
) -> None:
    coreml_stateful = load_coreml_stateful_module()
    calls: list[tuple[str, Any]] = []

    class FakeModels:
        def MLModel(self, path: str, *, compute_units: str) -> str:
            calls.append(("MLModel", path, compute_units))
            return f"loaded:{path}:{compute_units}"

    class FakeCoreMLTools:
        models = FakeModels()

        @staticmethod
        def convert(traced: object, **kwargs: object) -> str:
            calls.append(("convert", traced, kwargs))
            return "converted-without-framework-state"

    package_path = tmp_path / "stateful_step.mlpackage"

    loaded = coreml_stateful._convert_and_reload_mlmodel(
        FakeCoreMLTools,
        "traced-model",
        package_path=package_path,
        compute_unit="CPU_AND_NE",
        convert_kwargs={"inputs": ["x_t"], "states": ["state"]},
    )

    assert loaded == f"loaded:{package_path}:CPU_AND_NE"
    assert calls == [
        (
            "convert",
            "traced-model",
            {
                "inputs": ["x_t"],
                "states": ["state"],
                "package_dir": str(package_path),
            },
        ),
        ("MLModel", str(package_path), "CPU_AND_NE"),
    ]


def test_pack_context_kv_state_cond1_shapes_and_valid_mask() -> None:
    coreml_stateful = load_coreml_stateful_module()
    context_kv_cache = [
        (
            torch.full((1, 2, 1, 1), 1.0),
            torch.full((1, 2, 1, 1), 2.0),
            torch.full((1, 3, 1, 1), 3.0),
            torch.full((1, 3, 1, 1), 4.0),
        )
    ]

    payload = coreml_stateful.pack_context_kv_state(
        context_kv_cache,
        text_mask=torch.tensor([[True, False]]),
        speaker_mask=torch.tensor([[True, True, False]]),
        speaker_context_bucket=4,
        branch_layout=coreml_stateful.BRANCH_LAYOUT_COND1,
    )

    assert tuple(payload.state_payloads) == (
        "context_k_l00",
        "context_v_l00",
        "valid_mask_state",
    )
    assert payload.state_payloads["context_k_l00"].shape == (1, 6, 1, 1)
    assert payload.state_payloads["context_v_l00"].shape == (1, 6, 1, 1)
    assert payload.state_payloads["valid_mask_state"].shape == (1, 6)
    assert payload.state_payloads["context_k_l00"][0, :, 0, 0].tolist() == [
        1.0,
        1.0,
        3.0,
        3.0,
        3.0,
        0.0,
    ]
    assert payload.state_payloads["valid_mask_state"].tolist() == [[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]


def test_pack_context_kv_state_independent3_shapes_and_branch_masks() -> None:
    coreml_stateful = load_coreml_stateful_module()
    context_kv_cache = [
        (
            torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1),
            torch.zeros((3, 2, 1, 1)),
            torch.arange(30, 33, dtype=torch.float32).reshape(3, 1, 1, 1),
            torch.zeros((3, 1, 1, 1)),
        )
    ]

    payload = coreml_stateful.pack_context_kv_state(
        context_kv_cache,
        text_mask=torch.tensor(
            [
                [True, True],
                [False, False],
                [True, True],
            ]
        ),
        speaker_mask=torch.tensor(
            [
                [True],
                [True],
                [False],
            ]
        ),
        speaker_context_bucket=2,
        branch_layout=coreml_stateful.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    )

    assert payload.state_payloads["context_k_l00"].shape == (3, 4, 1, 1)
    assert payload.state_payloads["context_k_l00"][:, :, 0, 0].tolist() == [
        [0.0, 1.0, 30.0, 0.0],
        [2.0, 3.0, 31.0, 0.0],
        [4.0, 5.0, 32.0, 0.0],
    ]
    assert payload.state_payloads["valid_mask_state"].tolist() == [
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_write_state_payloads_are_float32_and_contiguous() -> None:
    coreml_stateful = load_coreml_stateful_module()
    payload = coreml_stateful.pack_context_kv_state(
        [
            (
                torch.ones((1, 1, 1, 1)),
                torch.ones((1, 1, 1, 1)),
                torch.ones((1, 1, 1, 1)),
                torch.ones((1, 1, 1, 1)),
            )
        ],
        text_mask=torch.tensor([[True]]),
        speaker_mask=torch.tensor([[True]]),
        speaker_context_bucket=1,
        branch_layout=coreml_stateful.BRANCH_LAYOUT_COND1,
    )

    write_payloads = coreml_stateful.state_write_payloads(payload)

    assert tuple(write_payloads) == tuple(payload.state_payloads)
    assert all(value.dtype == np.float32 for value in write_payloads.values())
    assert all(value.flags.c_contiguous for value in write_payloads.values())


def test_pack_context_kv_state_rejects_unsupported_branch_batch() -> None:
    coreml_stateful = load_coreml_stateful_module()

    with pytest.raises(ValueError, match="branch batch"):
        coreml_stateful.pack_context_kv_state(
            [
                (
                    torch.ones((2, 1, 1, 1)),
                    torch.ones((2, 1, 1, 1)),
                    torch.ones((2, 1, 1, 1)),
                    torch.ones((2, 1, 1, 1)),
                )
            ],
            text_mask=torch.ones((2, 1), dtype=torch.bool),
            speaker_mask=torch.ones((2, 1), dtype=torch.bool),
            speaker_context_bucket=1,
            branch_layout=coreml_stateful.BRANCH_LAYOUT_COND1,
        )


class TinyTokenizer:
    def batch_encode(
        self, texts: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((len(texts), max_length), dtype=torch.long),
            torch.ones((len(texts), max_length), dtype=torch.bool),
        )


class TinyCodec:
    sample_rate = 2
    model = SimpleNamespace(hop_length=1)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros((z.shape[0], 1, z.shape[1]), dtype=z.dtype)


class TinyModel(torch.nn.Module):
    def __init__(self, *, num_layers: int = 1) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(
            use_caption_condition=False,
            use_speaker_condition=True,
            patched_latent_dim=2,
            latent_patch_size=1,
            latent_dim=2,
            speaker_patch_size=1,
            num_layers=num_layers,
        )

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
        del ref_latent, caption_input_ids, caption_mask
        batch = int(text_input_ids.shape[0])
        text_state = torch.ones((batch, text_input_ids.shape[1], 1), dtype=self.weight.dtype)
        speaker_state = torch.ones((batch, 1, 1), dtype=self.weight.dtype)
        speaker_mask = (
            ref_mask
            if ref_mask is not None
            else torch.ones((batch, 1), dtype=torch.bool, device=text_input_ids.device)
        )
        return text_state, text_mask, speaker_state, speaker_mask, None, None

    def build_context_kv_cache(
        self,
        *,
        text_state: torch.Tensor,
        speaker_state: torch.Tensor,
        caption_state: torch.Tensor | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        del caption_state
        batch = int(text_state.shape[0])
        text_len = int(text_state.shape[1])
        speaker_len = int(speaker_state.shape[1])
        return [
            (
                torch.ones((batch, text_len, 1, 1), dtype=text_state.dtype),
                torch.full((batch, text_len, 1, 1), 2.0, dtype=text_state.dtype),
                torch.full((batch, speaker_len, 1, 1), 3.0, dtype=text_state.dtype),
                torch.full((batch, speaker_len, 1, 1), 4.0, dtype=text_state.dtype),
            )
        ]


class FakeBackend:
    def __init__(self, branch_layout: str, calls: list[tuple[str, Any]]) -> None:
        self.branch_layout = branch_layout
        self.calls = calls

    def prepare_state(self, payload: Any) -> SimpleNamespace:
        self.calls.append(("prepare", self.branch_layout, payload.batch_size))
        return SimpleNamespace(branch_layout=self.branch_layout)

    def predict_step(
        self,
        prepared_state: Any,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        del prepared_state, t, latent_mask
        self.calls.append(("predict", self.branch_layout, tuple(x_t.shape)))
        return torch.zeros_like(x_t)


class FormulaBackend:
    def __init__(self, branch_layout: str, calls: list[tuple[str, Any]]) -> None:
        self.branch_layout = branch_layout
        self.calls = calls
        self.prepare_kinds = ["cond", "text_uncond", "speaker_uncond"]

    def prepare_state(self, payload: Any) -> SimpleNamespace:
        kind = self.prepare_kinds.pop(0)
        self.calls.append(("prepare", self.branch_layout, kind, payload.batch_size))
        return SimpleNamespace(kind=kind)

    def predict_step(
        self,
        prepared_state: Any,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        del t, latent_mask
        values = {"cond": 10.0, "text_uncond": 4.0, "speaker_uncond": 2.0}
        self.calls.append(("predict", self.branch_layout, prepared_state.kind))
        return torch.full_like(x_t, values[prepared_state.kind])


def test_inference_runtime_fast_path_ignores_metadata_only_mlstate_key_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from irodori_tts import inference_runtime
    from irodori_tts.coreml_cache import (
        BRANCH_LAYOUT_COND1,
        BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
        STATE_LAYOUT_PER_LAYER,
        ConditionCacheHandle,
        expected_per_layer_state_names,
    )

    calls: list[tuple[str, Any]] = []

    def backend_factory(
        runtime: Any,
        *,
        condition_cache: ConditionCacheHandle,
        branch_layout: str,
    ) -> FakeBackend:
        del runtime, condition_cache
        calls.append(("factory", branch_layout))
        return FakeBackend(branch_layout, calls)

    monkeypatch.setattr(
        inference_runtime,
        "COREML_STATEFUL_BACKEND_FACTORY",
        backend_factory,
    )

    runtime = inference_runtime.InferenceRuntime.__new__(inference_runtime.InferenceRuntime)
    runtime.key = inference_runtime.RuntimeKey(checkpoint="tiny", model_device="cpu")
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = TinyModel(num_layers=2).cfg
    runtime.train_cfg = None
    runtime.model = TinyModel(num_layers=2)
    runtime.tokenizer = TinyTokenizer()
    runtime.caption_tokenizer = None
    runtime.codec = TinyCodec()
    runtime.default_text_max_len = 2
    runtime.default_caption_max_len = 2
    runtime._infer_lock = threading.Lock()
    runtime._coreml_stateful_backends = {}

    now = datetime.now(timezone.utc)
    condition_cache = ConditionCacheHandle(
        id="cond_test",
        reference_cache_id="ref_test",
        model_fingerprint="model:test",
        tokenizer_fingerprint="tokenizer:test",
        condition_fingerprint="condition:test",
        bucket_id="S4_T2_R1_independent_text_speaker3",
        state_layout=STATE_LAYOUT_PER_LAYER,
        sequence_length=4,
        text_len=2,
        speaker_context_len=1,
        speaker_context_len_bucket=1,
        c_ctx_bucket=3,
        branch_layouts=(BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3),
        mlstate_keys=expected_per_layer_state_names(),
        created_at=now,
        expires_at=None,
        memory_bytes=0,
        metadata={},
    )

    result = runtime.synthesize_with_condition_cache(
        inference_runtime.SamplingRequest(
            text="hello",
            no_ref=True,
            seconds=2.0,
            num_steps=1,
            seed=123,
            trim_tail=False,
            cfg_min_t=0.0,
            cfg_max_t=1.0,
        ),
        condition_cache=condition_cache,
    )

    assert result.sample_rate == 2
    assert result.used_seed == 123
    assert result.audio.shape == (1, 4)
    assert ("factory", BRANCH_LAYOUT_COND1) in calls
    assert all(call != ("factory", BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3) for call in calls)
    assert calls.count(("prepare", BRANCH_LAYOUT_COND1, 1)) == 3
    assert calls.count(("predict", BRANCH_LAYOUT_COND1, (1, 4, 2))) == 3


def test_coreml_split_cond1_cfg_combines_independent_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from irodori_tts import inference_runtime
    from irodori_tts.coreml_cache import (
        BRANCH_LAYOUT_COND1,
        BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
        STATE_LAYOUT_PER_LAYER,
        ConditionCacheHandle,
        expected_per_layer_state_names,
    )

    calls: list[tuple[str, Any]] = []

    def backend_factory(
        runtime: Any,
        *,
        condition_cache: ConditionCacheHandle,
        branch_layout: str,
    ) -> FormulaBackend:
        del runtime, condition_cache
        calls.append(("factory", branch_layout))
        return FormulaBackend(branch_layout, calls)

    monkeypatch.setattr(
        inference_runtime,
        "COREML_STATEFUL_BACKEND_FACTORY",
        backend_factory,
    )

    runtime = inference_runtime.InferenceRuntime.__new__(inference_runtime.InferenceRuntime)
    runtime.key = inference_runtime.RuntimeKey(checkpoint="tiny", model_device="cpu")
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = TinyModel().cfg
    runtime.train_cfg = None
    runtime.model = TinyModel()
    runtime.tokenizer = TinyTokenizer()
    runtime.caption_tokenizer = None
    runtime.codec = TinyCodec()
    runtime.default_text_max_len = 2
    runtime.default_caption_max_len = 2
    runtime._infer_lock = threading.Lock()
    runtime._coreml_stateful_backends = {}

    now = datetime.now(timezone.utc)
    condition_cache = ConditionCacheHandle(
        id="cond_test",
        reference_cache_id="ref_test",
        model_fingerprint="model:test",
        tokenizer_fingerprint="tokenizer:test",
        condition_fingerprint="condition:test",
        bucket_id="S2_T2_R1_independent_text_speaker3",
        state_layout=STATE_LAYOUT_PER_LAYER,
        sequence_length=2,
        text_len=2,
        speaker_context_len=1,
        speaker_context_len_bucket=1,
        c_ctx_bucket=3,
        branch_layouts=(BRANCH_LAYOUT_COND1, BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3),
        mlstate_keys=expected_per_layer_state_names(),
        created_at=now,
        expires_at=None,
        memory_bytes=0,
        metadata={},
    )

    z = runtime._sample_coreml_stateful_rf_cfg(
        condition_cache=condition_cache,
        text_ids=torch.zeros((1, 2), dtype=torch.long),
        text_mask=torch.ones((1, 2), dtype=torch.bool),
        ref_latent=torch.zeros((1, 1, 2)),
        ref_mask=torch.ones((1, 1), dtype=torch.bool),
        sequence_length=2,
        actual_sequence_length=2,
        num_steps=1,
        cfg_scale_text=2.0,
        cfg_scale_speaker=3.0,
        cfg_min_t=0.0,
        cfg_max_t=1.0,
        seed=123,
        truncation_factor=0.0,
        rescale_k=None,
        rescale_sigma=None,
    )

    torch.testing.assert_close(z, torch.full_like(z, -0.999 * 46.0))
    assert calls == [
        ("factory", BRANCH_LAYOUT_COND1),
        ("prepare", BRANCH_LAYOUT_COND1, "cond", 1),
        ("prepare", BRANCH_LAYOUT_COND1, "text_uncond", 1),
        ("prepare", BRANCH_LAYOUT_COND1, "speaker_uncond", 1),
        ("predict", BRANCH_LAYOUT_COND1, "cond"),
        ("predict", BRANCH_LAYOUT_COND1, "text_uncond"),
        ("predict", BRANCH_LAYOUT_COND1, "speaker_uncond"),
    ]


def test_inference_runtime_cond1_only_rejects_active_independent_cfg() -> None:
    from irodori_tts import inference_runtime
    from irodori_tts.coreml_cache import (
        BRANCH_LAYOUT_COND1,
        STATE_LAYOUT_PER_LAYER,
        ConditionCacheHandle,
        expected_per_layer_state_names,
    )

    runtime = inference_runtime.InferenceRuntime.__new__(inference_runtime.InferenceRuntime)
    runtime.key = inference_runtime.RuntimeKey(checkpoint="tiny", model_device="cpu")
    runtime.model_device = torch.device("cpu")
    runtime.codec_device = torch.device("cpu")
    runtime.model_cfg = TinyModel().cfg
    runtime.train_cfg = None
    runtime.model = TinyModel()
    runtime.tokenizer = TinyTokenizer()
    runtime.caption_tokenizer = None
    runtime.codec = TinyCodec()
    runtime.default_text_max_len = 2
    runtime.default_caption_max_len = 2
    runtime._infer_lock = threading.Lock()
    runtime._coreml_stateful_backends = {}

    now = datetime.now(timezone.utc)
    condition_cache = ConditionCacheHandle(
        id="cond_test",
        reference_cache_id="ref_test",
        model_fingerprint="model:test",
        tokenizer_fingerprint="tokenizer:test",
        condition_fingerprint="condition:test",
        bucket_id="S4_T2_R1_cond1",
        state_layout=STATE_LAYOUT_PER_LAYER,
        sequence_length=4,
        text_len=2,
        speaker_context_len=1,
        speaker_context_len_bucket=1,
        c_ctx_bucket=3,
        branch_layouts=(BRANCH_LAYOUT_COND1,),
        mlstate_keys=expected_per_layer_state_names(),
        created_at=now,
        expires_at=None,
        memory_bytes=0,
        metadata={},
    )

    with pytest.raises(inference_runtime.CoreMLStatefulUnavailableError, match="independent CFG"):
        runtime.synthesize_with_condition_cache(
            inference_runtime.SamplingRequest(
                text="hello",
                no_ref=True,
                seconds=2.0,
                num_steps=1,
                seed=123,
                trim_tail=False,
                cfg_min_t=0.0,
                cfg_max_t=1.0,
            ),
            condition_cache=condition_cache,
        )
